"""Fase 1 - Ingesta de mercado: OHLCV (yfinance) + market cap/ratios (Finnhub).

Precios ajustados por splits: Yahoo reescribe todo el historico cuando una accion hace un split o un reverse
split (consolidation). Por eso se descarga desde la primera fecha guardada, se compara con lo que ya hay en la
base y, si algun cierre guardado difiere mas de UMBRAL_REESCRITURA, se reemplaza el historial completo del ticker.
Asi la base nunca queda mezclando dos escalas de precio (el "+600%" falso de RETO del 30 oct 2026).

Uso:
    python -m src.ingestion.ingest_market_data
    python -m src.ingestion.ingest_market_data --tickers SNDL ZOM --days 90
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import finnhub
import pandas as pd
import yfinance as yf

from src.config import DEFAULT_TICKERS, FINNHUB_API_KEY, PENNY_STOCK_PRICE_THRESHOLD
from src.db import PrecioOHLCV, RatioFinanciero, Ticker, get_or_create_ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Diferencia relativa entre el cierre guardado y el que devuelve Yahoo a partir de la cual se reescribe el historial.
UMBRAL_REESCRITURA = 0.01
# Un cambio diario de +-60% casi siempre es un split mal ajustado; se avisa en el log para revisarlo a mano.
SALTO_SOSPECHOSO = 0.60


def fetch_ohlcv(symbol: str, days: int, desde: dt.date | None = None) -> pd.DataFrame:
    """OHLCV ajustado por splits desde `desde` (o los ultimos `days` dias, lo que sea mas antiguo) hasta hoy inclusive."""
    hoy = dt.date.today()
    start = hoy - dt.timedelta(days=days)
    if desde is not None and desde < start:
        start = desde
    # yfinance excluye la fecha `end`: con end=hoy nunca llegaba la vela del dia.
    end = hoy + dt.timedelta(days=1)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        logger.warning("yfinance no devolvio datos OHLCV para %s", symbol)
        return df
    # yfinance puede devolver columnas MultiIndex cuando se pide un solo ticker en algunas versiones
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(
        columns={
            "Date": "fecha",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volumen",
        }
    )
    df = df[["fecha", "open", "high", "low", "close", "volumen"]]

    # yfinance devuelve filas con OHLC en NaN en dias de volumen casi nulo (comun en penny stocks).
    filas_invalidas = df[["open", "high", "low", "close"]].isna().any(axis=1)
    if filas_invalidas.any():
        logger.warning("%s: %d fila(s) con OHLC invalido descartadas", symbol, filas_invalidas.sum())
        df = df[~filas_invalidas]

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    return df.reset_index(drop=True)


def fechas_desfasadas(guardados: dict[dt.date, float], nuevo_df: pd.DataFrame, umbral: float = UMBRAL_REESCRITURA) -> list[dt.date]:
    """Fechas en las que el cierre guardado ya no coincide con el ajustado que devuelve Yahoo (split posterior)."""
    desfasadas = []
    for fecha, close in zip(nuevo_df["fecha"], nuevo_df["close"]):
        guardado = guardados.get(fecha)
        if guardado is None or not close:
            continue
        if abs(guardado / float(close) - 1) > umbral:
            desfasadas.append(fecha)
    return desfasadas


def saltos_sospechosos(df: pd.DataFrame, umbral: float = SALTO_SOSPECHOSO) -> pd.DataFrame:
    """Dias con un cambio de cierre mayor al umbral respecto del dia anterior."""
    ordenado = df.sort_values("fecha").reset_index(drop=True)
    cambio = ordenado["close"].pct_change(fill_method=None)
    salida = ordenado.loc[cambio.abs() > umbral, ["fecha", "close"]].copy()
    salida["cambio"] = cambio[cambio.abs() > umbral]
    return salida


def fetch_finnhub_profile_and_ratios(client: finnhub.Client, symbol: str) -> tuple[dict, dict]:
    profile = client.company_profile2(symbol=symbol) or {}
    financials = client.company_basic_financials(symbol, "all") or {}
    metric = financials.get("metric", {}) or {}
    ratios = {
        "market_cap": profile.get("marketCapitalization"),
        "pe_ratio": metric.get("peNormalizedAnnual") or metric.get("peTTM"),
        "pb_ratio": metric.get("pbAnnual") or metric.get("pbQuarterly"),
        "eps": metric.get("epsNormalizedAnnual") or metric.get("epsTTM"),
        "shares_outstanding": profile.get("shareOutstanding"),
    }
    return profile, ratios


def run(tickers: list[str], days: int) -> None:
    if not FINNHUB_API_KEY:
        logger.error(
            "FINNHUB_API_KEY no esta configurada. Copia .env.example a .env y completa la key."
        )
        sys.exit(1)

    init_db()
    session = get_session()
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    DATA_DIR.mkdir(exist_ok=True)

    all_ohlcv_rows = []
    all_ratio_rows = []

    for symbol in tickers:
        logger.info("Procesando %s", symbol)

        ticker_existente = session.query(Ticker).filter_by(symbol=symbol).first()
        guardados: dict[dt.date, float] = {}
        if ticker_existente is not None:
            guardados = dict(
                session.query(PrecioOHLCV.fecha, PrecioOHLCV.close).filter_by(ticker_id=ticker_existente.id).all()
            )

        # Se pide desde la primera fecha guardada para poder comparar todo el historial con la version ajustada.
        ohlcv_df = fetch_ohlcv(symbol, days, desde=min(guardados) if guardados else None)
        if ohlcv_df.empty:
            continue

        last_close = float(ohlcv_df.iloc[-1]["close"])
        is_penny = last_close < PENNY_STOCK_PRICE_THRESHOLD
        logger.info(
            "%s ultimo cierre=%.2f (%s penny stock, umbral=%.2f)",
            symbol,
            last_close,
            "es" if is_penny else "NO es",
            PENNY_STOCK_PRICE_THRESHOLD,
        )

        try:
            profile, ratios = fetch_finnhub_profile_and_ratios(finnhub_client, symbol)
        except Exception:
            logger.exception("Fallo Finnhub para %s, se continua sin ratios", symbol)
            profile, ratios = {}, {}

        ticker_obj = get_or_create_ticker(
            session,
            symbol=symbol,
            nombre=profile.get("name"),
            sector=profile.get("finnhubIndustry"),
        )

        desfasadas = fechas_desfasadas(guardados, ohlcv_df)
        if desfasadas:
            logger.warning(
                "%s: %d cierre(s) guardado(s) no coinciden con el precio ajustado de Yahoo (primera fecha %s); "
                "probable split o reverse split. Se reemplaza el historial completo del ticker.",
                symbol, len(desfasadas), min(desfasadas),
            )
            session.query(PrecioOHLCV).filter_by(ticker_id=ticker_obj.id).delete(synchronize_session=False)
            guardados = {}

        for _, salto in saltos_sospechosos(ohlcv_df).iterrows():
            logger.warning(
                "%s: el %s el cierre cambio %+.0f%% en un dia (cierre %.4f); revisar si es un split sin ajustar.",
                symbol, salto["fecha"], salto["cambio"] * 100, salto["close"],
            )

        for _, row in ohlcv_df.iterrows():
            fecha = row["fecha"]
            if fecha in guardados:
                continue
            session.add(
                PrecioOHLCV(
                    ticker_id=ticker_obj.id,
                    fecha=fecha,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volumen=int(row["volumen"]) if pd.notna(row["volumen"]) else None,
                )
            )
            all_ohlcv_rows.append({"symbol": symbol, "fecha": fecha, **row.drop("fecha").to_dict()})

        hoy = dt.date.today()
        exists_ratio = (
            session.query(RatioFinanciero)
            .filter_by(ticker_id=ticker_obj.id, fecha=hoy)
            .first()
        )
        if not exists_ratio and ratios:
            session.add(RatioFinanciero(ticker_id=ticker_obj.id, fecha=hoy, **ratios))
            all_ratio_rows.append({"symbol": symbol, "fecha": hoy, **ratios})

        session.commit()

    if all_ohlcv_rows:
        pd.DataFrame(all_ohlcv_rows).to_csv(DATA_DIR / "ohlcv_export.csv", index=False)
        logger.info("Exportado data/ohlcv_export.csv (%d filas)", len(all_ohlcv_rows))
    if all_ratio_rows:
        pd.DataFrame(all_ratio_rows).to_csv(DATA_DIR / "ratios_export.csv", index=False)
        logger.info("Exportado data/ratios_export.csv (%d filas)", len(all_ratio_rows))

    session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de datos de mercado para penny stocks")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=90, help="Dias minimos de historico OHLCV (si ya hay datos se descarga desde la primera fecha guardada)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tickers, args.days)
