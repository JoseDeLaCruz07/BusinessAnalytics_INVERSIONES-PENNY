"""Fase 1 - Ingesta de datos de mercado (OHLCV, market cap, ratios financieros).

Fuentes: yfinance (OHLCV) + finnhub-python (perfil de compania y ratios).
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
from src.db import PrecioOHLCV, RatioFinanciero, get_or_create_ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_ohlcv(symbol: str, days: int) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
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
    return df[["fecha", "open", "high", "low", "close", "volumen"]]


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

        ohlcv_df = fetch_ohlcv(symbol, days)
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

        for _, row in ohlcv_df.iterrows():
            fecha = row["fecha"].date() if hasattr(row["fecha"], "date") else row["fecha"]
            exists = (
                session.query(PrecioOHLCV)
                .filter_by(ticker_id=ticker_obj.id, fecha=fecha)
                .first()
            )
            if exists:
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
    parser.add_argument("--days", type=int, default=90, help="Dias de historico OHLCV a descargar")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tickers, args.days)
