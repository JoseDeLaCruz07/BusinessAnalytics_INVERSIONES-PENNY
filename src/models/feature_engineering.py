"""Fase 2 - Feature engineering: RSI, media movil, volumen relativo, sentimiento, volatilidad y precio relativo.

Uso:
    python -m src.models.feature_engineering
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from src.config import RSI_WINDOW, SMA_WINDOW, VOLUME_RELATIVE_WINDOW
from src.db import Feature, PrecioOHLCV, Sentimiento, Texto, Ticker, get_session, init_db
from src.processing.clean_pipeline import limpiar_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Traduce la etiqueta de FinBERT a un signo, para poder promediar sentimiento por dia.
ETIQUETA_SIGNO = {"positivo": 1, "negativo": -1, "neutral": 0}

# Columnas de la tabla `features` que se calculan aqui (sentimiento_agregado se une aparte).
COLUMNAS_FEATURE = [
    "rsi", "media_movil", "volumen_relativo", "sentimiento_agregado", "precio_vs_sma", "vol_10d", "vol_20d", "rango_5d",
]


LOTE_COMMIT = 500
TOLERANCIA_RELATIVA = 1e-9


def valor_distinto(guardado, nuevo) -> bool:
    """True si el valor cambio mas alla del ruido de redondeo de punto flotante (NaN/None cuentan como iguales entre si)."""
    vacio_g = guardado is None or (isinstance(guardado, float) and math.isnan(guardado))
    vacio_n = nuevo is None or (isinstance(nuevo, float) and math.isnan(nuevo))
    if vacio_g or vacio_n:
        return vacio_g != vacio_n
    return not math.isclose(float(guardado), float(nuevo), rel_tol=TOLERANCIA_RELATIVA, abs_tol=1e-12)


def cargar_ohlcv(session) -> pd.DataFrame:
    rows = (
        session.query(PrecioOHLCV, Ticker.symbol)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .order_by(Ticker.symbol, PrecioOHLCV.fecha)
        .all()
    )
    df = pd.DataFrame(
        [
            {"ticker_id": p.ticker_id, "symbol": symbol, "fecha": pd.Timestamp(p.fecha), "high": p.high, "low": p.low, "close": p.close, "volumen": p.volumen}
            for p, symbol in rows
        ]
    )
    return limpiar_ohlcv(df)


def cargar_sentimiento_diario(session) -> pd.DataFrame:
    rows = (
        session.query(Texto.ticker_id, Texto.fecha, Sentimiento.score, Sentimiento.etiqueta)
        .join(Sentimiento, Sentimiento.texto_id == Texto.id)
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["ticker_id", "fecha", "sentimiento_agregado"])

    df = pd.DataFrame(rows, columns=["ticker_id", "fecha", "score", "etiqueta"])
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    df["score_firmado"] = df["score"] * df["etiqueta"].map(ETIQUETA_SIGNO).fillna(0)

    diario = (
        df.groupby(["ticker_id", "fecha"])["score_firmado"]
        .mean()
        .reset_index()
        .rename(columns={"score_firmado": "sentimiento_agregado"})
    )
    return diario


def calcular_indicadores(grupo: pd.DataFrame) -> pd.DataFrame:
    grupo = grupo.sort_values("fecha").copy()
    grupo["rsi"] = RSIIndicator(close=grupo["close"], window=RSI_WINDOW).rsi()
    grupo["media_movil"] = SMAIndicator(close=grupo["close"], window=SMA_WINDOW).sma_indicator()
    volumen_promedio = grupo["volumen"].rolling(window=VOLUME_RELATIVE_WINDOW, min_periods=1).mean()
    grupo["volumen_relativo"] = grupo["volumen"] / volumen_promedio.replace(0, pd.NA)
    # Escala-libres (no delatan el nivel de precio del ticker): se miden contra el propio precio.
    grupo["precio_vs_sma"] = grupo["close"] / grupo["media_movil"] - 1
    retorno = grupo["close"].pct_change(fill_method=None)
    grupo["vol_10d"] = retorno.rolling(10).std()
    grupo["vol_20d"] = retorno.rolling(20).std()
    grupo["rango_5d"] = ((grupo["high"] - grupo["low"]) / grupo["close"]).rolling(5).mean()
    return grupo


def run() -> pd.DataFrame:
    init_db()
    session = get_session()

    ohlcv = cargar_ohlcv(session)
    if ohlcv.empty:
        logger.warning("No hay precios en la base de datos. Corre primero la ingesta (real o sintetica).")
        session.close()
        return pd.DataFrame()

    ohlcv = ohlcv.groupby("symbol", group_keys=False)[ohlcv.columns].apply(calcular_indicadores)

    sentimiento = cargar_sentimiento_diario(session)
    if sentimiento.empty:
        logger.warning(
            "No hay sentimiento calculado todavia (corre sentiment_finbert.py). "
            "sentimiento_agregado se deja en 0.0 (neutral) por ahora."
        )
        ohlcv["sentimiento_agregado"] = 0.0
    else:
        ohlcv = ohlcv.merge(sentimiento, on=["ticker_id", "fecha"], how="left")
        ohlcv["sentimiento_agregado"] = ohlcv["sentimiento_agregado"].fillna(0.0)

    features_df = ohlcv.dropna(subset=[c for c in COLUMNAS_FEATURE if c != "sentimiento_agregado"]).copy()

    # Una sola consulta en vez de una por fila: contra Supabase cada consulta es un viaje de red.
    existentes = {(f.ticker_id, f.fecha): f for f in session.query(Feature).all()}

    nuevas = actualizadas = pendientes = 0
    for _, row in features_df.iterrows():
        fecha = row["fecha"].date()
        existe = existentes.get((int(row["ticker_id"]), fecha))
        valores = {c: float(row[c]) for c in COLUMNAS_FEATURE}
        if existe:
            # Solo se reescribe lo que cambio de verdad: la base devuelve los floats redondeados en el decimal 15, y
            # comparar exacto hacia reescribir las ~9.000 filas en cada corrida (Supabase cortaba por statement timeout).
            cambios = {c: v for c, v in valores.items() if valor_distinto(getattr(existe, c), v)}
            if not cambios:
                continue
            for columna, valor in cambios.items():
                setattr(existe, columna, valor)
            actualizadas += 1
        else:
            session.add(Feature(ticker_id=int(row["ticker_id"]), fecha=fecha, **valores))
            nuevas += 1
        pendientes += 1
        if pendientes >= LOTE_COMMIT:  # commits por lotes: ninguna transaccion queda demasiado larga
            session.commit()
            pendientes = 0

    session.commit()
    session.close()
    logger.info("Features: %d nuevas y %d actualizadas (el resto sin cambios)", nuevas, actualizadas)

    DATA_DIR.mkdir(exist_ok=True)
    export_cols = ["symbol", "fecha", "close"] + COLUMNAS_FEATURE
    features_df[export_cols].to_csv(DATA_DIR / "features_export.csv", index=False)

    logger.info("Features: %d filas totales (%d nuevas) -> data/features_export.csv", len(features_df), nuevas)
    return features_df


if __name__ == "__main__":
    run()
