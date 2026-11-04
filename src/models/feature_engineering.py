"""Fase 2 - Feature engineering: indicadores tecnicos + sentimiento agregado.

Combina RSI, media movil y volumen relativo (calculados sobre precios_ohlcv)
con el sentimiento agregado por ticker/dia (calculado sobre textos+sentimiento,
que produce sentiment_finbert.py) en un feature set unico, guardado en la
tabla `features` y exportado a data/features_export.csv.

Si todavia no se corrio sentiment_finbert.py, el sentimiento_agregado queda en
0.0 (neutral) para no bloquear el resto del pipeline.

Uso:
    python -m src.models.feature_engineering
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

from src.config import RSI_WINDOW, SMA_WINDOW, VOLUME_RELATIVE_WINDOW
from src.db import Feature, PrecioOHLCV, Sentimiento, Texto, Ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Traduce la etiqueta de FinBERT a un signo, para poder promediar sentimiento por dia.
ETIQUETA_SIGNO = {"positivo": 1, "negativo": -1, "neutral": 0}


def cargar_ohlcv(session) -> pd.DataFrame:
    rows = (
        session.query(PrecioOHLCV, Ticker.symbol)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .order_by(Ticker.symbol, PrecioOHLCV.fecha)
        .all()
    )
    df = pd.DataFrame(
        [
            {"ticker_id": p.ticker_id, "symbol": symbol, "fecha": pd.Timestamp(p.fecha), "close": p.close, "volumen": p.volumen}
            for p, symbol in rows
        ]
    )
    return df


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

    features_df = ohlcv.dropna(subset=["rsi", "media_movil", "volumen_relativo"]).copy()

    nuevas = 0
    for _, row in features_df.iterrows():
        fecha = row["fecha"].date()
        existe = session.query(Feature).filter_by(ticker_id=int(row["ticker_id"]), fecha=fecha).first()
        if existe:
            existe.rsi = float(row["rsi"])
            existe.media_movil = float(row["media_movil"])
            existe.volumen_relativo = float(row["volumen_relativo"])
            existe.sentimiento_agregado = float(row["sentimiento_agregado"])
            continue
        session.add(
            Feature(
                ticker_id=int(row["ticker_id"]),
                fecha=fecha,
                rsi=float(row["rsi"]),
                media_movil=float(row["media_movil"]),
                volumen_relativo=float(row["volumen_relativo"]),
                sentimiento_agregado=float(row["sentimiento_agregado"]),
            )
        )
        nuevas += 1

    session.commit()
    session.close()

    DATA_DIR.mkdir(exist_ok=True)
    export_cols = ["symbol", "fecha", "close", "rsi", "media_movil", "volumen_relativo", "sentimiento_agregado"]
    features_df[export_cols].to_csv(DATA_DIR / "features_export.csv", index=False)

    logger.info("Features: %d filas totales (%d nuevas) -> data/features_export.csv", len(features_df), nuevas)
    return features_df


if __name__ == "__main__":
    run()
