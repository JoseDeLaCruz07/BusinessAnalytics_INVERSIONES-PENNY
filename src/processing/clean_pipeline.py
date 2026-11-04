"""Fase 1 - Limpieza de datos numericos (OHLCV, ratios) y de texto (noticias/reddit).

Lee desde la base de datos (ya poblada por los scripts de ingesta), aplica
limpieza y exporta versiones limpias a data/*_clean.csv. No modifica las
tablas crudas: la limpieza es un paso de transformacion hacia feature_engineering.

Uso:
    python -m src.processing.clean_pipeline
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.db import PrecioOHLCV, RatioFinanciero, Texto, Ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_ohlcv(session) -> pd.DataFrame:
    rows = (
        session.query(PrecioOHLCV, Ticker.symbol)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "fecha": p.fecha,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volumen": p.volumen,
            }
            for p, symbol in rows
        ]
    )
    if df.empty:
        logger.warning("No hay datos OHLCV en la base de datos todavia.")
        return df

    df["fecha"] = pd.to_datetime(df["fecha"])
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["volumen"] = df["volumen"].fillna(0)

    # Outliers de precio: recorte por grupo (symbol) usando IQR sobre 'close'.
    def clip_outliers(group: pd.DataFrame) -> pd.DataFrame:
        q1, q3 = group["close"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        group["close"] = group["close"].clip(lower=max(lower, 0), upper=upper if iqr > 0 else group["close"].max())
        return group

    df = df.groupby("symbol", group_keys=False)[df.columns].apply(clip_outliers)
    df = df.sort_values(["symbol", "fecha"]).drop_duplicates(subset=["symbol", "fecha"])

    logger.info("OHLCV: %d filas crudas -> %d filas limpias", before, len(df))
    return df


def clean_ratios(session) -> pd.DataFrame:
    rows = (
        session.query(RatioFinanciero, Ticker.symbol)
        .join(Ticker, RatioFinanciero.ticker_id == Ticker.id)
        .all()
    )
    df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "fecha": r.fecha,
                "market_cap": r.market_cap,
                "pe_ratio": r.pe_ratio,
                "pb_ratio": r.pb_ratio,
                "eps": r.eps,
                "shares_outstanding": r.shares_outstanding,
            }
            for r, symbol in rows
        ]
    )
    if df.empty:
        logger.warning("No hay ratios financieros en la base de datos todavia.")
        return df

    df["fecha"] = pd.to_datetime(df["fecha"])
    # Ratios negativos o cero en pe/pb no tienen sentido de negocio -> se marcan como NA en vez de descartar la fila.
    for col in ["pe_ratio", "pb_ratio"]:
        df.loc[df[col] <= 0, col] = pd.NA

    logger.info("Ratios financieros: %d filas procesadas", len(df))
    return df


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_textos(session) -> pd.DataFrame:
    rows = (
        session.query(Texto, Ticker.symbol)
        .join(Ticker, Texto.ticker_id == Ticker.id)
        .all()
    )
    records = []
    for texto, symbol in rows:
        contenido_limpio = clean_text(texto.contenido)
        if len(contenido_limpio) < 10:
            continue  # ruido: contenido casi vacio tras limpieza
        records.append(
            {
                "symbol": symbol,
                "fuente": texto.fuente,
                "fecha": texto.fecha,
                "contenido": contenido_limpio,
                "url": texto.url,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No hay textos en la base de datos todavia.")
        return df

    df = df.drop_duplicates(subset=["symbol", "fuente", "contenido"])
    logger.info("Textos: %d filas limpias (fuentes: %s)", len(df), df["fuente"].unique().tolist())
    return df


def run() -> None:
    init_db()
    session = get_session()
    DATA_DIR.mkdir(exist_ok=True)

    ohlcv_clean = clean_ohlcv(session)
    ratios_clean = clean_ratios(session)
    textos_clean = clean_textos(session)

    if not ohlcv_clean.empty:
        ohlcv_clean.to_csv(DATA_DIR / "ohlcv_clean.csv", index=False)
    if not ratios_clean.empty:
        ratios_clean.to_csv(DATA_DIR / "ratios_clean.csv", index=False)
    if not textos_clean.empty:
        textos_clean.to_csv(DATA_DIR / "textos_clean.csv", index=False)

    session.close()


if __name__ == "__main__":
    run()
