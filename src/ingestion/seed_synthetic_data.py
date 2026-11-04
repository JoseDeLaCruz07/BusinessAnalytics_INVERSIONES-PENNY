"""Generador de datos SINTETICOS (ficticios) para probar el pipeline sin credenciales.

Esto NO reemplaza la ingesta real (ingest_market_data.py / ingest_text_data.py).
Sirve unicamente para que haya algo que mostrar/probar (DB y CSVs limpios) mientras
no se corren los scripts reales contra Finnhub/Yahoo. Los precios, ratios y noticias
generados aqui son inventados con un random walk reproducible (seed fija) y frases de
ejemplo, no informacion de mercado real.

Uso:
    python -m src.ingestion.seed_synthetic_data
    python -m src.ingestion.seed_synthetic_data --tickers SNDL ZOM --days 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import random

import numpy as np

from src.config import DEFAULT_TICKERS
from src.db import PrecioOHLCV, RatioFinanciero, Texto, get_or_create_ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RNG_SEED = 42

SECTORES = ["Cannabis", "Biotecnologia", "Shipping", "Tecnologia", "Mineria"]

HEADLINES_POSITIVAS = [
    "{symbol} anuncia acuerdo estrategico con nuevo socio",
    "{symbol} reporta crecimiento en ingresos trimestrales",
    "Analistas mejoran perspectiva sobre {symbol} tras noticia de expansion",
    "{symbol} recibe aprobacion regulatoria clave para su producto",
]
HEADLINES_NEGATIVAS = [
    "{symbol} enfrenta demanda colectiva de inversionistas",
    "{symbol} reporta perdidas mayores a lo esperado en el trimestre",
    "Analistas rebajan calificacion de {symbol} por riesgo de liquidez",
    "{symbol} anuncia dilucion de acciones para levantar capital",
]
HEADLINES_NEUTRAS = [
    "{symbol} publica su reporte anual 10-K",
    "{symbol} realiza cambios en su directorio ejecutivo",
    "Resumen semanal: como se movio {symbol} en el mercado",
    "{symbol} participa en conferencia del sector",
]

def generar_ohlcv(symbol: str, days: int, rng: np.random.Generator) -> list[dict]:
    precio = rng.uniform(0.5, 4.5)
    hoy = dt.date.today()
    filas = []
    for i in range(days, 0, -1):
        fecha = hoy - dt.timedelta(days=i)
        if fecha.weekday() >= 5:  
            continue
        cambio_pct = rng.normal(0, 0.05)  
        precio = max(0.05, precio * (1 + cambio_pct))
        open_ = precio * (1 + rng.normal(0, 0.01))
        close = precio
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.02)))
        low = min(open_, close) * (1 - abs(rng.normal(0, 0.02)))
        volumen = int(rng.uniform(200_000, 5_000_000))
        filas.append(
            {
                "fecha": fecha,
                "open": round(open_, 4),
                "high": round(high, 4),
                "low": round(max(low, 0.01), 4),
                "close": round(close, 4),
                "volumen": volumen,
            }
        )
    return filas

def generar_ratios(rng: np.random.Generator) -> dict:
    return {
        "market_cap": round(rng.uniform(5, 300), 2),
        "pe_ratio": round(rng.uniform(-20, 40), 2),
        "pb_ratio": round(rng.uniform(0.1, 8), 2),
        "eps": round(rng.uniform(-1.5, 0.5), 4),
        "shares_outstanding": round(rng.uniform(50e6, 800e6), 0),
    }

def generar_noticias(symbol: str, days: int, rng: np.random.Generator, rnd: random.Random) -> list[dict]:
    hoy = dt.date.today()
    n_noticias = rng.integers(4, 10)
    filas = []
    for i in range(n_noticias):
        offset = int(rng.integers(0, days))
        fecha = dt.datetime.combine(hoy - dt.timedelta(days=offset), dt.time(hour=int(rng.integers(8, 20))))
        categoria = rnd.choice(["positiva", "negativa", "neutra"])
        plantilla = rnd.choice(
            {"positiva": HEADLINES_POSITIVAS, "negativa": HEADLINES_NEGATIVAS, "neutra": HEADLINES_NEUTRAS}[categoria]
        )
        headline = plantilla.format(symbol=symbol)
        filas.append(
            {
                "fecha": fecha,
                "contenido": f"{headline}. (Nota generada de forma sintetica para pruebas del pipeline.)",
                "url": f"https://ejemplo-sintetico.local/{symbol.lower()}/{i}",
            }
        )
    return filas

def run(tickers: list[str], days: int) -> None:
    logger.warning(
        "Generando datos SINTETICOS/FICTICIOS (no son precios ni noticias reales) para: %s",
        ", ".join(tickers),
    )
    init_db()
    session = get_session()
    rng = np.random.default_rng(RNG_SEED)
    rnd = random.Random(RNG_SEED)

    for symbol in tickers:
        sector = rnd.choice(SECTORES)
        ticker_obj = get_or_create_ticker(session, symbol=symbol, nombre=f"{symbol} Inc. (demo)", sector=sector)

        ohlcv_rows = generar_ohlcv(symbol, days, rng)
        nuevas, existentes = 0, 0
        for row in ohlcv_rows:
            if session.query(PrecioOHLCV).filter_by(ticker_id=ticker_obj.id, fecha=row["fecha"]).first():
                existentes += 1
                continue
            session.add(PrecioOHLCV(ticker_id=ticker_obj.id, **row))
            nuevas += 1

        hoy = dt.date.today()
        if not session.query(RatioFinanciero).filter_by(ticker_id=ticker_obj.id, fecha=hoy).first():
            session.add(RatioFinanciero(ticker_id=ticker_obj.id, fecha=hoy, **generar_ratios(rng)))

        noticias = generar_noticias(symbol, days, rng, rnd)
        nuevas_textos = 0
        for item in noticias:
            if session.query(Texto).filter_by(ticker_id=ticker_obj.id, url=item["url"]).first():
                continue
            session.add(Texto(ticker_id=ticker_obj.id, fuente="noticia", **item))
            nuevas_textos += 1

        session.commit()
        logger.info(
            "%s: %d precios nuevos (%d ya existian), %d noticias sinteticas",
            symbol,
            nuevas,
            existentes,
            nuevas_textos,
        )

    session.close()
    logger.warning("Listo. Recuerda: estos datos son ficticios, solo para probar el pipeline end-to-end.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera datos sinteticos para probar el pipeline sin red/credenciales")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=90)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.tickers, args.days)