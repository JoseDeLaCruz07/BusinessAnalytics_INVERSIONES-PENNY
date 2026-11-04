"""Fase 1 - Ingesta de datos de texto (no estructurados).

Fuentes activas:
- 'noticia' via Finnhub News API.
- 'stocktwits' via la API publica de StockTwits (sin autenticacion), como
  sustituto de Reddit mientras no hay credenciales de PRAW. Da texto crudo
  real (mensajes de usuarios sobre el ticker).
  Pagina hacia atras (hasta 5 paginas, ~150 mensajes/ticker) hasta cubrir
  la ventana de --days o quedarse sin mensajes mas antiguos.

La funcion fetch_reddit_posts() queda como stub: cuando existan
REDDIT_CLIENT_ID/SECRET, se completa con praw sin tocar el resto del pipeline.

Uso:
    python -m src.ingestion.ingest_text_data
    python -m src.ingestion.ingest_text_data --tickers SNDL ZOM --days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from pathlib import Path

import finnhub
import pandas as pd
import requests

from src.config import (
    DEFAULT_TICKERS,
    FINNHUB_API_KEY,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
)
from src.db import Texto, get_or_create_ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

STOCKTWITS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

def fetch_finnhub_news(client: finnhub.Client, symbol: str, days: int) -> list[dict]:
    to_date = dt.date.today()
    from_date = to_date - dt.timedelta(days=days)
    news = client.company_news(symbol, _from=from_date.isoformat(), to=to_date.isoformat())
    return news or []

def fetch_stocktwits_messages(symbol: str, days: int, max_paginas: int = 5) -> list[dict]:
    url = STOCKTWITS_URL.format(symbol=symbol)
    limite = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    filas = []
    max_id = None

    for pagina in range(max_paginas):
        params = {"max": max_id} if max_id else {}
        resp = requests.get(url, headers=STOCKTWITS_HEADERS, params=params, timeout=15)
        if resp.status_code == 404:
            logger.info("StockTwits no tiene stream para %s (simbolo no encontrado en esa plataforma)", symbol)
            return filas
        resp.raise_for_status()
        data = resp.json()

        mensajes = data.get("messages", [])
        if not mensajes:
            break

        llego_al_limite = False
        for msg in mensajes:
            creado = dt.datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00"))
            if creado < limite:
                llego_al_limite = True
                continue
            cuerpo = (msg.get("body") or "").strip()
            if not cuerpo:
                continue
            filas.append(
                {
                    "fecha": creado.replace(tzinfo=None),
                    "contenido": cuerpo,
                    "url": f"https://stocktwits.com/symbol/{symbol}/message/{msg['id']}",
                }
            )

        if llego_al_limite or not data.get("cursor", {}).get("more"):
            break

        max_id = data["cursor"].get("max") or mensajes[-1]["id"] - 1
        time.sleep(0.5) 

    return filas

def fetch_reddit_posts(symbol: str, days: int) -> list[dict]:
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        logger.info("Reddit no configurado todavia; se omite esta fuente para %s", symbol)
        return []
    raise NotImplementedError("Ingesta de Reddit pendiente de implementar (PRAW).")

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

    all_rows = []

    for symbol in tickers:
        logger.info("Descargando texto (noticias + StockTwits) para %s", symbol)
        ticker_obj = get_or_create_ticker(session, symbol=symbol)

        try:
            news_items = fetch_finnhub_news(finnhub_client, symbol, days)
        except Exception:
            logger.exception("Fallo Finnhub News para %s", symbol)
            news_items = []

        try:
            stocktwits_items = fetch_stocktwits_messages(symbol, days)
        except Exception:
            logger.exception("Fallo StockTwits para %s", symbol)
            stocktwits_items = []

        reddit_items = fetch_reddit_posts(symbol, days)

        for item in news_items:
            fecha = dt.datetime.fromtimestamp(item.get("datetime", 0))
            contenido = f"{item.get('headline', '')}. {item.get('summary', '')}".strip()
            if not contenido:
                continue
            url = item.get("url", "")
            exists = (
                session.query(Texto)
                .filter_by(ticker_id=ticker_obj.id, url=url, fuente="noticia")
                .first()
            )
            if exists:
                continue
            session.add(
                Texto(
                    ticker_id=ticker_obj.id,
                    fuente="noticia",
                    fecha=fecha,
                    contenido=contenido,
                    url=url,
                )
            )
            all_rows.append(
                {"symbol": symbol, "fuente": "noticia", "fecha": fecha, "contenido": contenido, "url": url}
            )

        for item in stocktwits_items:
            exists = (
                session.query(Texto)
                .filter_by(ticker_id=ticker_obj.id, url=item["url"], fuente="stocktwits")
                .first()
            )
            if exists:
                continue
            session.add(
                Texto(
                    ticker_id=ticker_obj.id,
                    fuente="stocktwits",
                    fecha=item["fecha"],
                    contenido=item["contenido"],
                    url=item["url"],
                )
            )
            all_rows.append({"symbol": symbol, "fuente": "stocktwits", **item})

        for item in reddit_items:
            all_rows.append({"symbol": symbol, "fuente": "reddit", **item})

        session.commit()

    if all_rows:
        pd.DataFrame(all_rows).to_csv(DATA_DIR / "textos_export.csv", index=False)
        logger.info("Exportado data/textos_export.csv (%d filas)", len(all_rows))
    else:
        logger.warning("No se recolectaron textos en esta corrida.")

    session.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de texto (noticias / reddit) para penny stocks")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=30, help="Dias hacia atras de noticias a traer")
    return parser.parseargs()

if __name__ == "__main__":
    args = parse_args()
    run(args.tickers, args.days)