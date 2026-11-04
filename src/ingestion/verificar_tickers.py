"""Utilidad standalone: valida una lista de tickers candidatos contra yfinance.

No escribe nada en la base de datos. Sirve para filtrar, antes de agregarlos
a DEFAULT_TICKERS, cuales simbolos siguen activos y son realmente penny
stocks (precio < PENNY_STOCK_PRICE_THRESHOLD) hoy.

Uso:
    python -m src.ingestion.verificar_tickers AAPL SNDL CTRM TICKER_INVENTADO
"""
from __future__ import annotations

import argparse
import logging

import yfinance as yf

from src.config import PENNY_STOCK_PRICE_THRESHOLD

logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # silenciar el ruido de tickers invalidos


def verificar(symbol: str) -> dict:
    try:
        df = yf.download(symbol, period="5d", progress=False, auto_adjust=False)
    except Exception as e:
        return {"symbol": symbol, "valido": False, "motivo": str(e)}

    if df.empty:
        return {"symbol": symbol, "valido": False, "motivo": "sin datos (posible delisting o simbolo invalido)"}

    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    ultimo_close = float(df["Close"].iloc[-1])
    es_penny = ultimo_close < PENNY_STOCK_PRICE_THRESHOLD
    return {"symbol": symbol, "valido": True, "ultimo_close": round(ultimo_close, 2), "es_penny_stock": es_penny}


def run(tickers: list[str]) -> None:
    print(f"{'TICKER':<10}{'VALIDO':<10}{'CLOSE':<10}{'PENNY?':<10}DETALLE")
    print("-" * 60)
    for symbol in tickers:
        r = verificar(symbol)
        if r["valido"]:
            print(f"{r['symbol']:<10}{'si':<10}{r['ultimo_close']:<10}{'si' if r['es_penny_stock'] else 'no':<10}")
        else:
            print(f"{r['symbol']:<10}{'NO':<10}{'':<10}{'':<10}{r['motivo']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida tickers candidatos contra yfinance")
    parser.add_argument("tickers", nargs="+", help="Simbolos a probar, ej: SNDL CTRM AAPL")
    args = parser.parse_args()
    run(args.tickers)
