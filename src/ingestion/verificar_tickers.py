"""Valida tickers candidatos contra yfinance (existencia, penny stock e historial) antes de agregarlos a DEFAULT_TICKERS.

Uso:
    python -m src.ingestion.verificar_tickers KOPN EGHT MDXG TICKER_INVENTADO
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import yfinance as yf

from src.config import PENNY_STOCK_PRICE_THRESHOLD

logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # silenciar el ruido de tickers invalidos

HISTORIAL_DIAS = 730  # misma ventana que usa la ingesta con --days 730
MIN_DIAS_TRADING = 400
MIN_VOLUMEN_PROMEDIO = 1_000_000
MAX_PCT_SIN_VOLUMEN = 0.05


def verificar(symbol: str) -> dict:
    fin = dt.date.today()
    inicio = fin - dt.timedelta(days=HISTORIAL_DIAS)
    try:
        df = yf.download(symbol, start=inicio, end=fin, progress=False, auto_adjust=False)
    except Exception as e:
        return {"symbol": symbol, "valido": False, "motivo": str(e)}

    if df.empty:
        return {"symbol": symbol, "valido": False, "motivo": "sin datos (posible delisting o simbolo invalido)"}

    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)

    cierres = df.dropna(subset=["Close"])
    if cierres.empty:
        return {"symbol": symbol, "valido": False, "motivo": "todos los cierres son NaN"}

    ultimo_close = float(cierres["Close"].iloc[-1])
    return {
        "symbol": symbol,
        "valido": True,
        "ultimo_close": round(ultimo_close, 2),
        "es_penny_stock": ultimo_close < PENNY_STOCK_PRICE_THRESHOLD,
        "dias": len(cierres),
        "desde": cierres.index[0].date(),
        "vol_prom": float(cierres["Volume"].mean()),
        "pct_sin_volumen": float((cierres["Volume"].fillna(0) <= 0).mean()),
    }


def motivos_no_apto(r: dict) -> list[str]:
    motivos = []
    if not r["es_penny_stock"]:
        motivos.append("no es penny")
    if r["dias"] < MIN_DIAS_TRADING:
        motivos.append(f"poca historia (<{MIN_DIAS_TRADING} dias)")
    if r["vol_prom"] < MIN_VOLUMEN_PROMEDIO:
        motivos.append("volumen bajo")
    if r["pct_sin_volumen"] > MAX_PCT_SIN_VOLUMEN:
        motivos.append("muchos dias sin volumen")
    return motivos


def run(tickers: list[str]) -> None:
    print(f"{'TICKER':<9}{'CLOSE':<8}{'PENNY':<7}{'DIAS':<6}{'DESDE':<12}{'VOL PROM':<12}{'SIN VOL':<9}APTO")
    print("-" * 80)
    for symbol in tickers:
        r = verificar(symbol)
        if not r["valido"]:
            print(f"{r['symbol']:<9}NO VALIDO: {r['motivo']}")
            continue
        motivos = motivos_no_apto(r)
        veredicto = "SI" if not motivos else "NO (" + ", ".join(motivos) + ")"
        print(
            f"{r['symbol']:<9}{r['ultimo_close']:<8}{'si' if r['es_penny_stock'] else 'no':<7}"
            f"{r['dias']:<6}{str(r['desde']):<12}{r['vol_prom']:<12,.0f}{r['pct_sin_volumen']:<9.1%}{veredicto}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida tickers candidatos contra yfinance")
    parser.add_argument("tickers", nargs="+", help="Simbolos a probar, ej: KOPN EGHT AAPL")
    args = parser.parse_args()
    run(args.tickers)
