"""Prueba del asesor IA con Gemini real y datos reales, SIN guardar: lee la base en solo lectura e imprime el resultado.

Uso:
    python -m src.reports.probar_asesor --tickers RETO PDSB ABUS
Guarda la entrada y la salida en docs/evidencias/asesor_prueba_YYYYMMDD.json (no toca la base).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

import xgboost as xgb

from src.config import GEMINI_API_KEY, GEMINI_MODEL, TARGET_HORIZON_DAYS
from src.db import Ticker
from src.models.backtesting import EVIDENCIAS_DIR, sesion_solo_lectura
from src.models.train_xgboost import FEATURE_COLS, PUBLISHED_MODEL_PATH, construir_dataset, evaluar_walk_forward
from src.reports.asesor_ia import (
    PAUSA_ENTRE_LLAMADAS,
    _puestos,
    armar_prompt,
    construir_hechos,
    generar_texto,
    llamar_gemini,
    validar_respuesta,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["RETO", "PDSB", "ABUS"])
    args = parser.parse_args()
    if not GEMINI_API_KEY:
        raise SystemExit("Falta GEMINI_API_KEY en .env")

    session = sesion_solo_lectura()
    modelo = xgb.XGBClassifier()
    modelo.load_model(PUBLISHED_MODEL_PATH)
    columnas = list(modelo.get_booster().feature_names or FEATURE_COLS)
    confiabilidad = evaluar_walk_forward(construir_dataset(session), FEATURE_COLS, purga_dias=TARGET_HORIZON_DAYS) or {}
    ranking = _puestos(session, modelo, columnas)
    generador = lambda prompt: llamar_gemini(prompt, GEMINI_MODEL, GEMINI_API_KEY)  # noqa: E731

    resultados = []
    for symbol in args.tickers:
        ticker = session.query(Ticker).filter_by(symbol=symbol).first()
        hechos = construir_hechos(session, ticker, modelo, columnas, confiabilidad, ranking) if ticker else None
        if hechos is None:
            print(f"{symbol}: sin datos")
            continue
        prompt = armar_prompt(hechos)
        inicio = time.perf_counter()
        texto = generar_texto(generador, prompt, hechos, confiabilidad.get("auc"))
        segundos = round(time.perf_counter() - inicio, 1)
        revision = validar_respuesta(texto, hechos)["validacion"]
        resultados.append({"ticker": symbol, "modelo_llm": GEMINI_MODEL, "segundos": segundos, "caracteres_prompt": len(prompt),
                           "senal_regla": hechos["senal"], "puntaje": hechos["puntaje_modelo"],
                           "n_noticias": len(hechos["noticias"]), "n_stocktwits": len(hechos["mensajes_stocktwits"]),
                           "n_filings": len(hechos["ultimos_filings_sec"]), "salida": texto, "validacion": revision,
                           "prompt": prompt})
        print(f"\n=== {symbol} ({segundos}s, prompt {len(prompt)} caracteres) senal={hechos['senal']} puntaje={hechos['puntaje_modelo']}")
        print(json.dumps({k: v for k, v in texto.items()}, ensure_ascii=False, indent=1))
        print("validacion:", revision)
        time.sleep(PAUSA_ENTRE_LLAMADAS)
    session.rollback()
    session.close()

    EVIDENCIAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = EVIDENCIAS_DIR / f"asesor_prueba_{dt.date.today():%Y%m%d}.json"
    ruta.write_text(json.dumps(resultados, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nguardado en", ruta, "(la base no se modifico)")


if __name__ == "__main__":
    main()
