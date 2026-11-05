"""Evidencias para el informe (3V, calidad y entrada real del LLM), leyendo la base en una conexion de solo lectura.

Uso:
    python -m src.reports.evidencias_datos --solo-lectura            # contra SUPABASE_URL_LECTURA
    python -m src.reports.evidencias_datos --solo-lectura --ticker PDSB
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import xgboost as xgb
from sqlalchemy import text

from src.config import TARGET_HORIZON_DAYS
from src.models.backtesting import EVIDENCIAS_DIR, sesion_solo_lectura
from src.models.train_xgboost import FEATURE_COLS, PUBLISHED_MODEL_PATH, construir_dataset, evaluar_walk_forward

TABLAS = ["tickers", "precios_ohlcv", "ratios_financieros", "textos", "sentimiento", "features",
          "filings_sec", "predicciones", "recomendaciones", "recomendaciones_ia"]


def conteos(session) -> dict:
    datos = {t: session.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in TABLAS}
    datos["tamano_base"] = session.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))")).scalar()
    datos["rango_precios"] = [str(x) for x in session.execute(text("SELECT min(fecha), max(fecha) FROM precios_ohlcv")).one()]
    datos["textos_por_fuente"] = dict(session.execute(text("SELECT fuente, count(*) FROM textos GROUP BY fuente")).all())
    datos["sentimiento_por_etiqueta"] = dict(session.execute(text("SELECT etiqueta, count(*) FROM sentimiento GROUP BY etiqueta")).all())
    datos["filings_por_categoria"] = dict(session.execute(text("SELECT categoria, count(*) FROM filings_sec GROUP BY categoria")).all())
    return datos


def calidad(session) -> dict:
    q = lambda sql: session.execute(text(sql)).scalar()  # noqa: E731
    total_precios = q("SELECT count(*) FROM precios_ohlcv")
    total_ratios = q("SELECT count(*) FROM ratios_financieros")
    total_textos = q("SELECT count(*) FROM textos")
    total_features = q("SELECT count(*) FROM features")
    return {
        "precios_con_ohlc_nulo": q("SELECT count(*) FROM precios_ohlcv WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL"),
        "precios_con_valor_no_positivo": q("SELECT count(*) FROM precios_ohlcv WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0"),
        "precios_high_menor_que_low": q("SELECT count(*) FROM precios_ohlcv WHERE high < low"),
        "precios_volumen_nulo": q("SELECT count(*) FROM precios_ohlcv WHERE volumen IS NULL"),
        "precios_saltos_diarios_mayores_60pct": q(
            "SELECT count(*) FROM (SELECT close / NULLIF(lag(close) OVER (PARTITION BY ticker_id ORDER BY fecha), 0) - 1 AS r "
            "FROM precios_ohlcv) s WHERE abs(r) > 0.6"),
        "total_precios": total_precios,
        "ratios_pe_nulo": q("SELECT count(*) FROM ratios_financieros WHERE pe_ratio IS NULL"),
        "ratios_pe_positivo_con_eps_negativo": q("SELECT count(*) FROM ratios_financieros WHERE pe_ratio > 0 AND eps <= 0"),
        "total_ratios": total_ratios,
        "textos_sin_url": q("SELECT count(*) FROM textos WHERE url IS NULL OR url = ''"),
        "textos_menos_de_10_caracteres": q("SELECT count(*) FROM textos WHERE length(trim(contenido)) < 10"),
        "textos_duplicados": q("SELECT coalesce(sum(c - 1), 0) FROM (SELECT count(*) c FROM textos GROUP BY ticker_id, fuente, contenido HAVING count(*) > 1) d"),
        "textos_sin_sentimiento": q("SELECT count(*) FROM textos t LEFT JOIN sentimiento s ON s.texto_id = t.id WHERE s.id IS NULL"),
        "total_textos": total_textos,
        "total_features": total_features,
    }


def ejemplo_entrada_llm(session, symbol: str) -> dict | None:
    from src.db import Ticker
    from src.reports.asesor_ia import SYSTEM_PROMPT, _puestos, armar_prompt, construir_hechos

    modelo = xgb.XGBClassifier()
    modelo.load_model(PUBLISHED_MODEL_PATH)
    columnas = list(modelo.get_booster().feature_names or FEATURE_COLS)
    dataset = construir_dataset(session)
    confiabilidad = evaluar_walk_forward(dataset, FEATURE_COLS, purga_dias=TARGET_HORIZON_DAYS) or {}
    ticker = session.query(Ticker).filter_by(symbol=symbol).first()
    if ticker is None:
        return None
    hechos = construir_hechos(session, ticker, modelo, columnas, confiabilidad, _puestos(session, modelo, columnas))
    return {"system_prompt": SYSTEM_PROMPT, "prompt_usuario": armar_prompt(hechos), "hechos": hechos}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-lectura", action="store_true", required=True)
    parser.add_argument("--ticker", default="PDSB")
    args = parser.parse_args()
    session = sesion_solo_lectura()
    try:
        salida = {"conteos": conteos(session), "calidad": calidad(session), "entrada_llm": ejemplo_entrada_llm(session, args.ticker)}
    finally:
        session.rollback()
        session.close()
    EVIDENCIAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = EVIDENCIAS_DIR / f"evidencias_datos_{dt.date.today():%Y%m%d}.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: salida[k] for k in ("conteos", "calidad")}, indent=2, ensure_ascii=False, default=str))
    print("guardado en", ruta)


if __name__ == "__main__":
    main()
