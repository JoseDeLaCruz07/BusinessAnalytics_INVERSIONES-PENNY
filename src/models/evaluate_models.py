"""Fase 3 - QA: holdout temporal de XGBoost, accuracy de FinBERT, checklist de calidad de datos.

Uso:
    python -m src.models.evaluate_models
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import FINBERT_MODEL_VERSION
from src.db import PrecioOHLCV, RatioFinanciero, Texto, get_session, init_db
from src.config import TARGET_HORIZON_DAYS
from src.models.train_xgboost import FEATURE_COLS, construir_dataset, evaluar_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"


def evaluate_xgboost() -> dict:
    session = get_session()
    dataset = construir_dataset(session)
    session.close()

    resultado = evaluar_walk_forward(dataset, FEATURE_COLS, purga_dias=TARGET_HORIZON_DAYS) if not dataset.empty else None
    if not resultado:
        logger.warning("Dataset insuficiente para evaluar XGBoost (pocas fechas o un tramo sin ambas clases).")
        return {}
    logger.info(
        "XGBoost walk-forward: AUC=%.3f (std %.3f, tramos sobre azar %s) lift top 10%%=%.2fx tasa base=%.1f%%",
        resultado["auc"], resultado["auc_std"], resultado["folds_sobre_azar"], resultado["lift_top10"],
        resultado["tasa_base"] * 100,
    )
    return resultado


def evaluate_finbert() -> dict:
    fixture_path = FIXTURES_DIR / "finbert_validation_set.csv"
    if not fixture_path.exists():
        logger.warning("No existe %s, se omite evaluacion de FinBERT.", fixture_path)
        return {}

    df = pd.read_csv(fixture_path)

    from src.models.sentiment_finbert import classify_batch, load_pipeline

    logger.info("Clasificando %d ejemplos de validacion con FinBERT...", len(df))
    clf = load_pipeline()
    resultados = classify_batch(clf, df["contenido"].tolist())
    df["etiqueta_predicha"] = [r["etiqueta"] for r in resultados]
    df["score"] = [r["score"] for r in resultados]

    mal_clasificados = df[df["etiqueta_real"] != df["etiqueta_predicha"]]
    accuracy = float((df["etiqueta_real"] == df["etiqueta_predicha"]).mean())

    logger.info(
        "FinBERT (%s) sobre %d ejemplos etiquetados a mano: accuracy=%.3f (%d mal clasificados)",
        FINBERT_MODEL_VERSION,
        len(df),
        accuracy,
        len(mal_clasificados),
    )

    return {
        "n_ejemplos": len(df),
        "accuracy": accuracy,
        "casos_mal_clasificados": mal_clasificados[["contenido", "etiqueta_real", "etiqueta_predicha", "score"]].to_dict(
            orient="records"
        ),
    }


def checklist_calidad_datos() -> dict:
    session = get_session()

    total_precios = session.query(PrecioOHLCV).count()
    precios_con_nulos = (
        session.query(PrecioOHLCV)
        .filter(
            (PrecioOHLCV.open.is_(None))
            | (PrecioOHLCV.high.is_(None))
            | (PrecioOHLCV.low.is_(None))
            | (PrecioOHLCV.close.is_(None))
        )
        .count()
    )
    precios_negativos = session.query(PrecioOHLCV).filter(PrecioOHLCV.close <= 0).count()

    total_ratios = session.query(RatioFinanciero).count()
    ratios_sin_market_cap = session.query(RatioFinanciero).filter(RatioFinanciero.market_cap.is_(None)).count()

    total_textos = session.query(Texto).count()
    textos_sin_url = session.query(Texto).filter((Texto.url.is_(None)) | (Texto.url == "")).count()

    session.close()

    requirements_path = ROOT_DIR / "requirements.txt"
    lineas = [l.strip() for l in requirements_path.read_text().splitlines() if l.strip()]
    dependencias_pineadas = sum(1 for l in lineas if "==" in l)
    reproducibilidad_ok = requirements_path.exists() and dependencias_pineadas == len(lineas)

    checklist = {
        "completitud": {
            "precios_ohlcv_total": total_precios,
            "precios_ohlcv_con_nulos": precios_con_nulos,
            "ratios_financieros_total": total_ratios,
            "ratios_sin_market_cap": ratios_sin_market_cap,
        },
        "consistencia": {
            "precios_con_close_no_positivo": precios_negativos,
        },
        "trazabilidad": {
            "textos_total": total_textos,
            "textos_sin_url": textos_sin_url,
        },
        "reproducibilidad": {
            "requirements_txt_existe": requirements_path.exists(),
            "dependencias_totales": len(lineas),
            "dependencias_con_version_fija": dependencias_pineadas,
            "ok": reproducibilidad_ok,
        },
    }

    logger.info("Checklist de calidad de datos: %s", json.dumps(checklist, indent=2))
    return checklist


def run() -> dict:
    init_db()

    reporte = {
        "fecha": dt.date.today().isoformat(),
        "xgboost": evaluate_xgboost(),
        "finbert": evaluate_finbert(),
        "calidad_datos": checklist_calidad_datos(),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    salida = MODELS_DIR / f"evaluacion_{dt.date.today().strftime('%Y%m%d')}.json"
    with open(salida, "w") as f:
        json.dump(reporte, f, indent=2, default=str)
    logger.info("Reporte de evaluacion guardado en %s", salida)

    return reporte


if __name__ == "__main__":
    run()
