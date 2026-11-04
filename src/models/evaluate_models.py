"""Fase 3 - ML Quality Assurance.

Tres responsabilidades, independientes entre si:
1. evaluate_xgboost(): holdout temporal honesto (entrena solo con el pasado,
   evalua sobre el futuro mas reciente de cada ticker) - mas riguroso que el
   k-fold de train_xgboost.py, que mezcla fechas entre train/test.
2. evaluate_finbert(): accuracy de FinBERT contra un set de validacion
   etiquetado a mano (tests/fixtures/finbert_validation_set.csv) y lista de
   casos mal clasificados.
3. checklist_calidad_datos(): completitud, consistencia, trazabilidad y
   reproducibilidad sobre lo que hay en la base de datos ahora mismo.

Uso:
    python -m src.models.evaluate_models
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import FINBERT_MODEL_VERSION
from src.db import PrecioOHLCV, RatioFinanciero, Texto, get_session, init_db
from src.models.train_xgboost import FEATURE_COLS, calcular_scale_pos_weight, construir_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
TEST_SIZE = 0.2


def split_temporal(dataset: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Para cada ticker, las filas mas recientes van a test; el resto a train.

    Evita fuga de informacion (nunca se entrena con datos "del futuro" respecto al test).
    """
    train_parts, test_parts = [], []
    for _, grupo in dataset.groupby("symbol"):
        grupo = grupo.sort_values("fecha")
        n_test = max(1, math.ceil(len(grupo) * test_size))
        train_parts.append(grupo.iloc[:-n_test])
        test_parts.append(grupo.iloc[-n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


def evaluate_xgboost() -> dict:
    session = get_session()
    dataset = construir_dataset(session)
    session.close()

    if dataset.empty or dataset["target"].nunique() < 2:
        logger.warning("Dataset insuficiente para evaluar XGBoost (vacio o con una sola clase).")
        return {}

    train_df, test_df = split_temporal(dataset)
    if train_df.empty or test_df.empty or train_df["target"].nunique() < 2:
        logger.warning("Split temporal no genero suficientes datos/clases en train. Se omite evaluacion de XGBoost.")
        return {}

    peso = calcular_scale_pos_weight(train_df["target"])
    modelo = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=peso,
    )
    modelo.fit(train_df[FEATURE_COLS], train_df["target"])
    pred = modelo.predict(test_df[FEATURE_COLS])
    y_test = test_df["target"]

    resultado = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "feature_importance": dict(zip(FEATURE_COLS, [float(v) for v in modelo.feature_importances_])),
    }
    logger.info(
        "XGBoost holdout temporal: accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f (train=%d, test=%d)",
        resultado["accuracy"],
        resultado["precision"],
        resultado["recall"],
        resultado["f1"],
        resultado["n_train"],
        resultado["n_test"],
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
