"""Fase 2 - Entrenamiento de XGBoost.

Variable objetivo (definicion de negocio confirmada con el usuario):
    1 si el precio de cierre sube mas de TARGET_PCT_THRESHOLD (5%) en los
    proximos TARGET_HORIZON_DAYS (3) dias de trading, 0 en caso contrario.

Features: rsi, media_movil, volumen_relativo, sentimiento_agregado (tabla
`features`, generada por feature_engineering.py).

Uso:
    python -m src.models.train_xgboost
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_HORIZON_DAYS, TARGET_PCT_THRESHOLD
from src.db import Feature, PrecioOHLCV, Ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# A diferencia de MODELS_DIR (gitignored, historico local por fecha), esta
# carpeta SI se versiona en git: es el modelo que "publicas" para que lo vea
# el dashboard desplegado en Streamlit Cloud (que solo tiene lo que hay en
# el repo, no tu carpeta data/ local). Despues de entrenar, hay que
# `git add models/xgboost_latest.json && git commit && git push` para que
# la nube use el modelo nuevo.
PUBLISHED_MODEL_PATH = ROOT_DIR / "models" / "xgboost_latest.json"
PUBLISHED_MODEL_VERSION_PATH = ROOT_DIR / "models" / "xgboost_latest.version.txt"

FEATURE_COLS = ["rsi", "media_movil", "volumen_relativo", "sentimiento_agregado"]


def construir_dataset(session) -> pd.DataFrame:
    precios = (
        session.query(PrecioOHLCV, Ticker.symbol)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .order_by(Ticker.symbol, PrecioOHLCV.fecha)
        .all()
    )
    precios_df = pd.DataFrame(
        [{"ticker_id": p.ticker_id, "symbol": s, "fecha": pd.Timestamp(p.fecha), "close": p.close} for p, s in precios]
    )
    if precios_df.empty:
        return precios_df

    def con_target(grupo: pd.DataFrame) -> pd.DataFrame:
        grupo = grupo.sort_values("fecha").copy()
        future_close = grupo["close"].shift(-TARGET_HORIZON_DAYS)
        pct_change = (future_close - grupo["close"]) / grupo["close"]
        grupo["target"] = (pct_change > TARGET_PCT_THRESHOLD).astype("Int64")
        grupo.loc[future_close.isna(), "target"] = pd.NA
        return grupo

    precios_df = precios_df.groupby("symbol", group_keys=False)[precios_df.columns].apply(con_target)

    features = session.query(Feature).all()
    features_df = pd.DataFrame(
        [
            {
                "ticker_id": f.ticker_id,
                "fecha": pd.Timestamp(f.fecha),
                "rsi": f.rsi,
                "media_movil": f.media_movil,
                "volumen_relativo": f.volumen_relativo,
                "sentimiento_agregado": f.sentimiento_agregado,
            }
            for f in features
        ]
    )
    if features_df.empty:
        return features_df

    dataset = features_df.merge(precios_df[["ticker_id", "fecha", "symbol", "target"]], on=["ticker_id", "fecha"], how="inner")
    dataset = dataset.dropna(subset=FEATURE_COLS + ["target"])
    dataset["target"] = dataset["target"].astype(int)
    return dataset


def calcular_scale_pos_weight(y: pd.Series) -> float:
    """Peso para compensar el desbalance de clases (target=1 suele ser minoritario).

    XGBoost recomienda sum(negativos) / sum(positivos); sin esto, con un dataset
    donde la clase 1 es <20%, el modelo tiende a predecir casi siempre 0 y logra
    accuracy alto sin aportar nada (precision/recall ~0 en la clase de interes).
    """
    positivos = int((y == 1).sum())
    negativos = int((y == 0).sum())
    if positivos == 0:
        return 1.0
    return negativos / positivos


def evaluar_kfold(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    n_splits = min(n_splits, y.value_counts().min()) if y.nunique() > 1 else 1
    if n_splits < 2:
        logger.warning("Muy pocos datos/clases para k-fold (n_splits=%d); se omite validacion cruzada.", n_splits)
        return {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metricas = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    matriz_confusion_total = np.zeros((2, 2), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        peso = calcular_scale_pos_weight(y.iloc[train_idx])
        modelo = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=peso,
        )
        modelo.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = modelo.predict(X.iloc[test_idx])

        metricas["accuracy"].append(accuracy_score(y.iloc[test_idx], pred))
        metricas["precision"].append(precision_score(y.iloc[test_idx], pred, zero_division=0))
        metricas["recall"].append(recall_score(y.iloc[test_idx], pred, zero_division=0))
        metricas["f1"].append(f1_score(y.iloc[test_idx], pred, zero_division=0))
        matriz_confusion_total += confusion_matrix(y.iloc[test_idx], pred, labels=[0, 1])

        logger.info(
            "Fold %d/%d: accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f",
            fold,
            n_splits,
            metricas["accuracy"][-1],
            metricas["precision"][-1],
            metricas["recall"][-1],
            metricas["f1"][-1],
        )

    resumen = {k: float(np.mean(v)) for k, v in metricas.items()}
    resumen["confusion_matrix"] = matriz_confusion_total.tolist()
    return resumen


def run() -> dict:
    init_db()
    session = get_session()
    dataset = construir_dataset(session)
    session.close()

    if dataset.empty:
        logger.warning(
            "No hay suficientes datos para entrenar (features vacio o sin overlap con precios). "
            "Corre primero seed_synthetic_data.py (o la ingesta real) y feature_engineering.py."
        )
        return {}

    logger.info("Dataset de entrenamiento: %d filas, balance de clases:\n%s", len(dataset), dataset["target"].value_counts())

    if dataset["target"].nunique() < 2:
        logger.warning(
            "El dataset solo tiene una clase (%s). Con datos sinteticos random-walk esto puede pasar; "
            "no se puede entrenar un clasificador binario todavia. Prueba con mas dias de historico (--days).",
            dataset["target"].unique().tolist(),
        )
        return {}

    X = dataset[FEATURE_COLS]
    y = dataset["target"]

    metricas = evaluar_kfold(X, y)

    peso_final = calcular_scale_pos_weight(y)
    modelo_final = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=peso_final,
    )
    modelo_final.fit(X, y)

    importancias = dict(zip(FEATURE_COLS, [float(v) for v in modelo_final.feature_importances_]))
    logger.info("Feature importance: %s", importancias)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fecha_version = dt.date.today().strftime("%Y%m%d")
    modelo_path = MODELS_DIR / f"xgboost_{fecha_version}.json"
    modelo_final.save_model(modelo_path)

    PUBLISHED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    modelo_final.save_model(PUBLISHED_MODEL_PATH)
    PUBLISHED_MODEL_VERSION_PATH.write_text(f"xgboost-{fecha_version}\n")
    logger.info(
        "Modelo tambien guardado en %s (versionado en git) -- "
        "recuerda 'git add models/xgboost_latest.json && git commit && git push' "
        "para que el dashboard en la nube use este modelo nuevo.",
        PUBLISHED_MODEL_PATH,
    )

    reporte = {
        "modelo_version": f"xgboost-{fecha_version}",
        "n_filas_entrenamiento": len(dataset),
        "features": FEATURE_COLS,
        "target": {
            "definicion": f"close sube > {TARGET_PCT_THRESHOLD:.0%} en {TARGET_HORIZON_DAYS} dias de trading",
            "balance_clases": dataset["target"].value_counts().to_dict(),
        },
        "metricas_kfold": metricas,
        "feature_importance": importancias,
    }
    with open(MODELS_DIR / f"reporte_{fecha_version}.json", "w") as f:
        json.dump(reporte, f, indent=2, default=str)

    logger.info("Modelo guardado en %s", modelo_path)
    return reporte


if __name__ == "__main__":
    run()
