"""Fase 2 - Entrenamiento de XGBoost sobre las features de feature_engineering.py.

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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import TARGET_HORIZON_DAYS, TARGET_PCT_THRESHOLD
from src.db import Feature, PrecioOHLCV, Ticker, get_session, init_db
from src.processing.clean_pipeline import limpiar_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# A diferencia de MODELS_DIR (gitignored), esta SI se versiona en git: es el modelo que ve la nube.
PUBLISHED_MODEL_PATH = ROOT_DIR / "models" / "xgboost_latest.json"
PUBLISHED_MODEL_VERSION_PATH = ROOT_DIR / "models" / "xgboost_latest.version.txt"
# Historial diario de la calidad del modelo (tambien versionado): una fila por reentrenamiento.
MONITOREO_PATH = ROOT_DIR / "models" / "monitoreo_modelo.csv"

# Monitoreo: con AUC walk-forward por debajo de AUC_MINIMO_PUBLICAR el modelo nuevo no ordena mejor que el azar
# y no reemplaza al publicado; por debajo de AUC_AVISO se publica pero queda una advertencia en el log y el dashboard.
AUC_MINIMO_PUBLICAR = 0.52
AUC_AVISO = 0.55

# Elegidas con src/models/experimentos.py: sin la media movil en dolares (delata el nivel de precio del ticker) y con volatilidad.
FEATURE_COLS = ["rsi", "volumen_relativo", "sentimiento_agregado", "precio_vs_sma", "vol_10d", "vol_20d", "rango_5d"]

# Poca profundidad y hojas grandes: con datos tan ruidosos el modelo profundo memoriza.
MODELO_PARAMS = dict(
    n_estimators=150, max_depth=2, learning_rate=0.05, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
)

N_FOLDS = 4
FRACCION_TEST = 0.4  # los ultimos 40% de las fechas se reparten en N_FOLDS tramos consecutivos
MIN_FECHAS = 100  # menos dias que esto no alcanza para 4 tramos de test con sentido


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
    precios_df = limpiar_ohlcv(precios_df)
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

    features_df = pd.DataFrame(
        [
            {"ticker_id": f.ticker_id, "fecha": pd.Timestamp(f.fecha), **{c: getattr(f, c) for c in FEATURE_COLS}}
            for f in session.query(Feature).all()
        ]
    )
    if features_df.empty:
        return features_df

    dataset = features_df.merge(precios_df[["ticker_id", "fecha", "symbol", "target"]], on=["ticker_id", "fecha"], how="inner")
    dataset = dataset.dropna(subset=FEATURE_COLS + ["target"])
    dataset["target"] = dataset["target"].astype(int)
    return dataset


def nuevo_modelo(y_train: pd.Series) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        eval_metric="logloss", random_state=42, scale_pos_weight=calcular_scale_pos_weight(y_train), **MODELO_PARAMS
    )


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
        modelo = nuevo_modelo(y.iloc[train_idx])
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


def evaluar_walk_forward(
    df: pd.DataFrame, columnas: list[str], purga_dias: int, params: dict | None = None
) -> dict | None:
    """AUC y lift del 10% mejor puntuado en N_FOLDS tramos consecutivos del final; entrena solo con fechas anteriores.

    `df` necesita `fecha`, `target` y `columnas`. `params` cambia la configuracion de XGBoost (por defecto MODELO_PARAMS).
    """
    datos = df.dropna(subset=columnas + ["target"])
    fechas = np.sort(datos["fecha"].unique())
    if len(fechas) < MIN_FECHAS:
        return None
    inicio_test = int(len(fechas) * (1 - FRACCION_TEST))
    tramos = np.array_split(np.arange(inicio_test, len(fechas)), N_FOLDS)
    aucs, lifts = [], []
    for tramo in tramos:
        # La purga evita que filas de train cuyo objetivo mira dentro del tramo de test se cuelen en el entrenamiento.
        train = datos[datos["fecha"] < fechas[max(tramo[0] - purga_dias, 0)]]
        test = datos[(datos["fecha"] >= fechas[tramo[0]]) & (datos["fecha"] <= fechas[tramo[-1]])]
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            continue
        modelo = nuevo_modelo(train["target"]) if params is None else xgb.XGBClassifier(
            eval_metric="logloss", random_state=42, scale_pos_weight=calcular_scale_pos_weight(train["target"]), **params
        )
        modelo.fit(train[columnas], train["target"])
        puntaje = modelo.predict_proba(test[columnas])[:, 1]
        aucs.append(roc_auc_score(test["target"], puntaje))
        mejores = test["target"].values[puntaje >= np.quantile(puntaje, 0.9)]
        lifts.append(mejores.mean() / test["target"].mean())
    if not aucs:
        return None
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "folds_sobre_azar": f"{sum(a > 0.5 for a in aucs)}/{len(aucs)}",
        "lift_top10": float(np.mean(lifts)),
        "tasa_base": float(datos["target"].mean()),
        "n": len(datos),
    }


def decidir_publicacion(walk_forward: dict | None, hay_modelo_publicado: bool) -> tuple[bool, str]:
    """Decide si el modelo recien entrenado reemplaza al publicado segun su AUC fuera de muestra."""
    auc = (walk_forward or {}).get("auc")
    if auc is None:
        return True, "sin evaluacion walk-forward (pocos datos): se publica igual"
    if auc < AUC_MINIMO_PUBLICAR:
        if hay_modelo_publicado:
            return False, f"AUC {auc:.3f} < {AUC_MINIMO_PUBLICAR}: no supera al azar, se mantiene el modelo anterior"
        return True, f"AUC {auc:.3f} < {AUC_MINIMO_PUBLICAR}, pero no hay modelo anterior: se publica con advertencia"
    if auc < AUC_AVISO:
        return True, f"AUC {auc:.3f} < {AUC_AVISO}: se publica con advertencia (poder predictivo bajo)"
    return True, f"AUC {auc:.3f}: ok"


def registrar_monitoreo(walk_forward: dict | None, version: str, publicado: bool, motivo: str, ruta: Path = MONITOREO_PATH) -> pd.DataFrame:
    """Agrega (o reemplaza, si ya se corrio hoy) la fila del dia en el historial de monitoreo."""
    wf = walk_forward or {}
    fila = {
        "fecha": dt.date.today().isoformat(),
        "version": version,
        "auc": wf.get("auc"),
        "auc_std": wf.get("auc_std"),
        "folds_sobre_azar": wf.get("folds_sobre_azar"),
        "lift_top10": wf.get("lift_top10"),
        "tasa_base": wf.get("tasa_base"),
        "n": wf.get("n"),
        "publicado": publicado,
        "motivo": motivo,
    }
    historial = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=list(fila))
    historial = historial[historial["fecha"] != fila["fecha"]]
    historial = pd.concat([historial, pd.DataFrame([fila])], ignore_index=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    historial.to_csv(ruta, index=False)
    return historial


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

    modelo_final = nuevo_modelo(y)
    modelo_final.fit(X, y)

    importancias = dict(zip(FEATURE_COLS, [float(v) for v in modelo_final.feature_importances_]))
    logger.info("Feature importance: %s", importancias)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fecha_version = dt.date.today().strftime("%Y%m%d")
    modelo_path = MODELS_DIR / f"xgboost_{fecha_version}.json"
    modelo_final.save_model(modelo_path)

    walk_forward = evaluar_walk_forward(dataset, FEATURE_COLS, purga_dias=TARGET_HORIZON_DAYS)
    publicar, motivo = decidir_publicacion(walk_forward, PUBLISHED_MODEL_PATH.exists())
    if publicar:
        PUBLISHED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        modelo_final.save_model(PUBLISHED_MODEL_PATH)
        PUBLISHED_MODEL_VERSION_PATH.write_text(f"xgboost-{fecha_version}\n")
        logger.info(
            "Modelo tambien guardado en %s (versionado en git) -- "
            "recuerda 'git add models/xgboost_latest.json && git commit && git push' "
            "para que el dashboard en la nube use este modelo nuevo.",
            PUBLISHED_MODEL_PATH,
        )
    if publicar and motivo.endswith("ok"):
        logger.info("Monitoreo del modelo: %s", motivo)
    elif publicar:
        logger.warning("Monitoreo del modelo: %s", motivo)
    else:
        logger.error("Monitoreo del modelo: %s", motivo)
    registrar_monitoreo(walk_forward, f"xgboost-{fecha_version}", publicar, motivo)

    reporte = {
        "modelo_version": f"xgboost-{fecha_version}",
        "n_filas_entrenamiento": len(dataset),
        "features": FEATURE_COLS,
        "target": {
            "definicion": f"close sube > {TARGET_PCT_THRESHOLD:.0%} en {TARGET_HORIZON_DAYS} dias de trading",
            "balance_clases": dataset["target"].value_counts().to_dict(),
        },
        "metricas_kfold": metricas,
        "walk_forward": walk_forward,
        "publicado": publicar,
        "monitoreo": motivo,
        "feature_importance": importancias,
    }
    with open(MODELS_DIR / f"reporte_{fecha_version}.json", "w") as f:
        json.dump(reporte, f, indent=2, default=str)

    logger.info("Modelo guardado en %s", modelo_path)
    return reporte


if __name__ == "__main__":
    run()
