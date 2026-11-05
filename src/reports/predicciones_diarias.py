"""Fase 3 - Registro diario de predicciones: guarda el puntaje y la senal de cada ticker una vez al dia.

Antes la fila del dia solo se guardaba cuando alguien abria el dashboard, asi que el historial de aciertos tenia
huecos los dias sin visitas. Este paso corre en el workflow diario, despues de reentrenar, con el mismo modelo
publicado que ve la nube; el dashboard sigue llamando a la misma funcion y no duplica filas.

Uso:
    python -m src.reports.predicciones_diarias
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import pandas as pd
import xgboost as xgb

from src.db import Feature, Prediccion, Recomendacion, Ticker, get_session, init_db
from src.models.train_xgboost import FEATURE_COLS, PUBLISHED_MODEL_PATH, PUBLISHED_MODEL_VERSION_PATH
from src.reports.business_rules import generar_recomendacion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Lo que se guarda en features_importantes: las del modelo y las que usa la regla de negocio.
COLUMNAS_FEATURES = list(dict.fromkeys(["rsi", "media_movil", "volumen_relativo", "sentimiento_agregado", *FEATURE_COLS]))


def guardar_prediccion_y_recomendacion(
    session, ticker_id: int, modelo_version: str, prob: float, features: dict, recomendacion: dict, fecha: dt.date | None = None
) -> bool:
    """Guarda la prediccion y su senal; devuelve False si ya habia una de ese dia con ese modelo."""
    fecha = fecha or dt.date.today()
    existe = session.query(Prediccion).filter_by(ticker_id=ticker_id, fecha=fecha, modelo_version=modelo_version).first()
    if existe:
        return False
    pred = Prediccion(
        ticker_id=ticker_id,
        fecha=fecha,
        modelo_version=modelo_version,
        prediccion="sube" if prob >= 0.5 else "no_sube",
        score=prob,
        features_importantes=json.dumps(features, default=str),
    )
    session.add(pred)
    session.commit()
    session.refresh(pred)
    session.add(
        Recomendacion(
            prediccion_id=pred.id,
            tipo_senal=recomendacion["tipo_senal"],
            justificacion=recomendacion["justificacion"],
        )
    )
    session.commit()
    return True


def cargar_modelo_publicado():
    if not PUBLISHED_MODEL_PATH.exists():
        return None, None
    modelo = xgb.XGBClassifier()
    modelo.load_model(PUBLISHED_MODEL_PATH)
    version = PUBLISHED_MODEL_VERSION_PATH.read_text().strip() if PUBLISHED_MODEL_VERSION_PATH.exists() else "xgboost-latest"
    return modelo, version


def run(session=None, modelo=None, modelo_version: str | None = None) -> int:
    """Registra la prediccion de hoy de cada ticker con features; devuelve cuantas filas nuevas guardo."""
    propia = session is None
    if propia:
        init_db()
        session = get_session()
    if modelo is None:
        modelo, modelo_version = cargar_modelo_publicado()
    if modelo is None:
        logger.error("No hay modelo publicado en %s: entrena primero con train_xgboost.", PUBLISHED_MODEL_PATH)
        return 0
    columnas = list(modelo.get_booster().feature_names or FEATURE_COLS)

    nuevas = 0
    for ticker in session.query(Ticker).order_by(Ticker.symbol).all():
        fila = session.query(Feature).filter_by(ticker_id=ticker.id).order_by(Feature.fecha.desc()).first()
        if fila is None:
            logger.warning("%s: sin features; no se registra prediccion", ticker.symbol)
            continue
        features = {"fecha": fila.fecha, **{c: getattr(fila, c) for c in COLUMNAS_FEATURES}}
        prob = float(modelo.predict_proba(pd.DataFrame([{c: features[c] for c in columnas}]))[0][1])
        recomendacion = generar_recomendacion(prob, features)
        if guardar_prediccion_y_recomendacion(session, ticker.id, modelo_version, prob, features, recomendacion):
            nuevas += 1
            logger.info("%s: puntaje %.3f, senal %s", ticker.symbol, prob, recomendacion["tipo_senal"])

    logger.info("Predicciones registradas hoy: %d nuevas", nuevas)
    if propia:
        session.close()
    return nuevas


if __name__ == "__main__":
    run()
