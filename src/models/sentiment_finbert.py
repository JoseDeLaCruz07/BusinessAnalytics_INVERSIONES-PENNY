"""Fase 2 - Analisis de sentimiento con FinBERT (ProsusAI/finbert).

Clasifica cada texto pendiente en `textos` (positivo/negativo/neutral + score
de confianza) y guarda el resultado en `sentimiento`. La primera corrida
descarga los pesos del modelo desde Hugging Face (~440MB), asi que requiere
salida de red a huggingface.co.

Uso:
    python -m src.models.sentiment_finbert
    python -m src.models.sentiment_finbert --batch-size 16
"""
from __future__ import annotations

import argparse
import logging

from src.config import FINBERT_MODEL_NAME, FINBERT_MODEL_VERSION
from src.db import Sentimiento, Texto, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mapeo de las etiquetas nativas de ProsusAI/finbert a las de nuestro esquema.
LABEL_MAP = {
    "positive": "positivo",
    "negative": "negativo",
    "neutral": "neutral",
}


def load_pipeline():
    """Carga el pipeline de clasificacion de FinBERT (import perezoso: transformers/torch pesan)."""
    from transformers import pipeline

    logger.info("Cargando modelo %s (puede tardar la primera vez por la descarga)...", FINBERT_MODEL_NAME)
    return pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME, tokenizer=FINBERT_MODEL_NAME)


def classify_batch(clf, textos: list[str]) -> list[dict]:
    """Devuelve [{'etiqueta': ..., 'score': ...}, ...] alineado con `textos`."""
    resultados = clf(textos, truncation=True, max_length=512)
    salida = []
    for r in resultados:
        salida.append({"etiqueta": LABEL_MAP.get(r["label"].lower(), r["label"].lower()), "score": float(r["score"])})
    return salida


def run(batch_size: int = 16, limit: int | None = None) -> None:
    init_db()
    session = get_session()

    pendientes = (
        session.query(Texto)
        .outerjoin(Sentimiento, Sentimiento.texto_id == Texto.id)
        .filter(Sentimiento.id.is_(None))
    )
    if limit:
        pendientes = pendientes.limit(limit)
    pendientes = pendientes.all()

    if not pendientes:
        logger.info("No hay textos pendientes de clasificar.")
        session.close()
        return

    logger.info("Clasificando %d textos con %s...", len(pendientes), FINBERT_MODEL_NAME)
    clf = load_pipeline()

    for i in range(0, len(pendientes), batch_size):
        lote = pendientes[i : i + batch_size]
        resultados = classify_batch(clf, [t.contenido for t in lote])
        for texto, res in zip(lote, resultados):
            session.add(
                Sentimiento(
                    texto_id=texto.id,
                    score=res["score"],
                    etiqueta=res["etiqueta"],
                    modelo_version=FINBERT_MODEL_VERSION,
                )
            )
        session.commit()
        logger.info("Procesados %d/%d", min(i + batch_size, len(pendientes)), len(pendientes))

    session.close()
    logger.info("Listo.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica sentimiento de textos pendientes con FinBERT")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limitar cantidad de textos a procesar (debug)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(batch_size=args.batch_size, limit=args.limit)
