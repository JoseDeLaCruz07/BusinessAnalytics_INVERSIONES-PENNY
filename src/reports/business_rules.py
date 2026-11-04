"""Fase 4 - Reglas de negocio para el reporte prescriptivo.

Deliberadamente separado del modelo: la prediccion de XGBoost es una
probabilidad, no una recomendacion. Estas reglas son auditables a simple
vista (no estan "enterradas" en el modelo) y cualquier ajuste de umbral se
hace aqui, sin reentrenar nada.

Definicion de negocio (confirmar/ajustar segun se validen resultados reales):
- prob >= UMBRAL_SENAL_ALERTA                        -> "senal de alerta"
- UMBRAL_SENAL_MONITOREAR <= prob < UMBRAL_SENAL_ALERTA -> "monitorear"
- prob < UMBRAL_SENAL_MONITOREAR                     -> "sin senal"
Ademas, un RSI en zona de sobreventa/sobrecompra o un volumen anormalmente
alto se mencionan en la justificacion como contexto adicional (no cambian
el tipo de senal por si solos).
"""
from __future__ import annotations

from src.config import (
    RSI_SOBRECOMPRA,
    RSI_SOBREVENTA,
    UMBRAL_SENAL_ALERTA,
    UMBRAL_SENAL_MONITOREAR,
    VOLUMEN_RELATIVO_SPIKE,
)

TIPO_SENAL_ALERTA = "senal de alerta"
TIPO_SENAL_MONITOREAR = "monitorear"
TIPO_SENAL_NINGUNA = "sin senal"


def _contexto_tecnico(features: dict) -> list[str]:
    notas = []
    rsi = features.get("rsi")
    volumen_relativo = features.get("volumen_relativo")
    sentimiento_agregado = features.get("sentimiento_agregado")

    if rsi is not None:
        if rsi <= RSI_SOBREVENTA:
            notas.append(f"RSI={rsi:.1f} en zona de sobreventa (<= {RSI_SOBREVENTA})")
        elif rsi >= RSI_SOBRECOMPRA:
            notas.append(f"RSI={rsi:.1f} en zona de sobrecompra (>= {RSI_SOBRECOMPRA})")

    if volumen_relativo is not None and volumen_relativo >= VOLUMEN_RELATIVO_SPIKE:
        notas.append(f"volumen {volumen_relativo:.1f}x el promedio movil (spike)")

    if sentimiento_agregado is not None and abs(sentimiento_agregado) > 0.3:
        polaridad = "positivo" if sentimiento_agregado > 0 else "negativo"
        notas.append(f"sentimiento de noticias {polaridad} ({sentimiento_agregado:+.2f})")

    return notas


def generar_recomendacion(prediccion_prob: float, features: dict) -> dict:
    """Traduce una probabilidad de XGBoost + features tecnicas en una senal auditable.

    Args:
        prediccion_prob: probabilidad (0-1) de que el precio suba segun la
            definicion de TARGET_PCT_THRESHOLD/TARGET_HORIZON_DAYS.
        features: dict con al menos rsi, volumen_relativo, sentimiento_agregado.

    Returns:
        dict con tipo_senal y justificacion (texto explicando por que).
    """
    if prediccion_prob >= UMBRAL_SENAL_ALERTA:
        tipo_senal = TIPO_SENAL_ALERTA
    elif prediccion_prob >= UMBRAL_SENAL_MONITOREAR:
        tipo_senal = TIPO_SENAL_MONITOREAR
    else:
        tipo_senal = TIPO_SENAL_NINGUNA

    notas = _contexto_tecnico(features)
    justificacion = f"Probabilidad de suba (modelo XGBoost): {prediccion_prob:.1%}."
    if notas:
        justificacion += " Contexto: " + "; ".join(notas) + "."
    else:
        justificacion += " Sin senales tecnicas adicionales relevantes."

    return {"tipo_senal": tipo_senal, "justificacion": justificacion}
