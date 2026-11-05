"""Fase 4 - Reglas de negocio (umbrales sobre la probabilidad de XGBoost) para el reporte prescriptivo."""
from __future__ import annotations

from src.config import (
    PERFIL_RIESGO_DEFECTO,
    PERFILES_RIESGO,
    RSI_SOBRECOMPRA,
    RSI_SOBREVENTA,
    UMBRAL_SENAL_ALERTA,
    UMBRAL_SENAL_MONITOREAR,
    VOLATILIDAD_DIARIA_MAXIMA,
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


def volatilidad_extrema(features: dict) -> bool:
    vol = features.get("vol_10d")
    return vol is not None and vol == vol and vol > VOLATILIDAD_DIARIA_MAXIMA


def generar_recomendacion(prediccion_prob: float, features: dict, perfil: str = PERFIL_RIESGO_DEFECTO) -> dict:
    """Traduce una probabilidad de XGBoost + features tecnicas en una senal auditable.

    Args:
        prediccion_prob: probabilidad (0-1) de que el precio suba segun la
            definicion de TARGET_PCT_THRESHOLD/TARGET_HORIZON_DAYS.
        features: dict con al menos rsi, volumen_relativo, sentimiento_agregado (y vol_10d para el freno).
        perfil: perfil de riesgo del usuario (clave de PERFILES_RIESGO); cambia el umbral de alerta.

    Returns:
        dict con tipo_senal, justificacion, si se aplico el freno de volatilidad y los limites del perfil.
    """
    limites = PERFILES_RIESGO.get(perfil, PERFILES_RIESGO[PERFIL_RIESGO_DEFECTO])
    umbral_alerta = max(limites["umbral_alerta"], UMBRAL_SENAL_ALERTA)

    if prediccion_prob >= umbral_alerta:
        tipo_senal = TIPO_SENAL_ALERTA
    elif prediccion_prob >= UMBRAL_SENAL_MONITOREAR:
        tipo_senal = TIPO_SENAL_MONITOREAR
    else:
        tipo_senal = TIPO_SENAL_NINGUNA

    notas = _contexto_tecnico(features)
    freno = volatilidad_extrema(features)
    if freno and tipo_senal == TIPO_SENAL_ALERTA:
        tipo_senal = TIPO_SENAL_MONITOREAR
    if freno:
        notas.append(
            f"volatilidad diaria extrema ({features['vol_10d']:.0%} > {VOLATILIDAD_DIARIA_MAXIMA:.0%}): "
            "puede ser un split o reverse split sin ajustar; no se emite senal de alerta"
        )

    justificacion = f"Puntaje del modelo XGBoost: {prediccion_prob:.1%}."
    if notas:
        justificacion += " Contexto: " + "; ".join(notas) + "."
    else:
        justificacion += " Sin senales tecnicas adicionales relevantes."

    return {
        "tipo_senal": tipo_senal,
        "justificacion": justificacion,
        "freno_volatilidad": freno,
        "perfil": perfil if perfil in PERFILES_RIESGO else PERFIL_RIESGO_DEFECTO,
        "posicion_max": limites["posicion_max"] if tipo_senal != TIPO_SENAL_NINGUNA else 0.0,
        "perdida_max": limites["perdida_max"],
    }
