"""Pruebas de las reglas nuevas: freno de volatilidad, perfil de riesgo, validacion de cifras del LLM y backtesting."""
import numpy as np
import pandas as pd

from src.models.backtesting import caida_maxima, comprar_y_mantener, retornos_a_horizonte, simular
from src.reports.asesor_ia import cifras_no_respaldadas, contradicciones, marcar_no_validado, validar_respuesta
from src.reports.business_rules import generar_recomendacion

BASE = {"rsi": 50.0, "volumen_relativo": 1.0, "sentimiento_agregado": 0.0, "vol_10d": 0.05}


def test_freno_de_volatilidad_baja_la_alerta_a_monitorear():
    """Caso RETO: puntaje 68% pero volatilidad diaria absurda por un reverse split sin ajustar."""
    rec = generar_recomendacion(0.68, {**BASE, "vol_10d": 0.62})
    assert rec["tipo_senal"] == "monitorear"
    assert rec["freno_volatilidad"] is True
    assert "split" in rec["justificacion"]


def test_sin_volatilidad_extrema_la_alerta_se_mantiene():
    rec = generar_recomendacion(0.68, BASE)
    assert rec["tipo_senal"] == "senal de alerta"
    assert rec["freno_volatilidad"] is False


def test_perfil_conservador_exige_mas_puntaje_y_limita_la_posicion():
    conservador = generar_recomendacion(0.70, BASE, "Conservador")
    agresivo = generar_recomendacion(0.70, BASE, "Agresivo")
    assert conservador["tipo_senal"] == "monitorear"
    assert agresivo["tipo_senal"] == "senal de alerta"
    assert conservador["posicion_max"] < agresivo["posicion_max"]
    assert conservador["perdida_max"] < agresivo["perdida_max"]


def test_sin_senal_no_sugiere_posicion():
    rec = generar_recomendacion(0.10, BASE, "Agresivo")
    assert rec["tipo_senal"] == "sin senal"
    assert rec["posicion_max"] == 0.0


HECHOS_ABUS = {
    "puntaje_modelo": 0.386, "tasa_base": 0.187, "auc": 0.603, "senal": "sin senal",
    "puesto_entre_tickers": "16 de 18", "rango_precio_1_mes_80pct": [3.4879, 4.5873], "volatilidad_diaria": 0.0233,
    "sentimiento_noticias_30d": {"promedio_firmado": 0.175},
}


def test_detecta_el_error_real_de_abus():
    """Texto real que publico Gemini el 3 nov 2026: dice 'por debajo' cuando 0.386 > 0.187, y el puesto 17 en vez de 16."""
    texto = {
        "resumen": "Arbutus muestra un puntaje de modelo de 0.386, el cual esta por debajo de la tasa base de 0.187.",
        "a_favor": [], "riesgos": ["Se ubica en el puesto 17 de 18 entre los tickers analizados."], "vigilar": [],
    }
    problemas = contradicciones(texto, HECHOS_ABUS)
    assert any("por debajo" in p for p in problemas)
    assert any("puesto 17" in p for p in problemas)
    assert validar_respuesta(texto, HECHOS_ABUS)["validacion"]["ok"] is False


def test_cifras_del_json_pasan_y_las_inventadas_no():
    texto = {"resumen": "Puntaje 0.386 frente a 18.7% de tasa base; rango 3.4879 a 4.5873.", "a_favor": [],
             "riesgos": ["Podria llegar a 9.99 dolares."], "vigilar": []}
    assert cifras_no_respaldadas(texto, HECHOS_ABUS) == ["9.99"]


def test_texto_no_validado_queda_con_confianza_baja_y_aviso():
    texto = {"resumen": "x", "a_favor": [], "riesgos": ["r1"], "vigilar": [], "confianza": "media"}
    marcado = marcar_no_validado(texto, {"ok": False})
    assert marcado["confianza"] == "baja"
    assert marcado["riesgos"][0].startswith("Aviso automatico")


def test_caida_maxima():
    assert caida_maxima(pd.Series([1.0, 1.2, 0.6, 0.9])) == -0.5


def _precios(cierres_por_ticker: dict) -> pd.DataFrame:
    filas = []
    fechas = pd.bdate_range("2026-01-01", periods=len(next(iter(cierres_por_ticker.values()))))
    for tid, (symbol, cierres) in enumerate(cierres_por_ticker.items(), start=1):
        filas += [{"ticker_id": tid, "symbol": symbol, "fecha": f, "close": c} for f, c in zip(fechas, cierres)]
    return pd.DataFrame(filas)


def test_retorno_a_horizonte_marca_saltos_de_split():
    precios = _precios({"AAA": [1, 1, 1, 8, 8, 8, 8]})  # reverse split 8 a 1 el dia 3
    r = retornos_a_horizonte(precios, 3)
    assert bool(r.iloc[0]["salto_en_ventana"]) is True  # la ventana del dia 0 incluye el salto
    assert bool(r.iloc[3]["salto_en_ventana"]) is False


def test_comprar_y_mantener_neutraliza_splits_y_cobra_costos():
    precios = _precios({"AAA": [1, 1, 8, 8], "BBB": [1, 1, 1, 1]})
    r = comprar_y_mantener(precios, 0.0)
    assert abs(r["retorno_total"]) < 1e-9  # el x8 no cuenta como ganancia
    con_costo = comprar_y_mantener(precios, 0.01)
    assert con_costo["retorno_total"] < 0


def test_simular_cobra_costos_y_queda_en_efectivo_sin_senal():
    fechas = pd.bdate_range("2026-01-01", periods=6)
    oos = pd.DataFrame({
        "fecha": np.repeat(fechas, 1), "puntaje": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
        "ret_h": [0.10, 0, 0, 0.5, 0, 0], "salto_en_ventana": [False] * 6,
    })
    r = simular(oos, "alerta", 0.001, False)
    assert r["operaciones"] == 1  # solo el primer dia tiene senal; el dia 4 (rebalanceo) no supera el umbral
    assert abs(r["retorno_total"] - (0.999 * 1.10 * 0.999 - 1)) < 1e-9


def test_feature_engineering_ignora_ruido_de_redondeo():
    """La base devuelve los floats redondeados en el decimal 15: eso no debe contar como cambio (causaba el timeout)."""
    from src.models.feature_engineering import valor_distinto

    assert valor_distinto(23.7266147343571, 23.72661473435707) is False
    assert valor_distinto(1.26035285614418, 1.2603528561441757) is False
    assert valor_distinto(23.7266, 24.1) is True
    assert valor_distinto(None, float("nan")) is False
    assert valor_distinto(None, 1.0) is True
