"""Validacion de lo que devuelve el LLM y helpers del asesor IA (sin red)."""
import pytest

from src.reports.asesor_ia import _plano, _sentimiento, armar_prompt, normalizar_respuesta


def _crudo(**cambios):
    base = {
        "resumen": "  Puntaje   medio.\nSin catalizadores. ",
        "a_favor": ["uno", " ", "dos", "tres", "cuatro", "cinco"],
        "riesgos": ["r1"],
        "vigilar": [],
        "confianza": "Alta",
    }
    return {**base, **cambios}


def test_normaliza_espacios_vacios_y_limite_de_items():
    r = normalizar_respuesta(_crudo(), auc=0.7)
    assert r["resumen"] == "Puntaje medio. Sin catalizadores."
    assert r["a_favor"] == ["uno", "dos", "tres", "cuatro"]
    assert r["confianza"] == "alta"


def test_confianza_alta_se_rebaja_si_el_modelo_es_debil():
    assert normalizar_respuesta(_crudo(), auc=0.60)["confianza"] == "media"
    assert normalizar_respuesta(_crudo(), auc=None)["confianza"] == "media"


def test_confianza_invalida_cae_a_baja_y_resumen_vacio_falla():
    assert normalizar_respuesta(_crudo(confianza="segura"), auc=0.7)["confianza"] == "baja"
    with pytest.raises(ValueError):
        normalizar_respuesta(_crudo(resumen="   "), auc=0.7)


def test_textos_largos_se_recortan():
    assert len(_plano("x" * 500, 100)) == 100


def test_sentimiento_agregado_firmado():
    filas = [("noticia", "positivo", 0.8), ("noticia", "negativo", 0.6), ("noticia", "neutral", 0.9), ("stocktwits", "positivo", 1.0)]
    s = _sentimiento(filas, "noticia")
    assert (s["mensajes"], s["positivos"], s["negativos"], s["neutrales"]) == (3, 1, 1, 1)
    assert s["promedio_firmado"] == round((0.8 - 0.6 + 0.0) / 3, 3)
    assert _sentimiento([], "noticia")["promedio_firmado"] is None


def test_prompt_conserva_tildes_y_enie_del_json():
    assert "Ñandú" in armar_prompt({"empresa": "Ñandú"})
