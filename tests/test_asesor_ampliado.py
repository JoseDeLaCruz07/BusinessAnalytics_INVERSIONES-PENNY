"""Asesor IA ampliado: mas textos numerados, accion acotada por la senal, alternativas y citas verificadas."""
import datetime as dt
from types import SimpleNamespace as T

from src.reports.asesor_ia import (
    armar_prompt,
    generar_texto,
    normalizar_extras,
    seleccionar_noticias,
    seleccionar_stocktwits,
    validar_respuesta,
)


def _t(fuente, contenido, etiqueta="neutral", dias=0, url="https://ejemplo.com/x"):
    return T(fuente=fuente, contenido=contenido, etiqueta=etiqueta, url=url, fecha=dt.datetime(2026, 11, 3) - dt.timedelta(days=dias))


def test_noticias_hasta_10_sin_repetidas_y_numeradas():
    textos = [_t("noticia", f"Titular numero {i} con suficiente texto para contar", dias=i) for i in range(15)]
    textos.insert(1, _t("noticia", "Titular numero 0 con suficiente texto para contar"))  # repetida
    noticias = seleccionar_noticias(textos)
    assert len(noticias) == 10
    assert [n["id"] for n in noticias[:2]] == ["N-1", "N-2"]
    assert noticias[0]["sentimiento_finbert"] == "neutral" and "_url" in noticias[0]


def test_stocktwits_quita_ruido_y_mezcla_lados():
    textos = [_t("stocktwits", "$RETO"), _t("stocktwits", "$RETO pr?"), _t("stocktwits", "https://x.com/a $RETO")]
    textos += [_t("stocktwits", f"Me gusta mucho esta accion por el contrato {i}", "positivo", i) for i in range(6)]
    textos += [_t("stocktwits", f"Otra dilucion mas, la empresa quema caja {i}", "negativo", i) for i in range(6)]
    textos += [_t("stocktwits", "Otra dilucion mas, la empresa quema caja 0", "negativo")]  # duplicado
    mensajes = seleccionar_stocktwits(textos)
    assert len(mensajes) == 8
    etiquetas = [m["sentimiento_finbert"] for m in mensajes]
    assert "positivo" in etiquetas and "negativo" in etiquetas
    assert all("$RETO" != m["texto"] for m in mensajes)
    assert len({m["texto"] for m in mensajes}) == 8


HECHOS = {
    "senal": "sin senal", "freno_volatilidad": False, "puntaje_modelo": 0.3, "tasa_base": 0.19,
    "noticias": [{"id": "N-1", "fecha": "2026-11-02", "texto": "Oferta de acciones", "sentimiento_finbert": "negativo"}],
    "mensajes_stocktwits": [{"id": "ST-1", "fecha": "2026-11-02", "texto": "a la luna", "sentimiento_finbert": "positivo"}],
    "ultimos_filings_sec": [{"id": "SEC-1", "fecha": "2026-10-20", "formulario": "424B5", "detalle": "Oferta de acciones"}],
    "_enlaces": {"N-1": "https://n1", "ST-1": "https://st1", "SEC-1": "https://sec1"},
}


def test_con_sin_senal_la_accion_no_puede_ser_vigilar_de_cerca():
    r = normalizar_extras({"accion_sugerida": "Vigilar de cerca", "noticias_clave": []}, HECHOS)
    assert r["accion_sugerida"] == "esperar confirmacion" and r["accion_corregida"] is True
    assert r["accion_propuesta_ia"] == "vigilar de cerca"
    assert r["justificacion_accion"].startswith("Ajustado por la regla fija")


def test_la_ia_si_puede_ser_mas_prudente_que_la_regla():
    r = normalizar_extras({"accion_sugerida": "evitar"}, {**HECHOS, "senal": "senal de alerta"})
    assert r["accion_sugerida"] == "evitar" and r["accion_corregida"] is False


def test_freno_de_volatilidad_tambien_acota_la_accion():
    r = normalizar_extras({"accion_sugerida": "vigilar de cerca"}, {**HECHOS, "senal": "senal de alerta", "freno_volatilidad": True})
    assert r["accion_sugerida"] == "esperar confirmacion"


def test_citas_inexistentes_se_descartan_y_fallan_la_validacion():
    r = normalizar_extras({"accion_sugerida": "evitar", "noticias_clave": ["SEC-1", "n-1", "N-7", "SEC-1"]}, HECHOS)
    assert [n["id"] for n in r["noticias_clave"]] == ["SEC-1", "N-1"]
    assert r["noticias_clave"][0]["url"] == "https://sec1"
    assert r["citas_inexistentes"] == ["N-7"]
    texto = {"resumen": "x", "a_favor": [], "riesgos": [], "vigilar": [], **r}
    assert validar_respuesta(texto, HECHOS)["validacion"]["ok"] is False


def test_generar_texto_une_campos_base_y_nuevos():
    crudo = {"resumen": "Hay dilucion.", "a_favor": [], "riesgos": ["dilucion"], "vigilar": [], "confianza": "media",
             "accion_sugerida": "evitar", "justificacion_accion": "Oferta de acciones reciente.",
             "alternativas": ["Esperar el proximo 10-Q"], "noticias_clave": ["SEC-1"]}
    r = generar_texto(lambda _: crudo, "prompt", HECHOS, auc=0.6)
    assert r["accion_sugerida"] == "evitar" and r["alternativas"] == ["Esperar el proximo 10-Q"]
    assert validar_respuesta(r, HECHOS)["validacion"]["ok"] is True


def test_los_enlaces_no_viajan_al_llm():
    prompt = armar_prompt(HECHOS)
    assert "_enlaces" not in prompt and "https://sec1" not in prompt and "SEC-1" in prompt
