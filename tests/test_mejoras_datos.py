"""Pruebas de: precios ajustados por splits, limpieza compartida, registro diario de predicciones,
monitoreo del modelo y limite diario del LLM."""
import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db import Base, Prediccion, Recomendacion, Ticker, UsoLLM
from src.ingestion import ingest_market_data
from src.ingestion.ingest_market_data import fechas_desfasadas, saltos_sospechosos
from src.models.train_xgboost import AUC_AVISO, AUC_MINIMO_PUBLICAR, decidir_publicacion, registrar_monitoreo
from src.processing.clean_pipeline import limpiar_ohlcv
from src.reports.asesor_ia import reservar_llamada
from src.reports.predicciones_diarias import guardar_prediccion_y_recomendacion


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


# --- Splits ---

def d(dia: int) -> dt.date:
    return dt.date(2026, 10, dia)


def test_detecta_historial_guardado_antes_de_un_reverse_split():
    """Caso RETO: la base guardo 0.625 el 29 oct; tras el 1:8 Yahoo devuelve ese dia ajustado a 5.00."""
    guardados = {d(28): 0.61, d(29): 0.625, d(30): 4.45}
    nuevo = pd.DataFrame({"fecha": [d(28), d(29), d(30), d(31)], "close": [4.88, 5.00, 4.45, 4.50]})
    assert fechas_desfasadas(guardados, nuevo) == [d(28), d(29)]


def test_diferencias_de_redondeo_no_disparan_reescritura():
    guardados = {d(28): 1.000, d(29): 1.100}
    nuevo = pd.DataFrame({"fecha": [d(28), d(29)], "close": [1.004, 1.0995]})
    assert fechas_desfasadas(guardados, nuevo) == []


def test_avisa_saltos_de_mas_de_60_por_ciento():
    df = pd.DataFrame({"fecha": [d(28), d(29), d(30)], "close": [0.62, 0.625, 4.45]})
    saltos = saltos_sospechosos(df)
    assert saltos["fecha"].tolist() == [d(30)]
    assert saltos["cambio"].iloc[0] > 6


def test_descarga_ajustada_y_con_el_dia_de_hoy(monkeypatch):
    llamada = {}

    def falso_download(symbol, start, end, progress, auto_adjust):
        llamada.update(start=start, end=end, auto_adjust=auto_adjust)
        idx = pd.DatetimeIndex([pd.Timestamp(dt.date.today())], name="Date")
        return pd.DataFrame({"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.0], "Volume": [100]}, index=idx)

    monkeypatch.setattr(ingest_market_data.yf, "download", falso_download)
    desde = dt.date.today() - dt.timedelta(days=400)
    df = ingest_market_data.fetch_ohlcv("RETO", days=90, desde=desde)
    assert llamada["auto_adjust"] is True
    assert llamada["end"] > dt.date.today()  # yfinance excluye `end`
    assert llamada["start"] == desde  # se compara contra todo el historial guardado
    assert df["fecha"].iloc[0] == dt.date.today()


# --- Limpieza compartida ---

def test_limpieza_descarta_filas_invalidas_sin_recortar_subas_reales():
    df = pd.DataFrame({
        "symbol": ["A"] * 5,
        "fecha": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-03", "2026-10-06"]),
        "high": [1.1, None, 1.0, 2.6, 0.5],
        "low": [0.9, 0.9, 1.2, 2.0, 0.6],
        "close": [1.0, 1.0, 1.1, 2.5, 0.55],
        "volumen": [100, 100, None, None, 100],
    })
    limpio = limpiar_ohlcv(df)
    # sin high (02), high < low (06) y el duplicado del 03 se quedan afuera; la suba a 2.5 (+150%) se mantiene.
    assert limpio["close"].tolist() == [1.0, 2.5]
    assert limpio["volumen"].tolist() == [100, 0]


# --- Registro diario de predicciones ---

def test_prediccion_del_dia_se_guarda_una_sola_vez(session):
    session.add(Ticker(symbol="SNDL"))
    session.commit()
    tid = session.query(Ticker).first().id
    rec = {"tipo_senal": "monitorear", "justificacion": "prueba"}
    assert guardar_prediccion_y_recomendacion(session, tid, "xgboost-20261104", 0.55, {"rsi": 40}, rec) is True
    assert guardar_prediccion_y_recomendacion(session, tid, "xgboost-20261104", 0.55, {"rsi": 40}, rec) is False
    assert session.query(Prediccion).count() == 1
    assert session.query(Recomendacion).count() == 1


# --- Monitoreo del modelo ---

def test_modelo_peor_que_el_azar_no_reemplaza_al_publicado():
    publicar, motivo = decidir_publicacion({"auc": AUC_MINIMO_PUBLICAR - 0.03}, hay_modelo_publicado=True)
    assert publicar is False
    assert "se mantiene el modelo anterior" in motivo


def test_modelo_flojo_se_publica_con_advertencia_y_uno_bueno_sin_ella():
    assert decidir_publicacion({"auc": AUC_AVISO - 0.01}, True) == (True, f"AUC {AUC_AVISO - 0.01:.3f} < {AUC_AVISO}: se publica con advertencia (poder predictivo bajo)")
    assert decidir_publicacion({"auc": 0.61}, True)[1].endswith("ok")
    assert decidir_publicacion(None, True)[0] is True


def test_monitoreo_guarda_una_fila_por_dia(tmp_path):
    ruta = tmp_path / "monitoreo.csv"
    registrar_monitoreo({"auc": 0.58, "n": 100}, "xgboost-a", True, "ok", ruta)
    historial = registrar_monitoreo({"auc": 0.60, "n": 120}, "xgboost-b", True, "ok", ruta)
    assert len(historial) == 1
    assert historial["auc"].iloc[0] == 0.60


# --- Limite diario del LLM ---

def test_limite_diario_de_consultas_al_llm(session):
    hoy = dt.date(2026, 11, 4)
    assert [reservar_llamada(session, hoy, limite=2) for _ in range(3)] == [True, True, False]
    assert session.get(UsoLLM, hoy).llamadas == 2
    # Al dia siguiente el contador vuelve a cero.
    assert reservar_llamada(session, hoy + dt.timedelta(days=1), limite=2) is True
