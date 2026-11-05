"""Las features de los experimentos solo pueden usar datos hasta el dia t; el objetivo si mira adelante."""
import numpy as np
import pandas as pd

from src.models.experimentos import GRUPOS, con_objetivo, features_por_ticker
from src.models.train_xgboost import evaluar_walk_forward


def _precios(n=80, seed=1):
    rng = np.random.default_rng(seed)
    close = 3 * np.exp(np.cumsum(rng.normal(scale=0.03, size=n)))
    return pd.DataFrame(
        {
            "ticker_id": 1, "symbol": "AAA", "fecha": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * 0.99, "high": close * 1.02, "low": close * 0.97, "close": close,
        }
    )


def _filings(fechas):
    return pd.DataFrame(
        {"ticker_id": 1, "fecha": pd.to_datetime(fechas), "formulario": "8-K", "categoria": "Dilucion"}
    )


def test_features_no_cambian_si_se_altera_el_futuro():
    p = _precios()
    f = _filings(["2024-02-01", "2024-03-01"])
    original = features_por_ticker(p, f)
    alterado = p.copy()
    alterado.loc[alterado.index >= 50, ["open", "high", "low", "close"]] *= 10
    nuevo = features_por_ticker(alterado, _filings(["2024-02-01", "2024-03-01", "2024-04-30"]))
    columnas = [c for g in GRUPOS.values() for c in g if c in original.columns]
    pd.testing.assert_frame_equal(original.loc[:49, columnas], nuevo.loc[:49, columnas])


def test_filings_del_mismo_dia_no_cuentan():
    p = _precios()
    dia = p["fecha"].iloc[40]
    out = features_por_ticker(p, _filings([dia]))
    assert out.loc[out["fecha"] == dia, "n_filings_30d"].iloc[0] == 0
    assert out.loc[out["fecha"] == p["fecha"].iloc[41], "n_filings_30d"].iloc[0] == 1


def test_objetivo_descarta_filas_sin_futuro_y_marca_subidas():
    df = pd.DataFrame({"ticker_id": 1, "close": [1.0, 1.0, 1.2, 1.2, 1.2]})
    r = con_objetivo(df, dias=2, umbral=0.05)
    assert len(r) == 3 and r["target"].tolist() == [1.0, 1.0, 0.0]


def test_evaluar_detecta_senal_real_y_no_inventa_ninguna():
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2022-01-01", periods=400, freq="B")
    df = pd.DataFrame({"fecha": np.repeat(fechas, 5), "x": rng.normal(size=2000), "ruido": rng.normal(size=2000)})
    df["target"] = (df["x"] + rng.normal(scale=0.5, size=2000) > 0.5).astype(float)
    config = dict(n_estimators=50, max_depth=2, learning_rate=0.1)
    con_senal = evaluar_walk_forward(df, ["x"], purga_dias=3, params=config)
    sin_senal = evaluar_walk_forward(df, ["ruido"], purga_dias=3, params=config)
    assert con_senal["auc"] > 0.8
    assert sin_senal["auc"] < 0.6


def test_evaluar_con_pocas_fechas_o_una_sola_clase_devuelve_none():
    fechas = pd.date_range("2022-01-01", periods=400, freq="B")
    df = pd.DataFrame({"fecha": np.repeat(fechas, 2), "x": np.arange(800.0), "target": 0.0})
    assert evaluar_walk_forward(df, ["x"], purga_dias=3) is None
    df["target"] = (np.arange(800) % 2).astype(float)
    assert evaluar_walk_forward(df.iloc[:100], ["x"], purga_dias=3) is None
