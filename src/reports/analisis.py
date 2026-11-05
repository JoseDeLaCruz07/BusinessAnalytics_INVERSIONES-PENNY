"""Analisis compartido por el dashboard y el asesor IA: rango de precio a 1 mes y aporte de cada feature a la prediccion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

HORIZONTE_MES_DIAS = 21
VENTANA_VOLATILIDAD_DIAS = 60
Z_80, Z_50 = 1.2816, 0.6745


def contribuciones_prediccion(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """Aporte de cada feature a la prediccion de este ticker (log-odds, via XGBoost pred_contribs)."""
    aportes = modelo.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    return pd.DataFrame({"feature": list(X.columns), "valor": X.iloc[0].values, "contribucion": aportes})


def calcular_cono_volatilidad(ohlcv_df: pd.DataFrame) -> dict | None:
    """Rango de precios a 1 mes con volatilidad historica (lognormal, sin tendencia): un rango, no una prediccion."""
    ordenado = ohlcv_df.sort_values("fecha")
    cierres = ordenado["close"].dropna()
    if len(cierres) < 30:
        return None
    sigma = float(np.log(cierres).diff().dropna().tail(VENTANA_VOLATILIDAD_DIAS).std())
    ultimo = float(cierres.iloc[-1])
    fechas = pd.bdate_range(ordenado["fecha"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZONTE_MES_DIAS)
    escala = sigma * np.sqrt(np.arange(1, HORIZONTE_MES_DIAS + 1))
    return {
        "fechas": fechas,
        "ultimo": ultimo,
        "sigma": sigma,
        "p10": ultimo * np.exp(-Z_80 * escala),
        "p25": ultimo * np.exp(-Z_50 * escala),
        "p75": ultimo * np.exp(Z_50 * escala),
        "p90": ultimo * np.exp(Z_80 * escala),
    }
