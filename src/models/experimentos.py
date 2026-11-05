"""Experimentos para mejorar el XGBoost: compara features, objetivos y configuraciones con la misma evaluacion.

No modifica el modelo publicado: solo imprime una tabla y la guarda en data/experimentos_<fecha>.csv.

Uso:
    python -m src.models.experimentos            # rejilla completa (varios minutos)
    python -m src.models.experimentos --rapido   # rejilla reducida
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.db import Feature, FilingSEC, PrecioOHLCV, Ticker, get_session, init_db
from src.models.train_xgboost import FEATURE_COLS, MIN_FECHAS, MODELO_PARAMS, evaluar_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

RELATIVA = ["rsi", "volumen_relativo", "sentimiento_agregado", "precio_vs_sma"]  # media_movil absoluta delata al ticker
SENT_SUAVE = ["sent_7d", "sent_30d"]
MOMENTUM = ["ret_5d", "ret_10d", "ret_20d"]
VOLATILIDAD = ["vol_10d", "vol_20d", "rango_5d"]
PRECIO = ["dist_max_20d", "dist_min_20d", "gap"]
SEC = ["n_filings_30d", "n_dilucion_30d", "n_8k_30d"]

FEATURES_ANTERIORES = ["rsi", "media_movil", "volumen_relativo", "sentimiento_agregado"]  # antes de adoptar los resultados

GRUPOS = {
    "base (anterior)": FEATURES_ANTERIORES,
    "modelo vigente": FEATURE_COLS,
    "sin sentimiento": ["rsi", "media_movil", "volumen_relativo"],
    "relativa": RELATIVA,
    "relativa + sent. suave": ["rsi", "volumen_relativo", "precio_vs_sma"] + SENT_SUAVE,
    "relativa + momentum": RELATIVA + MOMENTUM,
    "relativa + volatilidad": RELATIVA + VOLATILIDAD,
    "relativa + SEC": RELATIVA + SEC,
    "todo": RELATIVA + SENT_SUAVE + MOMENTUM + VOLATILIDAD + PRECIO + SEC,
}
OBJETIVOS = [(3, 0.05), (3, 0.03), (5, 0.05), (5, 0.03), (10, 0.05), (21, 0.05), (21, 0.10)]  # (dias, subida minima)
CONFIGS = {
    "anterior": dict(n_estimators=200, max_depth=4, learning_rate=0.05),
    "regularizado": MODELO_PARAMS,
}


def cargar_precios(session) -> pd.DataFrame:
    filas = (
        session.query(PrecioOHLCV, Ticker.symbol)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .order_by(Ticker.symbol, PrecioOHLCV.fecha)
        .all()
    )
    df = pd.DataFrame(
        [
            {"ticker_id": p.ticker_id, "symbol": s, "fecha": pd.Timestamp(p.fecha), "open": p.open, "high": p.high,
             "low": p.low, "close": p.close}
            for p, s in filas
        ]
    )
    return df.dropna(subset=["close"]) if not df.empty else df


def cargar_features(session) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker_id": f.ticker_id, "fecha": pd.Timestamp(f.fecha), "rsi": f.rsi, "media_movil": f.media_movil,
             "volumen_relativo": f.volumen_relativo, "sentimiento_agregado": f.sentimiento_agregado}
            for f in session.query(Feature).all()
        ]
    )


def cargar_filings(session) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker_id": f.ticker_id, "fecha": pd.Timestamp(f.fecha), "formulario": f.formulario, "categoria": f.categoria}
            for f in session.query(FilingSEC).all()
        ],
        columns=["ticker_id", "fecha", "formulario", "categoria"],
    )


def _conteo_ventana(fechas_filings: np.ndarray, dias: pd.DatetimeIndex, ventana_dias: int = 30) -> np.ndarray:
    """Filings en [d - ventana, d): solo anteriores al dia d, para no usar informacion que aun no se conocia."""
    fin = np.searchsorted(fechas_filings, dias.values, side="left")
    ini = np.searchsorted(fechas_filings, (dias - pd.Timedelta(days=ventana_dias)).values, side="left")
    return fin - ini


def features_por_ticker(precios: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Features que dependen solo de datos hasta el dia t (sin mirar adelante), calculadas por ticker."""
    partes = []
    for ticker_id, g in precios.groupby("ticker_id"):
        g = g.sort_values("fecha").copy()
        c = g["close"]
        retorno = c.pct_change(fill_method=None)
        for k in (5, 10, 20):
            g[f"ret_{k}d"] = c.pct_change(k, fill_method=None)
        for k in (10, 20):
            g[f"vol_{k}d"] = retorno.rolling(k).std()
        g["rango_5d"] = ((g["high"] - g["low"]) / c).rolling(5).mean()
        g["dist_max_20d"] = c / c.rolling(20).max() - 1
        g["dist_min_20d"] = c / c.rolling(20).min() - 1
        g["gap"] = g["open"] / c.shift(1) - 1

        f = filings[filings["ticker_id"] == ticker_id].sort_values("fecha")
        dias = pd.DatetimeIndex(g["fecha"])
        for nombre, sub in (
            ("n_filings_30d", f),
            ("n_dilucion_30d", f[f["categoria"] == "Dilucion"]),
            ("n_8k_30d", f[f["formulario"].str.upper() == "8-K"]),
        ):
            g[nombre] = _conteo_ventana(sub["fecha"].values, dias)
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


def construir_base(session) -> pd.DataFrame:
    precios, features, filings = cargar_precios(session), cargar_features(session), cargar_filings(session)
    if precios.empty or features.empty:
        return pd.DataFrame()
    df = features_por_ticker(precios, filings).merge(features, on=["ticker_id", "fecha"], how="inner")
    df["precio_vs_sma"] = df["close"] / df["media_movil"] - 1
    df = df.sort_values(["ticker_id", "fecha"])
    for ventana in (7, 30):
        df[f"sent_{ventana}d"] = df.groupby("ticker_id")["sentimiento_agregado"].transform(lambda s: s.rolling(ventana, min_periods=1).mean())
    return df


def con_objetivo(df: pd.DataFrame, dias: int, umbral: float) -> pd.DataFrame:
    """target = 1 si el cierre sube mas de `umbral` en `dias` dias de trading; sin target donde no hay futuro."""
    futuro = df.groupby("ticker_id")["close"].shift(-dias)
    out = df.copy()
    out["target"] = ((futuro / out["close"] - 1) > umbral).astype(float)
    return out[futuro.notna()]


def run(rapido: bool = False) -> pd.DataFrame:
    init_db()
    session = get_session()
    base = construir_base(session)
    session.close()
    if base.empty:
        logger.error("Sin precios o features en la base: corre primero la ingesta y feature_engineering.")
        return pd.DataFrame()

    grupos = {k: v for k, v in GRUPOS.items() if not rapido or k in ("base (anterior)", "relativa", "todo")}
    objetivos = [(3, 0.05), (5, 0.03), (10, 0.05)] if rapido else OBJETIVOS
    configs = {"regularizado": CONFIGS["regularizado"]} if rapido else CONFIGS

    resultados = []
    hechos = 0
    total = len(grupos) * len(objetivos) * len(configs)
    for dias, umbral in objetivos:
        df = con_objetivo(base, dias, umbral)
        for nombre_grupo, columnas in grupos.items():
            for nombre_config, config in configs.items():
                r = evaluar_walk_forward(df, columnas, purga_dias=dias, params=config)
                hechos += 1
                if r is not None:
                    resultados.append({"grupo": nombre_grupo, "config": nombre_config, "dias": dias, "umbral": umbral, **r})
                logger.info("%d/%d  %s | %s | %dd >%.0f%%", hechos, total, nombre_grupo, nombre_config, dias, umbral * 100)

    if not resultados:
        logger.error("Ningun experimento tuvo datos suficientes (se necesitan al menos %d fechas con ambas clases).", MIN_FECHAS)
        return pd.DataFrame()
    tabla = pd.DataFrame(resultados).sort_values("auc", ascending=False).reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    salida = DATA_DIR / f"experimentos_{dt.date.today():%Y%m%d}.csv"
    tabla.to_csv(salida, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    formato = tabla.copy()
    for col in ("auc", "auc_std", "lift_top10"):
        formato[col] = formato[col].map(lambda v: f"{v:.3f}" if col != "lift_top10" else f"{v:.2f}x")
    formato["tasa_base"] = tabla["tasa_base"].map(lambda v: f"{v:.1%}")
    formato["umbral"] = tabla["umbral"].map(lambda v: f"{v:.0%}")
    print("\nMejores 15 (ordenado por AUC medio):")
    print(formato.drop(columns="n").head(15).to_string(index=False))
    actual = tabla[(tabla["grupo"] == "base (anterior)") & (tabla["config"] == "anterior") & (tabla["dias"] == 3) & (tabla["umbral"] == 0.05)]
    if not actual.empty:
        print(f"\nModelo anterior (base, 3 dias, >5%): AUC {actual.iloc[0]['auc']:.3f}")
    print(
        "\nLectura: el azar da AUC 0.50. Con tantas combinaciones algunas salen bien por suerte; solo confia en las que "
        "ganan mas de ~0.03 al modelo anterior, con 'folds_sobre_azar' alto (4/4) y poca desviacion.\n"
        "AUC de objetivos distintos mide preguntas distintas: compara tambien 'lift_top10' (cuantas veces mejor que la tasa base "
        "acierta el 10% de dias que el modelo puntua mas alto)."
    )
    logger.info("Tabla completa guardada en %s", salida)
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimentos de mejora del modelo XGBoost")
    parser.add_argument("--rapido", action="store_true", help="rejilla reducida")
    args = parser.parse_args()
    run(rapido=args.rapido)
