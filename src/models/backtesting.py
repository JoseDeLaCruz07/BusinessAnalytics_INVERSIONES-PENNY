"""Fase 3 - Backtesting fuera de muestra: senales del modelo vs. comprar y mantener, con comisiones y caida maxima.

El modelo de cada tramo se entrena solo con fechas anteriores (mismos tramos y purga que `evaluar_walk_forward`),
asi que cada operacion se decide con informacion que existia en ese momento.

Estrategias (se rebalancea cada TARGET_HORIZON_DAYS dias de trading, igual que el horizonte del modelo):
  - "senal de alerta": compra en partes iguales los tickers con puntaje >= UMBRAL_SENAL_ALERTA; si no hay, queda en efectivo.
  - "top 3 puntaje": compra los 3 tickers con mayor puntaje de ese dia.
  - "comprar y mantener": compra todos los tickers en partes iguales al inicio del periodo de prueba y los mantiene.

Uso:
    python -m src.models.backtesting                    # usa DATABASE_URL (base local por defecto)
    python -m src.models.backtesting --solo-lectura     # usa SUPABASE_URL_LECTURA en una conexion de solo lectura
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import TARGET_HORIZON_DAYS, UMBRAL_SENAL_ALERTA
from src.models.train_xgboost import FEATURE_COLS, FRACCION_TEST, MIN_FECHAS, N_FOLDS, construir_dataset, nuevo_modelo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
EVIDENCIAS_DIR = ROOT_DIR / "docs" / "evidencias"

COMISION_POR_LADO = 0.001  # 0.1% al comprar y 0.1% al vender
DESLIZAMIENTO_POR_LADO = 0.002  # 0.2%: diferencia entre el precio esperado y el real, alta en penny stocks
SALTO_SOSPECHOSO = 0.60  # un cambio diario de +-60% suele ser un split/reverse split sin ajustar, no un movimiento real
TOP_K = 3


def sesion_solo_lectura():
    """Sesion contra SUPABASE_URL_LECTURA donde Postgres rechaza cualquier escritura."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    url = os.getenv("SUPABASE_URL_LECTURA")
    if not url:
        raise SystemExit("Falta SUPABASE_URL_LECTURA en .env")
    engine = create_engine(url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _solo_lectura(conexion, _):
        with conexion.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")

    return sessionmaker(bind=engine)()


def cargar_precios(session) -> pd.DataFrame:
    from src.db import PrecioOHLCV, Ticker

    filas = (
        session.query(PrecioOHLCV.ticker_id, Ticker.symbol, PrecioOHLCV.fecha, PrecioOHLCV.close)
        .join(Ticker, PrecioOHLCV.ticker_id == Ticker.id)
        .filter(PrecioOHLCV.close.isnot(None))
        .all()
    )
    from src.processing.clean_pipeline import limpiar_ohlcv

    df = pd.DataFrame(filas, columns=["ticker_id", "symbol", "fecha", "close"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return limpiar_ohlcv(df)


def retornos_a_horizonte(precios: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Retorno de comprar al cierre de t y vender al cierre de t+horizonte, y si la ventana tiene un salto sospechoso."""

    def por_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("fecha").copy()
        g["ret_h"] = g["close"].shift(-horizonte) / g["close"] - 1
        salto = (g["close"].pct_change(fill_method=None).abs() > SALTO_SOSPECHOSO).astype(int)
        # Hay salto en la ventana (t, t+h] si alguno de los proximos h cambios diarios lo es.
        g["salto_en_ventana"] = salto[::-1].rolling(horizonte, min_periods=1).max()[::-1].shift(-1).fillna(0).astype(bool)
        return g

    return precios.groupby("symbol", group_keys=False)[precios.columns].apply(por_ticker)


def predicciones_fuera_de_muestra(dataset: pd.DataFrame) -> pd.DataFrame:
    """Puntaje de cada fila del 40% final, con un modelo entrenado solo con fechas anteriores a su tramo (y purga)."""
    datos = dataset.dropna(subset=FEATURE_COLS + ["target"])
    fechas = np.sort(datos["fecha"].unique())
    if len(fechas) < MIN_FECHAS:
        raise SystemExit(f"Solo hay {len(fechas)} fechas; se necesitan al menos {MIN_FECHAS}.")
    inicio_test = int(len(fechas) * (1 - FRACCION_TEST))
    partes = []
    for n, tramo in enumerate(np.array_split(np.arange(inicio_test, len(fechas)), N_FOLDS), start=1):
        train = datos[datos["fecha"] < fechas[max(tramo[0] - TARGET_HORIZON_DAYS, 0)]]
        test = datos[(datos["fecha"] >= fechas[tramo[0]]) & (datos["fecha"] <= fechas[tramo[-1]])].copy()
        modelo = nuevo_modelo(train["target"])
        modelo.fit(train[FEATURE_COLS], train["target"])
        test["puntaje"] = modelo.predict_proba(test[FEATURE_COLS])[:, 1]
        test["tramo"] = n
        partes.append(test)
        logger.info("Tramo %d: train hasta %s (%d filas), test %s a %s (%d filas)", n, pd.Timestamp(train["fecha"].max()).date(),
                    len(train), pd.Timestamp(fechas[tramo[0]]).date(), pd.Timestamp(fechas[tramo[-1]]).date(), len(test))
    return pd.concat(partes, ignore_index=True)


def metricas_clasificacion(oos: pd.DataFrame) -> dict:
    y, p = oos["target"].astype(int), oos["puntaje"]
    pred = (p >= 0.5).astype(int)
    ingenuo = np.zeros(len(y), dtype=int)
    return {
        "n_observaciones": int(len(y)),
        "tasa_base": float(y.mean()),
        "modelo": {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, p)),
            "precision_en_senal_alerta": float(y[p >= UMBRAL_SENAL_ALERTA].mean()) if (p >= UMBRAL_SENAL_ALERTA).any() else None,
            "n_senales_alerta": int((p >= UMBRAL_SENAL_ALERTA).sum()),
        },
        "linea_base_siempre_no_sube": {
            "accuracy": float(accuracy_score(y, ingenuo)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.5,
        },
    }


def caida_maxima(curva: pd.Series) -> float:
    return float((curva / curva.cummax() - 1).min())


def simular(oos: pd.DataFrame, regla: str, costo_por_lado: float, excluir_saltos: bool) -> dict:
    """Rebalanceo cada TARGET_HORIZON_DAYS fechas: cada ventana se invierte todo el capital (o se queda en efectivo)."""
    fechas = np.sort(oos["fecha"].unique())
    fechas_rebalanceo = fechas[::TARGET_HORIZON_DAYS]
    capital, curva, operaciones, ganadoras, ventanas_invertido = 1.0, [], 0, 0, 0
    for fecha in fechas_rebalanceo:
        dia = oos[(oos["fecha"] == fecha) & oos["ret_h"].notna()]
        if excluir_saltos:
            dia = dia[~dia["salto_en_ventana"]]
        if regla == "alerta":
            elegidos = dia[dia["puntaje"] >= UMBRAL_SENAL_ALERTA]
        else:
            elegidos = dia.nlargest(TOP_K, "puntaje")
        if len(elegidos):
            netos = (1 - costo_por_lado) * (1 + elegidos["ret_h"]) * (1 - costo_por_lado) - 1
            capital *= 1 + netos.mean()
            operaciones += len(elegidos)
            ganadoras += int((netos > 0).sum())
            ventanas_invertido += 1
        curva.append((pd.Timestamp(fecha), capital))
    serie = pd.Series([c for _, c in curva], index=[f for f, _ in curva])
    return {
        "retorno_total": float(capital - 1),
        "caida_maxima": caida_maxima(pd.concat([pd.Series([1.0]), serie.reset_index(drop=True)])),
        "operaciones": operaciones,
        "operaciones_ganadoras_pct": float(ganadoras / operaciones) if operaciones else None,
        "ventanas_invertido": f"{ventanas_invertido}/{len(fechas_rebalanceo)}",
        "_curva": serie,
    }


def comprar_y_mantener(precios_test: pd.DataFrame, costo_por_lado: float) -> dict:
    """Todos los tickers en partes iguales desde la primera hasta la ultima fecha de prueba (un costo de entrada y uno de salida)."""
    tabla = precios_test.pivot_table(index="fecha", columns="symbol", values="close").sort_index().ffill()
    tabla = tabla.dropna(axis=1, subset=[tabla.index[0]])
    # Los saltos de +-60% en un dia (splits sin ajustar) se neutralizan: ese dia cuenta como 0% para el ticker.
    diarios = tabla.pct_change(fill_method=None).fillna(0)
    diarios = diarios.mask(diarios.abs() > SALTO_SOSPECHOSO, 0.0)
    valor = (1 + diarios).cumprod().mean(axis=1) * (1 - costo_por_lado)
    valor.iloc[-1] *= 1 - costo_por_lado
    return {
        "retorno_total": float(valor.iloc[-1] - 1),
        "caida_maxima": caida_maxima(pd.concat([pd.Series([1.0]), valor.reset_index(drop=True)])),
        "tickers": int(tabla.shape[1]),
        "_curva": valor,
    }


def guardar_grafico(curvas: dict[str, pd.Series], ruta: Path, titulo: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no esta instalado; se omite el grafico.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for nombre, serie in curvas.items():
        ax.plot(serie.index, serie.values, label=nombre, linewidth=1.8)
    ax.axhline(1.0, color="#999", linewidth=0.8, linestyle="--")
    ax.set_title(titulo)
    ax.set_ylabel("Valor de US$1 invertido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def run(solo_lectura: bool = False) -> dict:
    if solo_lectura:
        session = sesion_solo_lectura()
    else:
        from src.db import get_session, init_db

        init_db()
        session = get_session()
    dataset = construir_dataset(session)
    precios = cargar_precios(session)
    session.close()
    if dataset.empty:
        raise SystemExit("No hay dataset (features + precios) para hacer el backtesting.")

    oos = predicciones_fuera_de_muestra(dataset)
    rets = retornos_a_horizonte(precios, TARGET_HORIZON_DAYS)
    oos = oos.merge(rets[["ticker_id", "fecha", "ret_h", "salto_en_ventana"]], on=["ticker_id", "fecha"], how="left")
    precios_test = precios[(precios["fecha"] >= oos["fecha"].min()) & (precios["fecha"] <= oos["fecha"].max())]
    costo = COMISION_POR_LADO + DESLIZAMIENTO_POR_LADO

    resultados = {"clasificacion": metricas_clasificacion(oos), "estrategias": {}}
    curvas = {}
    for excluir in (False, True):
        sufijo = " (sin ventanas con salto > 60%)" if excluir else ""
        for regla, nombre in (("alerta", "senal de alerta"), ("top", f"top {TOP_K} puntaje")):
            r = simular(oos, regla, costo, excluir)
            curvas[nombre + sufijo] = r.pop("_curva")
            resultados["estrategias"][nombre + sufijo] = r
    for regla, nombre in (("alerta", "senal de alerta"), ("top", f"top {TOP_K} puntaje")):
        r = simular(oos, regla, 0.0, False)
        r.pop("_curva")
        resultados["estrategias"][nombre + " (sin costos, referencia)"] = r
    bh = comprar_y_mantener(precios_test, costo)
    curvas["comprar y mantener"] = bh.pop("_curva")
    resultados["estrategias"]["comprar y mantener"] = bh
    resultados["periodo_prueba"] = [str(oos["fecha"].min().date()), str(oos["fecha"].max().date())]
    resultados["supuestos"] = {
        "comision_por_lado": COMISION_POR_LADO,
        "deslizamiento_por_lado": DESLIZAMIENTO_POR_LADO,
        "rebalanceo_cada_dias": TARGET_HORIZON_DAYS,
        "umbral_alerta": UMBRAL_SENAL_ALERTA,
        "salto_sospechoso": SALTO_SOSPECHOSO,
        "filas_con_salto_en_ventana": int(oos["salto_en_ventana"].fillna(False).sum()),
        "fuente": "Supabase (solo lectura)" if solo_lectura else "DATABASE_URL",
    }

    EVIDENCIAS_DIR.mkdir(parents=True, exist_ok=True)
    hoy = dt.date.today().strftime("%Y%m%d")
    with open(EVIDENCIAS_DIR / f"backtest_{hoy}.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    graficables = {k: v for k, v in curvas.items() if "salto" in k or k == "comprar y mantener"}
    guardar_grafico(graficables, EVIDENCIAS_DIR / f"backtest_{hoy}.png",
                    f"Backtesting fuera de muestra ({resultados['periodo_prueba'][0]} a {resultados['periodo_prueba'][1]}), con costos")
    logger.info("Resultados:\n%s", json.dumps(resultados, indent=2, ensure_ascii=False))
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting fuera de muestra con comisiones y caida maxima")
    parser.add_argument("--solo-lectura", action="store_true", help="leer de SUPABASE_URL_LECTURA sin poder escribir")
    run(solo_lectura=parser.parse_args().solo_lectura)
