"""Fase 4 - Dashboard Streamlit: reportes descriptivo, predictivo y prescriptivo.

Uso:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

# Debe ser el primer comando de Streamlit del script, o lanza StreamlitAPIException.
st.set_page_config(page_title="Penny Stocks Analytics", layout="wide")

# Streamlit Cloud usa st.secrets en vez de variables de entorno; las copiamos a
# os.environ antes de importar src.config, que las lee con os.getenv().
# En local, el .env ya aporta DATABASE_URL y se evita el aviso "No secrets found".
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("DATABASE_URL"):
    try:
        for _key, _value in st.secrets.items():
            os.environ.setdefault(_key, str(_value))
    except Exception:
        pass

import datetime as dt
import html
import json
import math
import types
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sqlalchemy import and_, func

from src.config import (
    TARGET_HORIZON_DAYS,
    TARGET_PCT_THRESHOLD,
    UMBRAL_SENAL_ALERTA,
    UMBRAL_SENAL_MONITOREAR,
)
from src.db import (
    Feature,
    PrecioOHLCV,
    Prediccion,
    RatioFinanciero,
    Recomendacion,
    Sentimiento,
    Texto,
    Ticker,
    get_session,
    init_db,
)
from src.models.train_xgboost import FEATURE_COLS, PUBLISHED_MODEL_PATH
from src.reports.business_rules import generar_recomendacion

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = DATA_DIR / "models"

RANGOS_FECHA = {"30 dias": 30, "90 dias": 90, "6 meses": 182, "1 anio": 365, "Todo": None}
VENTANA_SENTIMIENTO_DIAS = 30
TTL_DATOS = 600
# La sesion de DB y el modelo no se pueden hashear; no cambian dentro de una sesion de la app, asi que no entran a la clave.
HASH_SESION_MODELO = {
    "sqlalchemy.orm.session.Session": lambda _: 0,
    "xgboost.sklearn.XGBClassifier": lambda _: 0,
}


def con_cache(funcion):
    return st.cache_data(ttl=TTL_DATOS, show_spinner=False, hash_funcs=HASH_SESION_MODELO)(funcion)

FONT_DISPLAY = "'Sora', system-ui, sans-serif"
FONT_BODY = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace"

# Tokens de color light/dark; graficos y CSS los consumen desde aqui.
TEMAS = {
    "light": {
        "bg": "#f5f6fb",
        "surface": "#ffffff",
        "surface_2": "#eef0f8",
        "border": "#dfe3ee",
        "text": "#12131a",
        "text_muted": "#5b6172",
        "accent": "#4f46e5",
        "accent_2": "#d97706",
        "success": "#16a34a",
        "danger": "#dc2626",
        "neutral": "#6b7280",
        "grid": "#e7e9f2",
        "chip_bg": "#f1f1f7",
        "gauge_steps": ["#e5e7eb", "#fde68a", "#fecaca"],
        "feature_colors": {
            "rsi": "#2a78d6",
            "media_movil": "#eb6834",
            "volumen_relativo": "#1baf7a",
            "sentimiento_agregado": "#c98500",
        },
    },
    "dark": {
        "bg": "#0b0e14",
        "surface": "#131722",
        "surface_2": "#0e1117",
        "border": "#262b3a",
        "text": "#e8eaf0",
        "text_muted": "#8b93a7",
        "accent": "#818cf8",
        "accent_2": "#f59e0b",
        "success": "#22c55e",
        "danger": "#f87171",
        "neutral": "#9aa3b2",
        "grid": "#232733",
        "chip_bg": "#1b2030",
        "gauge_steps": ["#232733", "#5b4a1f", "#5c2323"],
        "feature_colors": {
            "rsi": "#6ea8fe",
            "media_movil": "#f0895a",
            "volumen_relativo": "#35d0a0",
            "sentimiento_agregado": "#e6ae2e",
        },
    },
}

COLOR_SENAL_KEY = {"senal de alerta": "danger", "monitorear": "accent_2", "sin senal": "neutral"}


def inyectar_css(tema: dict) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, .stApp {{
            background-color: {tema['bg']} !important;
            color: {tema['text']};
            font-family: {FONT_BODY};
        }}
        h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {{
            font-family: {FONT_DISPLAY} !important;
            color: {tema['text']} !important;
            letter-spacing: -0.01em;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {tema['surface_2']} !important;
            border-right: 1px solid {tema['border']};
        }}
        section[data-testid="stSidebar"] * {{ color: {tema['text']}; }}

        [data-testid="stMetric"] {{
            background: {tema['surface']};
            border: 1px solid {tema['border']};
            border-radius: 12px;
            padding: 14px 16px 10px;
        }}
        [data-testid="stMetricValue"] {{
            font-family: {FONT_MONO} !important;
            color: {tema['text']} !important;
        }}
        [data-testid="stMetricLabel"] p {{ color: {tema['text_muted']} !important; font-family: {FONT_BODY}; }}

        [data-testid="stTabs"] button[data-baseweb="tab"] {{
            font-family: {FONT_DISPLAY};
            font-weight: 600;
            color: {tema['text_muted']};
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: {tema['accent']} !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: {tema['accent']} !important;
        }}
        [data-testid="stTabs"] [data-baseweb="tab-border"] {{ background-color: {tema['border']}; }}

        div[data-testid="stExpander"], div[data-testid="stAlert"] {{
            background-color: {tema['surface']};
            border: 1px solid {tema['border']};
            border-radius: 10px;
        }}
        hr {{ border-color: {tema['border']}; }}
        code {{ font-family: {FONT_MONO}; }}

        [data-testid="stWidgetLabel"] p, [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
            color: {tema['text']} !important;
        }}

        header[data-testid="stHeader"] {{
            background-color: {tema['bg']} !important;
        }}
        [data-testid="stToolbar"] button, [data-testid="stToolbar"] span {{
            color: {tema['text_muted']} !important;
        }}
        [data-testid="stToolbar"] svg {{ fill: {tema['text_muted']} !important; }}

        div[data-baseweb="select"] > div {{
            background-color: {tema['surface']} !important;
            border-color: {tema['border']} !important;
            color: {tema['text']} !important;
        }}
        div[data-baseweb="popover"] {{
            background-color: {tema['surface']} !important;
        }}
        div[data-baseweb="popover"] ul[data-baseweb="menu"] {{
            background-color: {tema['surface']} !important;
            border: 1px solid {tema['border']};
        }}
        div[data-baseweb="popover"] li {{
            background-color: {tema['surface']} !important;
            color: {tema['text']} !important;
        }}
        div[data-baseweb="popover"] li:hover {{
            background-color: {tema['surface_2']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_tabla_html(
    df: pd.DataFrame,
    columnas: list[tuple[str, str]],
    tema: dict,
    numericas: tuple[str, ...] = (),
    largas: tuple[str, ...] = (),
    resaltar: tuple[str, str] | None = None,
    max_height: int | None = None,
) -> None:
    """Tabla HTML propia: st.dataframe() no sigue el toggle de modo oscuro."""
    def clase(col: str) -> str:
        return ' class="num"' if col in numericas else (' class="txt"' if col in largas else "")

    filas_html = ""
    for _, fila in df.iterrows():
        activa = resaltar is not None and str(fila[resaltar[0]]) == resaltar[1]
        celdas = "".join(f"<td{clase(col)}>{fila[col]}</td>" for col, _ in columnas)
        filas_html += f'<tr class="activa">{celdas}</tr>' if activa else f"<tr>{celdas}</tr>"

    encabezado = "".join(f"<th{clase(col)}>{etiqueta}</th>" for col, etiqueta in columnas)
    alto = f"max-height:{max_height}px;" if max_height else ""

    st.markdown(
        f"""<style>
.tbl-pro-wrap {{ overflow:auto; {alto} border:1px solid {tema['border']}; border-radius:12px; background:{tema['surface']}; }}
.tbl-pro {{ width:100%; border-collapse:separate; border-spacing:0; font-family:{FONT_BODY}; font-size:0.9rem; color:{tema['text']}; }}
.tbl-pro th {{ position:sticky; top:0; z-index:1; background:{tema['surface_2']}; color:{tema['text_muted']}; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; text-align:left; padding:11px 16px; border:none; border-bottom:1px solid {tema['border']}; white-space:nowrap; }}
.tbl-pro td {{ padding:10px 16px; border:none; border-bottom:1px solid {tema['border']}; white-space:nowrap; }}
.tbl-pro tbody tr:last-child td {{ border-bottom:none; }}
.tbl-pro tbody tr:nth-child(even) {{ background:{hex_to_rgba(tema['accent'], 0.04)}; }}
.tbl-pro tbody tr:hover {{ background:{hex_to_rgba(tema['accent'], 0.10)}; }}
.tbl-pro tbody tr.activa {{ background:{hex_to_rgba(tema['accent'], 0.14)}; }}
.tbl-pro tbody tr.activa td:first-child {{ box-shadow:inset 3px 0 0 {tema['accent']}; }}
.tbl-pro .num {{ text-align:right; font-family:{FONT_MONO}; font-variant-numeric:tabular-nums; }}
.tbl-pro .txt {{ white-space:normal; min-width:320px; line-height:1.45; }}
.tbl-pro a {{ color:{tema['accent']}; font-weight:600; text-decoration:none; }}
.tbl-pro a:hover {{ text-decoration:underline; }}
</style>
<div class="tbl-pro-wrap"><table class="tbl-pro"><thead><tr>{encabezado}</tr></thead><tbody>{filas_html}</tbody></table></div>""",
        unsafe_allow_html=True,
    )


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def aplicar_tema_plotly(fig: go.Figure, tema: dict) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=tema["surface"],
        plot_bgcolor=tema["surface"],
        font=dict(family=FONT_BODY, color=tema["text"], size=12),
    )
    fig.update_xaxes(gridcolor=tema["grid"], zerolinecolor=tema["grid"], linecolor=tema["border"])
    fig.update_yaxes(gridcolor=tema["grid"], zerolinecolor=tema["grid"], linecolor=tema["border"])
    return fig


@st.cache_resource
def get_db_session():
    init_db()
    return get_session()


@con_cache
def cargar_tickers(session) -> list[str]:
    return [t.symbol for t in session.query(Ticker).order_by(Ticker.symbol).all()]


@con_cache
def cargar_ticker_info(session, symbol: str) -> types.SimpleNamespace:
    t = session.query(Ticker).filter_by(symbol=symbol).first()
    return types.SimpleNamespace(id=t.id, symbol=t.symbol, nombre=t.nombre, sector=t.sector)


@con_cache
def cargar_ohlcv_df(session, ticker_id: int) -> pd.DataFrame:
    rows = session.query(PrecioOHLCV).filter_by(ticker_id=ticker_id).order_by(PrecioOHLCV.fecha).all()
    df = pd.DataFrame(
        [
            {"fecha": r.fecha, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volumen": r.volumen}
            for r in rows
        ]
    )
    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def grafico_dispersion_mercado(df: pd.DataFrame, tema: dict, ticker_actual: str) -> go.Figure:
    """Sentimiento (X) vs probabilidad XGBoost (Y) de todos los tickers.

    El cuadrante superior derecho (sentimiento > 0 y probabilidad >=
    UMBRAL_SENAL_MONITOREAR) es el "primer cuadrante" que se usa como filtro.
    Color = estado (cuadrante positivo / resto / ticker elegido), no identidad:
    con 18 tickers un color por bolita no se distingue; el nombre va en la
    etiqueta (solo cuadrante positivo y elegido) y en el hover. Tamano = market cap.
    """
    positivo = (df["sentimiento"] > 0) & (df["probabilidad"] >= UMBRAL_SENAL_MONITOREAR)
    elegido = df["symbol"] == ticker_actual

    def tamano(mc, extra=0):
        mc = mc if pd.notna(mc) and mc > 0 else 30
        escala = (math.log10(mc) - math.log10(30)) / (math.log10(1500) - math.log10(30))
        return 12 + 22 * min(max(escala, 0), 1) + extra

    grupos = [
        ("Cuadrante positivo", df[positivo & ~elegido], tema["success"], True, 0, tema["surface"], 2),
        ("Resto", df[~positivo & ~elegido], tema["neutral"], False, 0, tema["surface"], 2),
        ("Seleccionado", df[elegido], tema["accent"], True, 6, tema["text"], 3),
    ]

    fig = go.Figure()
    for nombre, sub, color, con_etiqueta, extra, borde, grosor in grupos:
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                name=nombre,
                x=sub["sentimiento"],
                y=sub["probabilidad"],
                mode="markers+text" if con_etiqueta else "markers",
                text=[f"<b>{s}</b>" for s in sub["symbol"]] if con_etiqueta else None,
                textposition="top center",
                customdata=sub[["symbol", "market_cap"]],
                marker=dict(
                    size=[tamano(mc, extra) for mc in sub["market_cap"]],
                    color=color,
                    opacity=0.9,
                    line=dict(width=grosor, color=borde),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Sentimiento: %{x:.2f}<br>Probabilidad: %{y:.0%}"
                    "<br>Market cap: $%{customdata[1]:,.0f}M<extra></extra>"
                ),
            )
        )

    # El rango sigue a los datos (incluyendo el 0) para que el cuadrante sombreado no estire el eje.
    x_min, x_max = min(df["sentimiento"].min(), 0), max(df["sentimiento"].max(), 0)
    margen = max(0.1, 0.15 * (x_max - x_min))
    rango_x = [x_min - margen, x_max + margen]
    fig.add_shape(
        type="rect", xref="x", yref="y", x0=0, x1=rango_x[1], y0=UMBRAL_SENAL_MONITOREAR, y1=1,
        fillcolor=hex_to_rgba(tema["success"], 0.08), line_width=0, layer="below",
    )
    fig.add_vline(x=0, line=dict(color=tema["text_muted"], width=1))
    fig.add_hline(y=UMBRAL_SENAL_MONITOREAR, line=dict(color=tema["text_muted"], width=1, dash="dash"))
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(title=f"Sentimiento promedio ({VENTANA_SENTIMIENTO_DIAS} dias)", range=rango_x),
        yaxis=dict(title="Probabilidad de suba (XGBoost)", range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=tema["text"])),
        showlegend=True,
    )
    return aplicar_tema_plotly(fig, tema)


def chip_html(texto: str, color: str) -> str:
    return (
        f'<span style="display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; '
        f'font-weight:600; color:{color}; background:{hex_to_rgba(color, 0.15)};">{texto}</span>'
    )


def tabla_indicadores_pe(df: pd.DataFrame, tema: dict, ticker_actual: str) -> None:
    """P/E = precio / EPS: se muestran juntos con P/B y market cap para ver que mueve el P/E de cada ticker."""
    nd = f'<span style="color:{tema["text_muted"]};">N/D</span>'

    def dinero(v):
        return f"{'-' if v < 0 else ''}${abs(v):.2f}"

    def eps_html(v):
        if pd.isna(v):
            return nd
        return f'<span style="color:{tema["success"] if v > 0 else tema["danger"]};">{dinero(v)}</span>'

    def resultado_html(v):
        if pd.isna(v):
            return nd
        return chip_html("Ganancia", tema["success"]) if v > 0 else chip_html("Perdida", tema["danger"])

    def market_cap_html(v):
        if pd.isna(v):
            return nd
        return f"${v / 1000:.2f}B" if v >= 1000 else f"${v:,.0f}M"

    tabla = df.sort_values("market_cap", ascending=False, na_position="last").copy()
    tabla["ticker"] = tabla["symbol"].map(lambda v: f"<b>{v}</b>")
    tabla["precio_fmt"] = tabla["precio"].map(lambda v: dinero(v) if pd.notna(v) else nd)
    tabla["eps_fmt"] = tabla["eps"].map(eps_html)
    tabla["resultado"] = tabla["eps"].map(resultado_html)
    tabla["pe_fmt"] = tabla["pe_ratio"].map(lambda v: f"{v:.1f}" if pd.notna(v) and v > 0 else nd)
    tabla["pb_fmt"] = tabla["pb_ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else nd)
    tabla["market_cap_fmt"] = tabla["market_cap"].map(market_cap_html)

    renderizar_tabla_html(
        tabla,
        [
            ("ticker", "Ticker"),
            ("precio_fmt", "Precio"),
            ("eps_fmt", '<span title="Utilidad por accion">EPS</span>'),
            ("resultado", '<span title="Con EPS negativo la empresa tiene perdidas y no hay P/E">Resultado</span>'),
            ("pe_fmt", '<span title="Precio / EPS. Solo existe con EPS positivo">P/E</span>'),
            ("pb_fmt", '<span title="Precio / valor en libros">P/B</span>'),
            ("market_cap_fmt", "Market cap"),
        ],
        tema,
        numericas=("precio_fmt", "eps_fmt", "pe_fmt", "pb_fmt", "market_cap_fmt"),
        resaltar=("symbol", ticker_actual),
        max_height=420,
    )


@con_cache
def cargar_textos_df(session, ticker_id: int) -> pd.DataFrame:
    rows = (
        session.query(Texto.fecha, Texto.fuente, Texto.contenido, Texto.url, Sentimiento.etiqueta, Sentimiento.score)
        .join(Sentimiento, Sentimiento.texto_id == Texto.id)
        .filter(Texto.ticker_id == ticker_id)
        .order_by(Texto.fecha.desc())
        .limit(500)
        .all()
    )
    df = pd.DataFrame(rows, columns=["fecha", "fuente", "contenido", "url", "etiqueta", "score"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def texto_seguro(texto: str, limite: int | None = None) -> str:
    """Los textos son de terceros: se escapa el HTML y el `$` (Streamlit lo toma como LaTeX) y se aplana el espacio."""
    plano = " ".join(str(texto).split())
    if limite is not None and len(plano) > limite:
        plano = plano[: limite - 1].rstrip() + "…"
    return html.escape(plano, quote=True).replace("$", "&#36;")


def tabla_textos(df: pd.DataFrame, tema: dict, symbol: str) -> None:
    nd = f'<span style="color:{tema["text_muted"]};">N/D</span>'
    fuentes = {"noticia": ("Noticia", tema["accent"]), "stocktwits": ("StockTwits", tema["accent_2"])}
    sentimientos = {
        "positivo": ("Positivo", tema["success"]),
        "negativo": ("Negativo", tema["danger"]),
        "neutral": ("Neutral", tema["neutral"]),
    }

    def fuente_html(fuente):
        nombre, color = fuentes.get(fuente, (texto_seguro(fuente), tema["neutral"]))
        return chip_html(nombre, color)

    def sentimiento_html(fila):
        nombre, color = sentimientos.get(fila["etiqueta"], (texto_seguro(fila["etiqueta"]), tema["neutral"]))
        return chip_html(f"{nombre} {fila['score']:.0%}", color)

    def enlace_html(fila):
        if fila["fuente"] == "stocktwits":
            # La URL guardada de cada mensaje no es un permalink real; se enlaza a la pagina del ticker.
            url, texto = f"https://stocktwits.com/symbol/{symbol}", "Ver ticker"
        else:
            url, texto = fila["url"], "Ver"
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            destino = html.escape(url, quote=True).replace("$", "&#36;")
            return f'<a href="{destino}" target="_blank" rel="noopener noreferrer">{texto}</a>'
        return nd

    tabla = df.copy()
    tabla["fecha_fmt"] = tabla["fecha"].dt.strftime("%Y-%m-%d")
    tabla["fuente_fmt"] = tabla["fuente"].map(fuente_html)
    tabla["texto_fmt"] = tabla["contenido"].map(
        lambda t: f'<span title="{texto_seguro(t)}">{texto_seguro(t, 200)}</span>'
    )
    tabla["sentimiento_fmt"] = tabla.apply(sentimiento_html, axis=1)
    tabla["link_fmt"] = tabla.apply(enlace_html, axis=1)

    renderizar_tabla_html(
        tabla,
        [
            ("fecha_fmt", "Fecha"),
            ("fuente_fmt", "Fuente"),
            ("texto_fmt", "Texto"),
            ("sentimiento_fmt", "Sentimiento"),
            ("link_fmt", "Link"),
        ],
        tema,
        largas=("texto_fmt",),
        max_height=460,
    )


def filtrar_por_rango(df: pd.DataFrame, dias: int | None) -> pd.DataFrame:
    if df.empty or dias is None:
        return df
    limite = df["fecha"].max() - pd.Timedelta(days=dias)
    return df[df["fecha"] >= limite]


@con_cache
def cargar_sentimiento_diario_df(session, ticker_id: int) -> pd.DataFrame:
    rows = (
        session.query(Texto.fecha, Sentimiento.score, Sentimiento.etiqueta)
        .join(Sentimiento, Sentimiento.texto_id == Texto.id)
        .filter(Texto.ticker_id == ticker_id)
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["fecha", "sentimiento"])
    df = pd.DataFrame(rows, columns=["fecha", "score", "etiqueta"])
    signo = {"positivo": 1, "negativo": -1, "neutral": 0}
    df["sentimiento"] = df["score"] * df["etiqueta"].map(signo).fillna(0)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
    return df.groupby("fecha")["sentimiento"].mean().reset_index()


@con_cache
def cargar_ratios_df(session, ticker_id: int) -> pd.DataFrame:
    rows = session.query(RatioFinanciero).filter_by(ticker_id=ticker_id).order_by(RatioFinanciero.fecha.desc()).all()
    return pd.DataFrame(
        [
            {
                "fecha": r.fecha,
                "market_cap": r.market_cap,
                "pe_ratio": r.pe_ratio,
                "pb_ratio": r.pb_ratio,
                "eps": r.eps,
                "shares_outstanding": r.shares_outstanding,
            }
            for r in rows
        ]
    )


@con_cache
def ultimo_feature_row(session, ticker_id: int) -> dict | None:
    row = session.query(Feature).filter_by(ticker_id=ticker_id).order_by(Feature.fecha.desc()).first()
    if row is None:
        return None
    return {
        "fecha": row.fecha,
        "rsi": row.rsi,
        "media_movil": row.media_movil,
        "volumen_relativo": row.volumen_relativo,
        "sentimiento_agregado": row.sentimiento_agregado,
    }


@st.cache_resource
def cargar_ultimo_modelo():
    # Modelo publicado (versionado en git) -- el unico que existe en la nube.
    if PUBLISHED_MODEL_PATH.exists():
        modelo = xgb.XGBClassifier()
        modelo.load_model(PUBLISHED_MODEL_PATH)
        version_path = PUBLISHED_MODEL_PATH.parent / "xgboost_latest.version.txt"
        modelo_version = version_path.read_text().strip() if version_path.exists() else "xgboost-latest"
        return modelo, modelo_version

    # Fallback local: ultimo modelo en data/models/ (gitignored).
    if not MODELS_DIR.exists():
        return None, None
    modelos = sorted(MODELS_DIR.glob("xgboost_*.json"))
    if not modelos:
        return None, None
    modelo_path = modelos[-1]
    modelo = xgb.XGBClassifier()
    modelo.load_model(modelo_path)
    modelo_version = modelo_path.stem.replace("xgboost_", "xgboost-")
    return modelo, modelo_version


def guardar_prediccion_y_recomendacion(session, ticker_id: int, modelo_version: str, prob: float, features: dict, recomendacion: dict) -> None:
    hoy = dt.date.today()
    existe = session.query(Prediccion).filter_by(ticker_id=ticker_id, fecha=hoy, modelo_version=modelo_version).first()
    if existe:
        return
    pred = Prediccion(
        ticker_id=ticker_id,
        fecha=hoy,
        modelo_version=modelo_version,
        prediccion="sube" if prob >= 0.5 else "no_sube",
        score=prob,
        features_importantes=json.dumps(features, default=str),
    )
    session.add(pred)
    session.commit()
    session.refresh(pred)
    session.add(
        Recomendacion(
            prediccion_id=pred.id,
            tipo_senal=recomendacion["tipo_senal"],
            justificacion=recomendacion["justificacion"],
        )
    )
    session.commit()


def grafico_precio(df: pd.DataFrame, tema: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["fecha"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=tema["success"],
            decreasing_line_color=tema["danger"],
            name="",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.08)),
        yaxis=dict(title="USD", tickprefix="$"),
    )
    return aplicar_tema_plotly(fig, tema)


def grafico_volumen(df: pd.DataFrame, tema: dict) -> go.Figure:
    colores = [tema["success"] if chg >= 0 else tema["danger"] for chg in df["close"].diff().fillna(0)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["fecha"],
            y=df["volumen"],
            marker_color=colores,
            hovertemplate="%{x|%d %b %Y}<br>Volumen: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="Acciones"))
    return aplicar_tema_plotly(fig, tema)


def grafico_sentimiento(df: pd.DataFrame, tema: dict) -> go.Figure:
    colores = [tema["success"] if v > 0 else (tema["danger"] if v < 0 else tema["neutral"]) for v in df["sentimiento"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["fecha"],
            y=df["sentimiento"],
            marker_color=colores,
            hovertemplate="%{x|%d %b %Y}<br>Sentimiento: %{y:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Score (-1 a 1)", range=[-1, 1], zeroline=True, zerolinecolor=tema["grid"]),
    )
    return aplicar_tema_plotly(fig, tema)


def grafico_gauge(prob: float, tema: dict) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": tema["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": tema["text_muted"]},
                "bar": {"color": tema["accent"]},
                "bgcolor": tema["surface"],
                "steps": [
                    {"range": [0, UMBRAL_SENAL_MONITOREAR * 100], "color": tema["gauge_steps"][0]},
                    {"range": [UMBRAL_SENAL_MONITOREAR * 100, UMBRAL_SENAL_ALERTA * 100], "color": tema["gauge_steps"][1]},
                    {"range": [UMBRAL_SENAL_ALERTA * 100, 100], "color": tema["gauge_steps"][2]},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0))
    return aplicar_tema_plotly(fig, tema)


def grafico_zona_objetivo(ohlcv_df: pd.DataFrame, prob: float, tema: dict) -> go.Figure:
    """Precio reciente + zona objetivo (precio y horizonte) que el modelo esta evaluando."""
    recientes = ohlcv_df.tail(45)
    ultimo_close = float(recientes.iloc[-1]["close"])
    ultima_fecha = recientes.iloc[-1]["fecha"]
    precio_objetivo = ultimo_close * (1 + TARGET_PCT_THRESHOLD)
    fecha_horizonte = ultima_fecha + pd.Timedelta(days=TARGET_HORIZON_DAYS)

    if prob >= UMBRAL_SENAL_ALERTA:
        color_zona = hex_to_rgba(tema["danger"], 0.12)
    elif prob >= UMBRAL_SENAL_MONITOREAR:
        color_zona = hex_to_rgba(tema["accent_2"], 0.12)
    else:
        color_zona = hex_to_rgba(tema["neutral"], 0.10)

    # Margen despues del horizonte para que la zona no pegue con el borde del grafico.
    fecha_fin_eje = fecha_horizonte + pd.Timedelta(days=max(3, TARGET_HORIZON_DAYS))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recientes["fecha"],
            y=recientes["close"],
            mode="lines",
            line=dict(color=tema["accent"], width=2),
            name="Precio",
            hovertemplate="%{x|%d %b %Y}<br>Cierre: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_vrect(x0=ultima_fecha, x1=fecha_horizonte, fillcolor=color_zona, line_width=0)
    fig.add_hline(y=precio_objetivo, line=dict(color=tema["text_muted"], width=1, dash="dash"))
    fig.add_vline(x=ultima_fecha, line=dict(color=tema["border"], width=1, dash="dot"))

    # Anotaciones en coordenadas de "paper" (no se cortan con el rango del eje).
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=1.12,
        showarrow=False,
        align="left",
        font=dict(size=12, color=tema["text_muted"]),
        text=(
            f"Objetivo: ${precio_objetivo:.2f} (+{TARGET_PCT_THRESHOLD:.0%}) · "
            f"Horizonte: {TARGET_HORIZON_DAYS} dias de trading (zona sombreada)"
        ),
    )

    fig.update_layout(
        height=320,
        margin=dict(l=0, r=10, t=45, b=0),
        xaxis=dict(range=[recientes["fecha"].min(), fecha_fin_eje]),
        yaxis=dict(title="USD", tickprefix="$"),
        showlegend=False,
    )
    return aplicar_tema_plotly(fig, tema)


def grafico_feature_importance(modelo, tema: dict) -> go.Figure:
    importancias = pd.Series(dict(zip(FEATURE_COLS, modelo.feature_importances_))).sort_values(ascending=True)
    colores = [tema["feature_colors"][nombre] for nombre in importancias.index]
    fig = go.Figure(
        go.Bar(
            x=importancias.values,
            y=importancias.index,
            orientation="h",
            marker_color=colores,
            text=[f"{v:.2f}" for v in importancias.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=260, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, importancias.max() * 1.25]))
    return aplicar_tema_plotly(fig, tema)


def cargar_historial_predicciones(session, ticker_id: int, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Predicciones pasadas de este ticker + si acertaron o no (o "pendiente" si aun no vence el horizonte)."""
    predicciones = (
        session.query(Prediccion).filter_by(ticker_id=ticker_id).order_by(Prediccion.fecha.desc()).limit(30).all()
    )
    if not predicciones or ohlcv_df.empty:
        return pd.DataFrame()

    ohlcv_ordenado = ohlcv_df.sort_values("fecha").reset_index(drop=True)
    fecha_a_idx = {f.date(): i for i, f in enumerate(ohlcv_ordenado["fecha"])}

    recomendaciones = {
        r.prediccion_id: r
        for r in session.query(Recomendacion).filter(Recomendacion.prediccion_id.in_([p.id for p in predicciones]))
    }

    filas = []
    for p in predicciones:
        recomendacion = recomendaciones.get(p.id)
        idx = fecha_a_idx.get(p.fecha)
        resultado, cambio_real = "pendiente", None
        if idx is not None:
            idx_horizonte = idx + TARGET_HORIZON_DAYS
            if idx_horizonte < len(ohlcv_ordenado):
                close_inicio = ohlcv_ordenado.loc[idx, "close"]
                close_final = ohlcv_ordenado.loc[idx_horizonte, "close"]
                cambio_real = (close_final - close_inicio) / close_inicio if close_inicio else 0
                subio_de_verdad = cambio_real > TARGET_PCT_THRESHOLD
                predijo_sube = p.prediccion == "sube"
                resultado = "acerto" if subio_de_verdad == predijo_sube else "fallo"
        filas.append(
            {
                "fecha": p.fecha,
                "probabilidad": p.score,
                "prediccion": p.prediccion,
                "senal": recomendacion.tipo_senal if recomendacion else "-",
                "cambio_real": cambio_real,
                "resultado": resultado,
            }
        )
    df = pd.DataFrame(filas)
    df["probabilidad"] = df["probabilidad"].map(lambda v: f"{v:.1%}")
    df["cambio_real"] = df["cambio_real"].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
    return df


@con_cache
def cargar_resumen_mercado(session, modelo) -> pd.DataFrame:
    """Snapshot por ticker (sentimiento, probabilidad XGBoost, ratios y precio) en pocas consultas, no una por ticker."""
    tickers = session.query(Ticker.id, Ticker.symbol).order_by(Ticker.symbol).all()

    def ultimo_por_ticker(modelo_db):
        ultima = (
            session.query(modelo_db.ticker_id, func.max(modelo_db.fecha).label("fecha"))
            .group_by(modelo_db.ticker_id)
            .subquery()
        )
        filas = (
            session.query(modelo_db)
            .join(ultima, and_(modelo_db.ticker_id == ultima.c.ticker_id, modelo_db.fecha == ultima.c.fecha))
            .all()
        )
        return {f.ticker_id: f for f in filas}

    features = ultimo_por_ticker(Feature)
    ratios = ultimo_por_ticker(RatioFinanciero)

    precios = {}
    fecha_max_precio = session.query(func.max(PrecioOHLCV.fecha)).scalar()
    if fecha_max_precio is not None:
        recientes = (
            session.query(PrecioOHLCV.ticker_id, PrecioOHLCV.close)
            .filter(PrecioOHLCV.fecha >= fecha_max_precio - dt.timedelta(days=10))
            .order_by(PrecioOHLCV.fecha)
        )
        for ticker_id, close in recientes:
            if close is not None and pd.notna(close):
                precios[ticker_id] = float(close)  # orden ascendente: queda el ultimo cierre valido

    # El sentimiento del ultimo dia casi siempre es 0 (dia sin noticias); se promedia la ventana reciente.
    sentimientos = {}
    fecha_max_texto = session.query(func.max(Texto.fecha)).scalar()
    if fecha_max_texto is not None:
        # Ventana de dias completos, anclada al ultimo texto de toda la base (un ticker sin noticias recientes queda en 0).
        corte = dt.datetime.combine(fecha_max_texto.date() - dt.timedelta(days=VENTANA_SENTIMIENTO_DIAS), dt.time.min)
        filas_s = (
            session.query(Texto.ticker_id, Texto.fecha, Sentimiento.score, Sentimiento.etiqueta)
            .join(Sentimiento, Sentimiento.texto_id == Texto.id)
            .filter(Texto.fecha >= corte)
            .all()
        )
        if filas_s:
            sent = pd.DataFrame(filas_s, columns=["ticker_id", "fecha", "score", "etiqueta"])
            sent["fecha"] = pd.to_datetime(sent["fecha"]).dt.normalize()
            sent["firmado"] = sent["score"] * sent["etiqueta"].map({"positivo": 1, "negativo": -1}).fillna(0)
            diario = sent.groupby(["ticker_id", "fecha"])["firmado"].mean()
            sentimientos = diario.groupby("ticker_id").mean().to_dict()

    con_features = [(tid, symbol) for tid, symbol in tickers if tid in features]
    if not con_features:
        return pd.DataFrame()

    filas_features = [
        {"fecha": features[tid].fecha, **{c: getattr(features[tid], c) for c in FEATURE_COLS}} for tid, _ in con_features
    ]
    probabilidades = modelo.predict_proba(pd.DataFrame(filas_features)[FEATURE_COLS])[:, 1]

    filas = []
    for (tid, symbol), feats, prob in zip(con_features, filas_features, probabilidades):
        ratio = ratios.get(tid)
        filas.append(
            {
                "symbol": symbol,
                "precio": precios.get(tid),
                "sentimiento": float(sentimientos.get(tid, 0.0)),
                "probabilidad": float(prob),
                "senal": generar_recomendacion(float(prob), feats)["tipo_senal"],
                "market_cap": ratio.market_cap if ratio else None,
                "pe_ratio": ratio.pe_ratio if ratio else None,
                "pb_ratio": ratio.pb_ratio if ratio else None,
                "eps": ratio.eps if ratio else None,
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    with st.sidebar:
        modo_oscuro = st.toggle("🌙 Modo oscuro", value=False, key="modo_oscuro")
    tema = TEMAS["dark"] if modo_oscuro else TEMAS["light"]
    inyectar_css(tema)

    st.title("Penny Stocks Analytics")

    session = get_db_session()
    tickers = cargar_tickers(session)

    if not tickers:
        st.warning(
            "No hay tickers en la base de datos todavia. Corre `seed_synthetic_data.py` "
            "(demo) o `ingest_market_data.py` (real) primero."
        )
        return

    ticker_symbol = st.sidebar.selectbox("Ticker", tickers)
    ticker = cargar_ticker_info(session, ticker_symbol)
    st.sidebar.caption(f"{ticker.nombre or '(sin nombre)'} · {ticker.sector or '(sin sector)'}")

    rango_label = st.sidebar.radio("Periodo (reporte descriptivo)", list(RANGOS_FECHA.keys()), index=2)

    tab_desc, tab_pred, tab_presc = st.tabs(["Descriptivo", "Predictivo", "Prescriptivo"])

    # --- Reporte descriptivo ---
    with tab_desc:
        st.subheader("Panorama del mercado")
        modelo_radar, _ = cargar_ultimo_modelo()
        if modelo_radar is None:
            st.info("Sin modelo entrenado, no se puede calcular el panorama de mercado.")
        else:
            resumen_mercado_df = cargar_resumen_mercado(session, modelo_radar)
            if resumen_mercado_df.empty:
                st.info("Sin features calculadas todavia para ningun ticker.")
            else:
                st.plotly_chart(
                    grafico_dispersion_mercado(resumen_mercado_df, tema, ticker_symbol),
                    use_container_width=True,
                )
                st.caption(
                    "Cuadrante superior derecho (verde) = sentimiento positivo y probabilidad de suba "
                    "por encima del umbral. Tamano de burbuja = market cap."
                )

                with st.expander("Indicadores que influyen en el P/E", expanded=True):
                    tabla_indicadores_pe(resumen_mercado_df, tema, ticker_symbol)

        st.divider()
        st.subheader(f"Detalle de {ticker_symbol}")

        ohlcv_df_full = cargar_ohlcv_df(session, ticker.id)
        ohlcv_df_full = ohlcv_df_full.dropna(subset=["close"])
        if ohlcv_df_full.empty:
            st.info("Sin datos de precio para este ticker.")
        else:
            ohlcv_df = filtrar_por_rango(ohlcv_df_full, RANGOS_FECHA[rango_label])

            ultimo = ohlcv_df_full.iloc[-1]
            previo = ohlcv_df_full.iloc[-2] if len(ohlcv_df_full) > 1 else ultimo
            cambio_pct = (ultimo["close"] - previo["close"]) / previo["close"] if previo["close"] else 0
            ratios_df_kpi = cargar_ratios_df(session, ticker.id)
            ultimo_ratio_kpi = ratios_df_kpi.iloc[0] if not ratios_df_kpi.empty else None
            market_cap_actual = ultimo_ratio_kpi["market_cap"] if ultimo_ratio_kpi is not None else None
            pe_actual = ultimo_ratio_kpi["pe_ratio"] if ultimo_ratio_kpi is not None else None

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Precio de cierre", f"${ultimo['close']:.2f}", f"{cambio_pct:+.1%}")
            k2.metric("Volumen (ult. dia)", f"{ultimo['volumen']:,.0f}")
            k3.metric(
                "Market cap",
                f"${market_cap_actual:,.1f}M" if market_cap_actual else "N/D",
            )
            k4.metric("P/E", f"{pe_actual:.2f}" if pd.notna(pe_actual) else "N/D")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Precio de cierre**")
                st.plotly_chart(grafico_precio(ohlcv_df, tema), use_container_width=True)
            with col2:
                st.markdown("**Volumen** (verde = dia al alza, rojo = a la baja)")
                st.plotly_chart(grafico_volumen(ohlcv_df, tema), use_container_width=True)

        sentimiento_df_full = cargar_sentimiento_diario_df(session, ticker.id)
        st.markdown("**Evolucion del sentimiento (noticias)**")
        if sentimiento_df_full.empty:
            st.info("Sin sentimiento calculado todavia (corre sentiment_finbert.py).")
        else:
            sentimiento_df = filtrar_por_rango(sentimiento_df_full, RANGOS_FECHA[rango_label])
            st.plotly_chart(grafico_sentimiento(sentimiento_df, tema), use_container_width=True)

        with st.expander("Que dicen las noticias", expanded=True):
            textos_df = cargar_textos_df(session, ticker.id)
            if textos_df.empty:
                st.info("Sin textos analizados todavia para este ticker.")
            else:
                col_fuente, col_sent = st.columns(2)
                fuente_sel = col_fuente.radio("Fuente", ["Todas", "Noticias", "StockTwits"], horizontal=True)
                sent_sel = col_sent.radio("Sentimiento", ["Todos", "Positivo", "Negativo", "Neutral"], horizontal=True)
                vista = filtrar_por_rango(textos_df, RANGOS_FECHA[rango_label])
                if fuente_sel != "Todas":
                    vista = vista[vista["fuente"] == {"Noticias": "noticia", "StockTwits": "stocktwits"}[fuente_sel]]
                if sent_sel != "Todos":
                    vista = vista[vista["etiqueta"] == sent_sel.lower()]
                if vista.empty:
                    st.info("Ningun texto coincide con los filtros.")
                else:
                    tabla_textos(vista.head(60), tema, ticker_symbol)
                    if len(vista) > 60:
                        st.caption(f"Mostrando los 60 mas recientes de {len(vista)}.")

        ratios_df = cargar_ratios_df(session, ticker.id)
        if not ratios_df.empty:
            with st.expander("Ver historial completo de ratios financieros"):
                ratios_fmt = ratios_df.copy()
                ratios_fmt["fecha"] = ratios_fmt["fecha"].astype(str)
                ratios_fmt["market_cap"] = ratios_fmt["market_cap"].map(
                    lambda v: f"${v:,.1f}M" if pd.notna(v) else "N/D"
                )
                ratios_fmt["pe_ratio"] = ratios_fmt["pe_ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/D")
                ratios_fmt["pb_ratio"] = ratios_fmt["pb_ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/D")
                ratios_fmt["eps"] = ratios_fmt["eps"].map(lambda v: f"${v:.4f}" if pd.notna(v) else "N/D")
                ratios_fmt["shares_outstanding"] = ratios_fmt["shares_outstanding"].map(
                    lambda v: f"{v:,.0f}" if pd.notna(v) else "N/D"
                )
                renderizar_tabla_html(
                    ratios_fmt,
                    [
                        ("fecha", "Fecha"),
                        ("market_cap", "Market cap"),
                        ("pe_ratio", "P/E"),
                        ("pb_ratio", "P/B"),
                        ("eps", "EPS"),
                        ("shares_outstanding", "Acciones en circulacion"),
                    ],
                    tema,
                    numericas=("market_cap", "pe_ratio", "pb_ratio", "eps", "shares_outstanding"),
                )

    # --- Reporte predictivo ---
    prob = None
    features = None
    modelo_version = None
    with tab_pred:
        st.subheader(f"Prediccion XGBoost para {ticker_symbol}")

        modelo, modelo_version = cargar_ultimo_modelo()
        features = ultimo_feature_row(session, ticker.id)

        if modelo is None:
            st.warning("No hay ningun modelo entrenado todavia. Corre `train_xgboost.py` primero.")
        elif features is None:
            st.warning("No hay features calculadas para este ticker. Corre `feature_engineering.py` primero.")
        else:
            X = pd.DataFrame([{c: features[c] for c in FEATURE_COLS}])
            prob = float(modelo.predict_proba(X)[0][1])

            if not ohlcv_df_full.empty:
                st.markdown("**Zona objetivo**")
                st.plotly_chart(grafico_zona_objetivo(ohlcv_df_full, prob, tema), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(grafico_gauge(prob, tema), use_container_width=True)
                st.caption(f"Modelo: {modelo_version} · features del {features['fecha']}")
            with col2:
                st.markdown("**Feature importance (global del modelo)**")
                st.plotly_chart(grafico_feature_importance(modelo, tema), use_container_width=True)

            st.markdown("**Features usadas para esta prediccion**")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("RSI", f"{features['rsi']:.1f}")
            fc2.metric("Media movil", f"${features['media_movil']:.2f}")
            fc3.metric("Volumen relativo", f"{features['volumen_relativo']:.2f}x")
            fc4.metric("Sentimiento agregado", f"{features['sentimiento_agregado']:+.2f}")

    # --- Reporte prescriptivo ---
    with tab_presc:
        st.subheader(f"Recomendacion para {ticker_symbol}")

        if prob is None or features is None:
            st.info("Necesita una prediccion valida (pestana Predictivo) para generar una recomendacion.")
        else:
            recomendacion = generar_recomendacion(prob, features)
            guardar_prediccion_y_recomendacion(session, ticker.id, modelo_version, prob, features, recomendacion)

            color = tema[COLOR_SENAL_KEY[recomendacion["tipo_senal"]]]
            st.markdown(
                f"""
                <div style="
                    background-color:{color}1a;
                    border-left: 6px solid {color};
                    border-radius: 8px;
                    padding: 1.25rem 1.5rem;
                ">
                    <div style="font-size:1.4rem; font-weight:700; color:{color}; font-family:{FONT_DISPLAY};">
                        {recomendacion['tipo_senal'].upper()}
                    </div>
                    <div style="margin-top:0.5rem; font-size:1rem; color:{tema['text']};">
                        {recomendacion['justificacion']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if recomendacion["tipo_senal"] == "sin senal":
                faltante = UMBRAL_SENAL_MONITOREAR - prob
                st.caption(f"A {faltante:.1%} de pasar a **monitorear** (umbral: {UMBRAL_SENAL_MONITOREAR:.0%}).")
            elif recomendacion["tipo_senal"] == "monitorear":
                faltante = UMBRAL_SENAL_ALERTA - prob
                st.caption(f"A {faltante:.1%} de pasar a **señal de alerta** (umbral: {UMBRAL_SENAL_ALERTA:.0%}).")
            else:
                st.caption("Ya esta en el nivel de señal mas alto.")

        st.divider()
        st.markdown("**Historial de predicciones para este ticker**")
        historial_df = cargar_historial_predicciones(session, ticker.id, ohlcv_df_full)
        if historial_df.empty:
            st.info("Sin historial todavia — cada vez que abres la pestaña Predictivo se guarda una fila nueva.")
        else:
            resueltas = historial_df[historial_df["resultado"] != "pendiente"]
            if not resueltas.empty:
                tasa_acierto = (resueltas["resultado"] == "acerto").mean()
                st.caption(
                    f"Acierto historico: {tasa_acierto:.0%} sobre {len(resueltas)} predicciones ya resueltas "
                    f"(de {len(historial_df)} totales; el resto sigue dentro del horizonte de {TARGET_HORIZON_DAYS} dias)."
                )
            historial_fmt = historial_df.copy()
            historial_fmt["fecha"] = historial_fmt["fecha"].astype(str)
            renderizar_tabla_html(
                historial_fmt,
                [
                    ("fecha", "Fecha"),
                    ("probabilidad", "Probabilidad"),
                    ("prediccion", "Prediccion"),
                    ("senal", "Señal"),
                    ("cambio_real", "Cambio real"),
                    ("resultado", "Resultado"),
                ],
                tema,
                numericas=("probabilidad", "cambio_real"),
            )

        st.divider()
        st.markdown("**Señales de todos los tickers hoy**")
        if modelo is None:
            st.info("Sin modelo entrenado, no se puede comparar.")
        else:
            comparativa_senales_df = cargar_resumen_mercado(session, modelo).sort_values("probabilidad", ascending=False)
            if comparativa_senales_df.empty:
                st.info("Sin features calculadas para comparar tickers.")
            else:
                badges = ""
                for _, fila in comparativa_senales_df.iterrows():
                    color = tema[COLOR_SENAL_KEY[fila["senal"]]]
                    borde = f"2px solid {tema['text']}" if fila["symbol"] == ticker_symbol else "none"
                    badges += (
                        f'<div style="display:inline-block; margin:4px; padding:8px 14px; '
                        f'border-radius:8px; background-color:{color}1a; border:{borde}; '
                        f'color:{tema["text"]};">'
                        f'<b>{fila["symbol"]}</b> · {fila["probabilidad"]:.0%} · '
                        f'<span style="color:{color};">{fila["senal"]}</span></div>'
                    )
                st.markdown(badges, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
