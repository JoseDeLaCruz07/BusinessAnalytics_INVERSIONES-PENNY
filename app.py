"""Dashboard Streamlit: reportes descriptivo, predictivo y prescriptivo. Uso: streamlit run app.py"""
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sqlalchemy import and_, func

from src.config import (
    TARGET_HORIZON_DAYS,
    TARGET_PCT_THRESHOLD,
    PERFIL_RIESGO_DEFECTO,
    PERFILES_RIESGO,
    UMBRAL_SENAL_ALERTA,
    UMBRAL_SENAL_MONITOREAR,
)
from src.db import (
    Feature,
    FilingSEC,
    PrecioOHLCV,
    Prediccion,
    RatioFinanciero,
    Recomendacion,
    RecomendacionIA,
    Sentimiento,
    Texto,
    Ticker,
    get_session,
    init_db,
)
from src.models.train_xgboost import (
    AUC_AVISO,
    FEATURE_COLS,
    MONITOREO_PATH,
    PUBLISHED_MODEL_PATH,
    construir_dataset,
    evaluar_walk_forward,
)
from src.reports.analisis import (
    VENTANA_VOLATILIDAD_DIAS,
    calcular_cono_volatilidad,
    contribuciones_prediccion,
)
from src.reports.business_rules import generar_recomendacion
from src.reports.predicciones_diarias import guardar_prediccion_y_recomendacion

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
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {tema['text_muted']} !important; }}
        [data-testid="stMetric"] [data-testid="stTooltipIcon"] svg {{ color: {tema['text_muted']} !important; }}

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


        .element-container:has(.vista-nav) + .element-container [role="radiogroup"] {{
            gap: 0.2rem; border-bottom: 1px solid {tema['border']}; margin-bottom: 0.6rem;
        }}
        .element-container:has(.vista-nav) + .element-container label {{
            padding: 0.45rem 1.1rem; margin: 0; cursor: pointer; border-bottom: 2px solid transparent;
        }}
        .element-container:has(.vista-nav) + .element-container label > div:first-child {{ display: none; }}
        .element-container:has(.vista-nav) + .element-container label p {{
            color: {tema['text_muted']} !important; font-size: 0.95rem; font-weight: 500;
        }}
        .element-container:has(.vista-nav) + .element-container label:has(input:checked) {{ border-bottom-color: {tema['accent']}; }}
        .element-container:has(.vista-nav) + .element-container label:has(input:checked) p {{
            color: {tema['accent']} !important; font-weight: 600;
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
def preparar_db() -> None:
    init_db()


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
    """Sentimiento (X) vs puntaje XGBoost (Y) de todos los tickers.

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
                    "<b>%{customdata[0]}</b><br>Sentimiento: %{x:.2f}<br>Puntaje: %{y:.0%}"
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
        yaxis=dict(title="Puntaje del modelo (XGBoost)", range=[0, 1], tickformat=".0%"),
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
    fuentes = {
        "noticia": ("Noticia", tema["accent"]),
        "stocktwits": ("StockTwits", tema["accent_2"]),
        "sec": ("SEC", tema["text"]),
    }
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
        return chip_html(nombre if pd.isna(fila["score"]) else f"{nombre} {fila['score']:.0%}", color)

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


@con_cache
def cargar_filings_df(session, ticker_id: int) -> pd.DataFrame:
    rows = (
        session.query(FilingSEC.fecha, FilingSEC.formulario, FilingSEC.categoria, FilingSEC.signo, FilingSEC.descripcion, FilingSEC.url)
        .filter(FilingSEC.ticker_id == ticker_id)
        .order_by(FilingSEC.fecha.desc())
        .limit(200)
        .all()
    )
    df = pd.DataFrame(rows, columns=["fecha", "formulario", "categoria", "signo", "descripcion", "url"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def unir_filings(textos_df: pd.DataFrame, filings_df: pd.DataFrame) -> pd.DataFrame:
    """Los filings de la SEC se muestran junto a los textos; su 'sentimiento' es el signo de la regla, sin score."""
    if filings_df.empty:
        return textos_df
    sec = pd.DataFrame(
        {
            "fecha": filings_df["fecha"],
            "fuente": "sec",
            "contenido": filings_df["formulario"] + ": " + filings_df["descripcion"],
            "url": filings_df["url"],
            "etiqueta": filings_df["signo"],
            "score": float("nan"),
        }
    )
    return pd.concat([textos_df, sec], ignore_index=True).sort_values("fecha", ascending=False)


@con_cache
def cargar_recomendacion_ia(session, ticker_id: int) -> dict | None:
    fila = session.query(RecomendacionIA).filter_by(ticker_id=ticker_id).order_by(RecomendacionIA.fecha.desc()).first()
    if fila is None:
        return None
    return {
        "fecha": fila.fecha,
        "modelo_llm": fila.modelo_llm,
        "resumen": fila.resumen,
        "a_favor": json.loads(fila.a_favor),
        "riesgos": json.loads(fila.riesgos),
        "vigilar": json.loads(fila.vigilar),
        "confianza": fila.confianza,
        # Campos nuevos: las recomendaciones anteriores no los tienen (NULL) y la tarjeta simplemente no se muestra.
        "accion_sugerida": fila.accion_sugerida,
        "justificacion_accion": fila.justificacion_accion,
        "alternativas": json.loads(fila.alternativas) if fila.alternativas else [],
        "noticias_clave": json.loads(fila.noticias_clave) if fila.noticias_clave else [],
    }


def tarjeta_html(contenido: str, tema: dict, borde_color: str | None = None, lado: str = "left", alto_min: str = "0") -> str:
    acento = f"border-{lado}:4px solid {borde_color};" if borde_color else ""
    return (
        f'<div style="border:1px solid {tema["border"]}; {acento} border-radius:12px; background-color:{tema["surface"]}; '
        f'padding:1.1rem 1.4rem; box-shadow:0 1px 3px rgba(15,23,42,0.08); height:100%; min-height:{alto_min}; '
        f'display:flex; flex-direction:column; justify-content:center;">{contenido}</div>'
    )


def etiqueta_html(texto: str, tema: dict, color: str | None = None) -> str:
    return (
        f'<div style="font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; '
        f'color:{color or tema["text_muted"]}; margin-bottom:0.35rem;">{texto}</div>'
    )


def bloque_senal(recomendacion: dict, prob: float, tema: dict) -> str:
    color = tema[COLOR_SENAL_KEY[recomendacion["tipo_senal"]]]
    contexto = recomendacion["justificacion"].replace(f"Puntaje del modelo XGBoost: {prob:.1%}. ", "")
    return tarjeta_html(
        etiqueta_html("Senal del modelo", tema)
        + f'<div style="font-family:{FONT_DISPLAY}; font-size:2rem; font-weight:800; line-height:1.1; color:{color};">'
        f'{texto_seguro(recomendacion["tipo_senal"].upper())}</div>'
        f'<div style="margin-top:0.5rem; font-family:{FONT_DISPLAY}; font-size:1.5rem; font-weight:700; color:{tema["text"]};">{prob:.1%}'
        f'<span style="font-family:{FONT_BODY}; font-size:0.85rem; color:{tema["text_muted"]};"> puntaje del modelo</span></div>'
        f'<div style="margin-top:0.6rem; font-size:0.9rem; color:{tema["text_muted"]};">{texto_seguro(contexto)}</div>',
        tema,
        borde_color=color,
        alto_min="13.4rem",
    )


def bloque_asesor_ia(rec: dict, tema: dict) -> None:
    """El texto lo escribe un LLM: se escapa igual que cualquier texto de terceros."""
    colores_confianza = {"baja": tema["danger"], "media": tema["accent_2"], "alta": tema["success"]}
    chip = chip_html(f"Confianza {texto_seguro(rec['confianza'])}", colores_confianza.get(rec["confianza"], tema["neutral"]))
    cabecera = (
        f'<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.6rem;">'
        f'<div style="display:flex; align-items:center; gap:0.7rem;">'
        f'<span style="font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; color:{tema["text_muted"]};">'
        f"Resumen del asesor</span>{chip}</div>"
        f'<span style="font-size:0.78rem; color:{tema["text_muted"]};">{texto_seguro(rec["modelo_llm"])} · {rec["fecha"]}</span></div>'
    )
    resumen = f'<div style="font-size:1.05rem; line-height:1.55; color:{tema["text"]};">{texto_seguro(rec["resumen"])}</div>'

    def lista(titulo: str, clave: str, color: str) -> str:
        items = "".join(
            f'<div style="display:flex; gap:0.55rem; margin-bottom:0.45rem; font-size:0.92rem; line-height:1.4; color:{tema["text"]};">'
            f'<span style="color:{color}; font-size:0.6rem; padding-top:0.35rem;">●</span><span>{texto_seguro(t)}</span></div>'
            for t in rec[clave]
        )
        return tarjeta_html(etiqueta_html(titulo, tema, color) + items, tema, borde_color=color, lado="top")

    columnas = (
        lista("A favor", "a_favor", tema["success"]),
        lista("Riesgos", "riesgos", tema["danger"]),
        lista("Que vigilar", "vigilar", tema["accent_2"]),
    )
    st.markdown(
        f'<div style="display:grid; gap:1rem;">{tarjeta_html(cabecera + resumen, tema)}'
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:1rem;">{"".join(columnas)}</div></div>',
        unsafe_allow_html=True,
    )
    if rec.get("accion_sugerida"):
        colores_accion = {"vigilar de cerca": tema["accent_2"], "esperar confirmacion": tema["neutral"], "evitar": tema["danger"]}
        color_accion = colores_accion.get(rec["accion_sugerida"], tema["neutral"])
        accion = (
            etiqueta_html("Accion sugerida por la IA (acotada por la regla fija)", tema)
            + f'<div style="font-family:{FONT_DISPLAY}; font-size:1.4rem; font-weight:800; color:{color_accion};">'
            f'{texto_seguro(rec["accion_sugerida"].upper())}</div>'
            f'<div style="margin-top:0.4rem; font-size:0.92rem; line-height:1.45; color:{tema["text"]};">'
            f'{texto_seguro(rec.get("justificacion_accion") or "")}</div>'
        )
        alternativas = "".join(
            f'<div style="display:flex; gap:0.55rem; margin-bottom:0.45rem; font-size:0.92rem; line-height:1.4; color:{tema["text"]};">'
            f'<span style="color:{tema["accent_2"]}; font-size:0.6rem; padding-top:0.35rem;">●</span><span>{texto_seguro(t)}</span></div>'
            for t in rec.get("alternativas", [])
        ) or f'<span style="color:{tema["text_muted"]};">Sin alternativas.</span>'

        def cita(n: dict) -> str:
            enlace = (
                f' · <a href="{texto_seguro(n["url"])}" target="_blank" rel="noopener noreferrer">ver fuente</a>'
                if str(n.get("url") or "").startswith("http") else ""
            )
            return (
                f'<div style="margin-bottom:0.55rem; font-size:0.88rem; line-height:1.4; color:{tema["text"]};">'
                f'<b>{texto_seguro(n["id"])}</b> <span style="color:{tema["text_muted"]};">{texto_seguro(str(n.get("fecha") or ""))}</span>'
                f'{enlace}<br>{texto_seguro(n.get("texto") or "", 220)}</div>'
            )

        citas = "".join(cita(n) for n in rec.get("noticias_clave", [])) or (
            f'<span style="color:{tema["text_muted"]};">La IA no cito textos verificables.</span>'
        )
        st.markdown(
            f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:1rem; margin-top:1rem;">'
            f'{tarjeta_html(accion, tema, borde_color=color_accion, lado="top")}'
            f'{tarjeta_html(etiqueta_html("Alternativas mas prudentes", tema, tema["accent_2"]) + alternativas, tema, borde_color=tema["accent_2"], lado="top")}'
            f'{tarjeta_html(etiqueta_html("Noticias clave citadas (verificadas)", tema) + citas, tema, borde_color=tema["neutral"], lado="top")}'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="margin-top:0.6rem; font-size:0.8rem; color:{tema["text_muted"]};">'
        "Generado con IA a partir de los datos de este dashboard. Analisis educativo, no es asesoria financiera.</div>",
        unsafe_allow_html=True,
    )


def historial_con_etiquetas(df: pd.DataFrame, tema: dict) -> pd.DataFrame:
    colores_resultado = {"acerto": tema["success"], "fallo": tema["danger"], "pendiente": tema["neutral"]}
    out = df.copy()
    out["fecha"] = out["fecha"].astype(str)
    out["resultado"] = out["resultado"].map(lambda r: chip_html(texto_seguro(r), colores_resultado.get(r, tema["neutral"])))
    out["senal"] = out["senal"].map(lambda r: chip_html(texto_seguro(r), tema[COLOR_SENAL_KEY.get(r, "neutral")]))
    return out


def grafico_ranking_senales(df: pd.DataFrame, tema: dict, ticker_actual: str, tasa_base: float | None) -> go.Figure:
    df = df.sort_values("probabilidad", ascending=False)
    fig = go.Figure()
    for senal, nombre in (("senal de alerta", "Senal de alerta"), ("monitorear", "Monitorear"), ("sin senal", "Sin senal")):
        sub = df[df["senal"] == senal]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["probabilidad"], y=sub["symbol"], orientation="h", name=nombre,
                marker=dict(
                    color=tema[COLOR_SENAL_KEY[senal]],
                    line=dict(color=tema["text"], width=[2 if t == ticker_actual else 0 for t in sub["symbol"]]),
                ),
                text=[f"{v:.0%}" for v in sub["probabilidad"]], textposition="outside", textfont=dict(color=tema["text"]),
                hovertemplate="<b>%{y}</b><br>Puntaje: %{x:.1%}<extra></extra>",
            )
        )
    referencias = [(UMBRAL_SENAL_MONITOREAR, "Monitorear"), (UMBRAL_SENAL_ALERTA, "Alerta")]
    if tasa_base:
        referencias.append((tasa_base, "Tasa base"))
    for x, texto in referencias:
        fig.add_vline(x=x, line=dict(color=tema["text_muted"], width=1, dash="dot"))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{texto} {x:.0%}", showarrow=False, yanchor="bottom", font=dict(size=10, color=tema["text_muted"]))
    fig.update_layout(
        height=30 * len(df) + 90,
        margin=dict(l=0, r=30, t=30, b=0),
        barmode="overlay",
        bargap=0.35,
        xaxis=dict(range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(categoryorder="array", categoryarray=list(df["symbol"]), autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0, font=dict(color=tema["text"])),
    )
    return aplicar_tema_plotly(fig, tema)


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


def pe_valido(pe, eps):
    """Un P/E solo tiene sentido con utilidad positiva; si el EPS es <= 0 se muestra N/D."""
    if pe is None or eps is None or pd.isna(pe) or pd.isna(eps) or eps <= 0 or pe <= 0:
        return None
    return pe


@con_cache
def cargar_ratios_df(session, ticker_id: int) -> pd.DataFrame:
    rows = session.query(RatioFinanciero).filter_by(ticker_id=ticker_id).order_by(RatioFinanciero.fecha.desc()).all()
    return pd.DataFrame(
        [
            {
                "fecha": r.fecha,
                "market_cap": r.market_cap,
                "pe_ratio": pe_valido(r.pe_ratio, r.eps),
                "pb_ratio": r.pb_ratio,
                "eps": r.eps,
                "shares_outstanding": r.shares_outstanding,
            }
            for r in rows
        ]
    )


# Todas las columnas que puede necesitar un modelo, el vigente o uno publicado antes del cambio de features.
COLUMNAS_FEATURES = list(dict.fromkeys(["rsi", "media_movil", "volumen_relativo", "sentimiento_agregado", *FEATURE_COLS]))


def columnas_del_modelo(modelo) -> list[str]:
    """El modelo publicado manda: asi la app no se rompe si aun no se reentreno con las features nuevas."""
    return list(modelo.get_booster().feature_names or FEATURE_COLS)


@con_cache
def ultimo_feature_row(session, ticker_id: int) -> dict | None:
    row = session.query(Feature).filter_by(ticker_id=ticker_id).order_by(Feature.fecha.desc()).first()
    if row is None:
        return None
    return {"fecha": row.fecha, **{c: getattr(row, c) for c in COLUMNAS_FEATURES}}


def version_publicada() -> str:
    version_path = PUBLISHED_MODEL_PATH.parent / "xgboost_latest.version.txt"
    return version_path.read_text().strip() if version_path.exists() else ""


def cargar_ultimo_modelo():
    return _cargar_modelo(version_publicada())


@st.cache_resource
def _cargar_modelo(_version: str):
    # `_version` solo cambia la clave de cache: con un modelo nuevo publicado en git se vuelve a cargar.
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
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=30, b=30))
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


ETIQUETAS_FEATURES = {
    "rsi": ("RSI", lambda v: f"{v:.1f}"),
    "media_movil": ("Media movil", lambda v: f"${v:.2f}"),
    "volumen_relativo": ("Volumen relativo", lambda v: f"{v:.2f}x"),
    "sentimiento_agregado": ("Sentimiento", lambda v: f"{v:+.2f}"),
    "precio_vs_sma": ("Precio vs media 10d", lambda v: f"{v:+.1%}"),
    "vol_10d": ("Volatilidad 10d", lambda v: f"{v:.1%}"),
    "vol_20d": ("Volatilidad 20d", lambda v: f"{v:.1%}"),
    "rango_5d": ("Rango diario 5d", lambda v: f"{v:.1%}"),
}
def grafico_contribuciones(df: pd.DataFrame, tema: dict) -> go.Figure:
    df = df.reindex(df["contribucion"].abs().sort_values().index)
    etiquetas = [f"{ETIQUETAS_FEATURES[f][0]} ({ETIQUETAS_FEATURES[f][1](v)})" for f, v in zip(df["feature"], df["valor"])]
    colores = [tema["success"] if c > 0 else tema["danger"] for c in df["contribucion"]]
    limite = max(df["contribucion"].abs().max() * 1.35, 0.1)
    fig = go.Figure(
        go.Bar(
            x=df["contribucion"],
            y=etiquetas,
            orientation="h",
            marker_color=colores,
            text=[f"{c:+.2f}" for c in df["contribucion"]],
            textposition="outside",
            textfont=dict(color=tema["text"]),
            hovertemplate="%{y}<br>Aporte: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=70 + 38 * len(df),
        margin=dict(l=0, r=30, t=10, b=0),
        xaxis=dict(range=[-limite, limite], title="Baja el puntaje  <  >  Sube el puntaje", zeroline=True),
    )
    return aplicar_tema_plotly(fig, tema)


def grafico_cono_mes(ohlcv_df: pd.DataFrame, cono: dict, tema: dict) -> go.Figure:
    recientes = ohlcv_df.sort_values("fecha").tail(VENTANA_VOLATILIDAD_DIAS)
    fechas = list(cono["fechas"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=cono["p90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=fechas, y=cono["p10"], mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=hex_to_rgba(tema["accent"], 0.2), name="Rango 80%",
            customdata=np.column_stack([cono["p10"], cono["p90"]]),
            hovertemplate="%{x|%d %b}<br>80%%: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=fechas, y=cono["p75"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=fechas, y=cono["p25"], mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor=hex_to_rgba(tema["accent"], 0.38), name="Rango 50%",
            customdata=np.column_stack([cono["p25"], cono["p75"]]),
            hovertemplate="%{x|%d %b}<br>50%%: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recientes["fecha"], y=recientes["close"], mode="lines", name="Precio de cierre",
            line=dict(color=tema["text"], width=2), hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Precio (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=tema["text"])),
    )
    return aplicar_tema_plotly(fig, tema)


@st.cache_data(ttl=3600, show_spinner=False, hash_funcs=HASH_SESION_MODELO)
def cargar_confiabilidad_modelo(session) -> dict:
    dataset = construir_dataset(session)
    if dataset.empty:
        return {}
    return evaluar_walk_forward(dataset, FEATURE_COLS, purga_dias=TARGET_HORIZON_DAYS) or {}


EVIDENCIAS_DIR = Path(__file__).resolve().parent / "docs" / "evidencias"


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_ultimo_backtest() -> tuple[dict | None, str | None]:
    """Ultimo resultado de `python -m src.models.backtesting` guardado en docs/evidencias (json y fecha del archivo)."""
    archivos = sorted(EVIDENCIAS_DIR.glob("backtest_*.json"))
    if not archivos:
        return None, None
    with open(archivos[-1], encoding="utf-8") as f:
        return json.load(f), archivos[-1].stem.replace("backtest_", "")


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_monitoreo_modelo() -> pd.DataFrame:
    if not MONITOREO_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(MONITOREO_PATH)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.dropna(subset=["auc"]).sort_values("fecha")


def grafico_monitoreo_auc(df: pd.DataFrame, tema: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["auc"], mode="lines+markers", name="AUC", line=dict(color=tema["accent"], width=2)))
    fig.add_hline(y=0.5, line_dash="dot", line_color=tema["text_muted"], annotation_text="azar", annotation_position="bottom right")
    fig.add_hline(y=AUC_AVISO, line_dash="dash", line_color=tema["danger"], annotation_text="umbral de aviso", annotation_position="top right")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, yaxis=dict(title="AUC fuera de muestra"))
    return aplicar_tema_plotly(fig, tema)


def seccion_backtest(tema: dict) -> None:
    resultado, fecha = cargar_ultimo_backtest()
    if resultado is None:
        st.info("Todavia no hay un backtest guardado. Corre `python -m src.models.backtesting`.")
        return
    inicio, fin = resultado.get("periodo_prueba", ["?", "?"])
    sup = resultado.get("supuestos", {})
    nombres = {
        "senal de alerta": "Comprar con senal de alerta",
        "top 3 puntaje": "Comprar los 3 de mayor puntaje",
        "comprar y mantener": "Comprar y mantener los 18",
    }
    filas = []
    for clave, etiqueta in nombres.items():
        e = resultado.get("estrategias", {}).get(clave)
        if not e:
            continue
        ganadoras = e.get("operaciones_ganadoras_pct")
        filas.append({
            "estrategia": etiqueta,
            "retorno": f"{e['retorno_total']:+.1%}",
            "caida": f"{e['caida_maxima']:.1%}",
            "operaciones": e.get("operaciones", "-"),
            "ganadoras": f"{ganadoras:.0%}" if ganadoras is not None else "-",
        })
    renderizar_tabla_html(
        pd.DataFrame(filas),
        [("estrategia", "Estrategia"), ("retorno", "Retorno total"), ("caida", "Caida maxima"),
         ("operaciones", "Operaciones"), ("ganadoras", "% ganadoras")],
        tema,
        numericas=("retorno", "caida", "operaciones", "ganadoras"),
    )
    clasif = resultado.get("clasificacion", {}).get("modelo", {})
    precision_alerta = clasif.get("precision_en_senal_alerta")
    st.caption(
        f"Backtest del {pd.to_datetime(fecha, format='%Y%m%d'):%d/%m/%Y}, periodo de prueba {inicio} a {fin}. Cada decision usa un modelo entrenado solo con fechas "
        f"anteriores; rebalanceo cada {sup.get('rebalanceo_cada_dias', TARGET_HORIZON_DAYS)} dias con comision de "
        f"{sup.get('comision_por_lado', 0):.1%} y deslizamiento de {sup.get('deslizamiento_por_lado', 0):.1%} por lado. "
        + (f"Acierto cuando hubo senal de alerta: {precision_alerta:.0%} (tasa base "
           f"{resultado.get('clasificacion', {}).get('tasa_base', 0):.0%}). " if precision_alerta is not None else "")
        + "Un resultado negativo es evidencia de que la senal sola no alcanza para operar: por eso el reporte "
        "prescriptivo limita la posicion y sugiere stop-loss."
    )
    grafico = EVIDENCIAS_DIR / f"backtest_{fecha}.png"
    if grafico.exists():
        with st.expander("Ver curva de capital del backtest"):
            st.image(str(grafico), use_column_width=True)


def catalizadores_recientes(filings_df: pd.DataFrame, tema: dict, dias: int = 30) -> None:
    colores = {"positivo": tema["success"], "negativo": tema["danger"], "neutral": tema["neutral"]}
    corte = pd.Timestamp.now() - pd.Timedelta(days=dias)
    recientes = filings_df[filings_df["fecha"] >= corte]
    if recientes.empty:
        st.markdown(f'<span style="color:{tema["text_muted"]};">Sin filings relevantes en los ultimos {dias} dias.</span>', unsafe_allow_html=True)
        return
    conteo = recientes.groupby(["categoria", "signo"]).size().reset_index(name="n").sort_values(["signo", "n"], ascending=[True, False])
    chips = " ".join(
        chip_html(f"{texto_seguro(f.categoria)} · {f.n}", colores.get(f.signo, tema["neutral"])) for f in conteo.itertuples()
    )
    st.markdown(chips, unsafe_allow_html=True)


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

    columnas = columnas_del_modelo(modelo)
    filas_features = [
        {"fecha": features[tid].fecha, **{c: getattr(features[tid], c) for c in COLUMNAS_FEATURES}} for tid, _ in con_features
    ]
    probabilidades = modelo.predict_proba(pd.DataFrame(filas_features)[columnas])[:, 1]

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
                "pe_ratio": pe_valido(ratio.pe_ratio, ratio.eps) if ratio else None,
                "pb_ratio": ratio.pb_ratio if ratio else None,
                "eps": ratio.eps if ratio else None,
            }
        )
    return pd.DataFrame(filas)


def main() -> None:
    # Sesion propia por ejecucion: una compartida queda trabada si un rerun interrumpe un commit.
    preparar_db()
    session = get_session()
    try:
        renderizar(session)
    finally:
        session.close()


def renderizar(session) -> None:
    with st.sidebar:
        modo_oscuro = st.toggle("🌙 Modo oscuro", value=False, key="modo_oscuro")
    tema = TEMAS["dark"] if modo_oscuro else TEMAS["light"]
    inyectar_css(tema)

    st.title("Penny Stocks Analytics")

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
    perfiles = list(PERFILES_RIESGO.keys())
    perfil = st.sidebar.selectbox(
        "Perfil de riesgo",
        perfiles,
        index=perfiles.index(PERFIL_RIESGO_DEFECTO),
        help="Cambia el umbral de la senal de alerta y los limites de posicion y perdida del reporte prescriptivo.",
    )

    st.markdown('<div class="vista-nav"></div>', unsafe_allow_html=True)
    vista_activa = st.radio("Vista", ["Descriptivo", "Predictivo", "Prescriptivo"], horizontal=True, key="vista", label_visibility="collapsed")

    # El filtro solo existe en el Descriptivo; se recuerda al salir porque Streamlit olvida los widgets que no se dibujan.
    opciones_rango = list(RANGOS_FECHA.keys())
    if vista_activa == "Descriptivo":
        rango_label = st.sidebar.radio(
            "Periodo", opciones_rango, index=opciones_rango.index(st.session_state.get("rango_label", "6 meses")), key="rango_widget"
        )
        st.session_state["rango_label"] = rango_label
    else:
        rango_label = st.session_state.get("rango_label", "6 meses")

    ohlcv_df_full = cargar_ohlcv_df(session, ticker.id).dropna(subset=["close"])

    # La prediccion y la senal se calculan (y se guardan una vez al dia) sea cual sea la vista abierta.
    modelo, modelo_version = cargar_ultimo_modelo()
    features = ultimo_feature_row(session, ticker.id)
    X = prob = recomendacion = None
    if modelo is not None and features is not None:
        X = pd.DataFrame([{c: features[c] for c in columnas_del_modelo(modelo)}])
        prob = float(modelo.predict_proba(X)[0][1])
        # Se guarda siempre la senal del perfil por defecto (el historial no depende de lo que elija cada visitante).
        guardar_prediccion_y_recomendacion(session, ticker.id, modelo_version, prob, features, generar_recomendacion(prob, features))
        recomendacion = generar_recomendacion(prob, features, perfil)

    # --- Reporte descriptivo ---
    if vista_activa == "Descriptivo":
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
                    "Cuadrante superior derecho (verde) = sentimiento positivo y puntaje del modelo "
                    "por encima del umbral. Tamano de burbuja = market cap."
                )

                with st.expander("Indicadores que influyen en el P/E", expanded=True):
                    tabla_indicadores_pe(resumen_mercado_df, tema, ticker_symbol)

        st.divider()
        st.subheader(f"Detalle de {ticker_symbol}")

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
            textos_df = unir_filings(cargar_textos_df(session, ticker.id), cargar_filings_df(session, ticker.id))
            if textos_df.empty:
                st.info("Sin textos analizados todavia para este ticker.")
            else:
                col_fuente, col_sent = st.columns(2)
                fuente_sel = col_fuente.radio("Fuente", ["Todas", "Noticias", "StockTwits", "SEC"], horizontal=True)
                sent_sel = col_sent.radio("Sentimiento", ["Todos", "Positivo", "Negativo", "Neutral"], horizontal=True)
                textos_vista = filtrar_por_rango(textos_df, RANGOS_FECHA[rango_label])
                if fuente_sel != "Todas":
                    fuente_clave = {"Noticias": "noticia", "StockTwits": "stocktwits", "SEC": "sec"}[fuente_sel]
                    textos_vista = textos_vista[textos_vista["fuente"] == fuente_clave]
                if sent_sel != "Todos":
                    textos_vista = textos_vista[textos_vista["etiqueta"] == sent_sel.lower()]
                if textos_vista.empty:
                    st.info("Ningun texto coincide con los filtros.")
                else:
                    tabla_textos(textos_vista.head(60), tema, ticker_symbol)
                    if len(textos_vista) > 60:
                        st.caption(f"Mostrando los 60 mas recientes de {len(textos_vista)}.")

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
    if vista_activa == "Predictivo":
        st.subheader(f"Prediccion XGBoost para {ticker_symbol}")

        if modelo is None:
            st.warning("No hay ningun modelo entrenado todavia. Corre `train_xgboost.py` primero.")
        elif features is None:
            st.warning("No hay features calculadas para este ticker. Corre `feature_engineering.py` primero.")
        else:
            conf = cargar_confiabilidad_modelo(session)
            col_gauge, col_contrib = st.columns([1, 1.3])
            with col_gauge:
                st.markdown("**Puntaje del modelo**")
                st.plotly_chart(grafico_gauge(prob, tema), use_container_width=True)
                tasa_base = f" · Tasa base: {conf['tasa_base']:.0%}" if conf else ""
                st.caption(f"Modelo {modelo_version} · features del {features['fecha']}{tasa_base}")
            with col_contrib:
                st.markdown("**Por que este puntaje**")
                st.plotly_chart(grafico_contribuciones(contribuciones_prediccion(modelo, X), tema), use_container_width=True)

            st.divider()
            st.markdown("**Catalizadores recientes (SEC, ultimos 30 dias)**")
            catalizadores_recientes(cargar_filings_df(session, ticker.id), tema)

            st.divider()
            st.markdown("**Horizonte de precio**")
            cono = calcular_cono_volatilidad(ohlcv_df_full) if not ohlcv_df_full.empty else None
            if cono is None:
                st.info("Historial de precios insuficiente para estimar el horizonte.")
            else:
                tab_3d, tab_mes = st.tabs([f"{TARGET_HORIZON_DAYS} dias", "1 mes"])
                with tab_3d:
                    st.plotly_chart(grafico_zona_objetivo(ohlcv_df_full, prob, tema), use_container_width=True)
                with tab_mes:
                    st.plotly_chart(grafico_cono_mes(ohlcv_df_full, cono, tema), use_container_width=True)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Rango 80% a 1 mes", f"${cono['p10'][-1]:.2f} - ${cono['p90'][-1]:.2f}")
                    k2.metric("Rango 50% a 1 mes", f"${cono['p25'][-1]:.2f} - ${cono['p75'][-1]:.2f}")
                    k3.metric("Volatilidad diaria", f"{cono['sigma']:.1%}")

            st.divider()
            st.markdown("**Confiabilidad del modelo**")
            if not conf:
                st.info("Datos insuficientes para evaluar el modelo.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC", f"{conf['auc']:.2f}", help="0.50 = azar, 1.00 = perfecto. Promedio de 4 tramos consecutivos.")
                m2.metric("Tramos sobre el azar", conf["folds_sobre_azar"], help="En cuantos de los 4 tramos de prueba el AUC supera 0.50.")
                m3.metric("Tasa base", f"{conf['tasa_base']:.0%}", help="Porcentaje de dias con suba real: lo que acertaria un modelo al azar.")
                m4.metric(
                    "Mejora en el top 10%",
                    f"{conf['lift_top10']:.1f}x",
                    help="Acierto del 10% de dias con mayor probabilidad, dividido entre la tasa base; 1.0x = no mejora al azar.",
                )
                st.caption(
                    f"Evaluado en 4 tramos consecutivos del final de la historia ({conf['n']:,} observaciones); "
                    "cada tramo se predice con un modelo entrenado solo con fechas anteriores."
                )
                if conf["auc"] < AUC_AVISO:
                    st.warning(
                        f"El AUC actual ({conf['auc']:.2f}) esta por debajo de {AUC_AVISO:.2f}: el modelo no ordena mucho "
                        "mejor que el azar (0.50). Toma el puntaje como una pista mas, no como senal de compra."
                    )

            monitoreo = cargar_monitoreo_modelo()
            if len(monitoreo) >= 2:
                st.markdown("**Seguimiento diario del modelo**")
                st.plotly_chart(grafico_monitoreo_auc(monitoreo, tema), use_container_width=True)
                ultimo = monitoreo.iloc[-1]
                st.caption(f"Ultimo reentrenamiento ({ultimo['fecha']:%d/%m/%Y}): {ultimo['motivo']}.")

            st.divider()
            st.markdown("**Backtesting: que hubiera pasado siguiendo las senales**")
            seccion_backtest(tema)

    # --- Reporte prescriptivo ---
    if vista_activa == "Prescriptivo":
        st.subheader(f"Recomendacion para {ticker_symbol}")

        if prob is None:
            st.info("Necesita una prediccion valida (modelo y features) para generar una recomendacion.")
        else:
            conf = cargar_confiabilidad_modelo(session)
            resumen_df = cargar_resumen_mercado(session, modelo)
            puesto = (
                int((resumen_df["probabilidad"] > prob).sum()) + 1 if not resumen_df.empty else None
            )
            cono = calcular_cono_volatilidad(ohlcv_df_full) if not ohlcv_df_full.empty else None

            col_senal, col_a, col_b = st.columns([1.35, 1, 1])
            col_senal.markdown(bloque_senal(recomendacion, prob, tema), unsafe_allow_html=True)
            col_a.metric("Tasa base", f"{conf['tasa_base']:.0%}" if conf else "N/D")
            col_a.metric("Puesto entre tickers", f"{puesto} de {len(resumen_df)}" if puesto else "N/D")
            col_b.metric("Rango 80% a 1 mes", f"${cono['p10'][-1]:.2f} - ${cono['p90'][-1]:.2f}" if cono else "N/D")
            col_b.metric("Volatilidad diaria", f"{cono['sigma']:.1%}" if cono else "N/D")

            st.markdown(f"**Limites de riesgo (perfil {recomendacion['perfil']}, regla fija)**")
            col_p, col_s, col_c = st.columns(3)
            col_p.metric(
                "Posicion maxima sugerida",
                f"{recomendacion['posicion_max']:.0%} del capital" if recomendacion["posicion_max"] else "No entrar",
            )
            ultimo_cierre = float(ohlcv_df_full["close"].iloc[-1]) if not ohlcv_df_full.empty else None
            col_s.metric(
                "Salida si cae (stop-loss)",
                f"-{recomendacion['perdida_max']:.0%}" + (f" (${ultimo_cierre * (1 - recomendacion['perdida_max']):.2f})" if ultimo_cierre else ""),
            )
            col_c.metric("Freno por volatilidad extrema", "Activado" if recomendacion["freno_volatilidad"] else "No")
            st.caption(
                "Los limites salen de reglas fijas por perfil (src/config.py, PERFILES_RIESGO), no del modelo ni del LLM. "
                "Son una guia educativa: no se ejecuta ninguna operacion."
            )

        st.divider()
        st.markdown("**Asesor IA**")
        rec_ia = cargar_recomendacion_ia(session, ticker.id)
        if rec_ia is None:
            st.info("Sin recomendacion IA todavia para este ticker: se genera una vez al dia.")
        else:
            bloque_asesor_ia(rec_ia, tema)

        st.divider()
        st.markdown("**Historial de predicciones para este ticker**")
        historial_df = cargar_historial_predicciones(session, ticker.id, ohlcv_df_full)
        if historial_df.empty:
            st.info("Sin historial todavia: el pipeline diario guarda una fila por dia (tambien al abrir el dashboard).")
        else:
            resueltas = historial_df[historial_df["resultado"] != "pendiente"]
            if not resueltas.empty:
                tasa_acierto = (resueltas["resultado"] == "acerto").mean()
                st.caption(
                    f"Acierto historico: {tasa_acierto:.0%} sobre {len(resueltas)} predicciones ya resueltas "
                    f"(de {len(historial_df)} totales; el resto sigue dentro del horizonte de {TARGET_HORIZON_DAYS} dias)."
                )
            renderizar_tabla_html(
                historial_con_etiquetas(historial_df, tema),
                [
                    ("fecha", "Fecha"),
                    ("probabilidad", "Puntaje"),
                    ("prediccion", "Prediccion"),
                    ("senal", "Señal"),
                    ("cambio_real", "Cambio real"),
                    ("resultado", "Resultado"),
                ],
                tema,
                numericas=("probabilidad", "cambio_real"),
            )

        st.divider()
        st.markdown("**Ranking de puntaje de todos los tickers**")
        if modelo is None:
            st.info("Sin modelo entrenado, no se puede comparar.")
        else:
            ranking_df = cargar_resumen_mercado(session, modelo)
            if ranking_df.empty:
                st.info("Sin features calculadas para comparar tickers.")
            else:
                tasa_base_ranking = cargar_confiabilidad_modelo(session).get("tasa_base")
                st.plotly_chart(grafico_ranking_senales(ranking_df, tema, ticker_symbol, tasa_base_ranking), use_container_width=True)

if __name__ == "__main__":
    main()
