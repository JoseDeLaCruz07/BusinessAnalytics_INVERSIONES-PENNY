"""Configuracion central del proyecto: tickers monitoreados y umbrales de negocio."""
import os
from dotenv import load_dotenv

load_dotenv()

# Definicion de negocio: "penny stock" = precio de cierre menor a este umbral (USD).
PENNY_STOCK_PRICE_THRESHOLD = 5.0

# MVP: tickers reales, liquidos, conocidos como penny stocks al momento de escribir esto.
# Se filtran igualmente en ingest_market_data.py contra PENNY_STOCK_PRICE_THRESHOLD,
# asi que si alguno sube de precio quedara marcado pero no descartado del historico.
# NOTA: ZOM, IDEX y GNUS estaban en la lista original pero fueron deslistados/dejaron
# de existir en Yahoo Finance (confirmado corriendo ingest_market_data.py en 2026-09).
# El mercado de penny stocks rota mucho (delistings, cambios de ticker); si alguno de
# estos tambien deja de funcionar, validar candidatos con
# `python -m src.ingestion.verificar_tickers SIMBOLO1 SIMBOLO2 ...` antes de reemplazarlo.
# PDSB, RETO y BNGO se sacaron del screener "Most active penny stocks" de Yahoo Finance
# y se validaron con verificar_tickers.py el 2026-09.DEFAULT_TICKERS = ["SNDL", "CTRM", "PDSB", "RETO", "BNGO"]
DEFAULT_TICKERS = ["SNDL", "CTRM", "PDSB", "RETO", "BNGO"]

# Credenciales y Base de Datos (Fase 1)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/penny_stocks.db")

# --- Fase 2: NLP y ML predictivo ---

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_MODEL_VERSION = "finbert-prosusai-v1"

# Variable objetivo de XGBoost (definicion de negocio, confirmada con el usuario):
# 1 si el precio de cierre sube mas de TARGET_PCT_THRESHOLD en los proximos
# TARGET_HORIZON_DAYS dias de trading, 0 en caso contrario.
TARGET_PCT_THRESHOLD = 0.05
TARGET_HORIZON_DAYS = 3

# Ventanas de indicadores tecnicos.
RSI_WINDOW = 14
SMA_WINDOW = 10
VOLUME_RELATIVE_WINDOW = 20

# --- Fase 4: reglas de negocio para el reporte prescriptivo ---
# Umbrales sobre la probabilidad predicha por XGBoost (definicion de negocio,
# ver src/reports/business_rules.py para la logica completa).
UMBRAL_SENAL_ALERTA = 0.65
UMBRAL_SENAL_MONITOREAR = 0.40
RSI_SOBREVENTA = 30
RSI_SOBRECOMPRA = 70
VOLUMEN_RELATIVO_SPIKE = 2.0