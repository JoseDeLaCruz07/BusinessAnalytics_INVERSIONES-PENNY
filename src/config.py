"""Configuracion central del proyecto: tickers monitoreados y umbrales de negocio."""
import os

from dotenv import load_dotenv

load_dotenv()

# Definicion de negocio: "penny stock" = precio de cierre menor a este umbral (USD).
PENNY_STOCK_PRICE_THRESHOLD = 5.0

# Si alguno se deslista, validar el reemplazo con `python -m src.ingestion.verificar_tickers`.
DEFAULT_TICKERS = [
    "SNDL", "CTRM", "PDSB", "RETO", "BNGO",
    "KOPN", "ABUS", "EOSE", "UPXI", "BATL", "SLND", "LCID",
    "VRRM", "EGHT", "PTLO", "WOOF", "JBI", "TBLA",
]

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
# La SEC exige identificarse con nombre y contacto, ej. "Penny Stock Analytics tu_correo@dominio.com".
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
# Asesor IA (Gemini). El alias "-latest" evita fijar una version que Google retire; `--listar-modelos` muestra los de tu clave.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"  # vacio (variable sin definir en CI) tambien usa el alias
# Tope de consultas a Gemini por dia (todas las corridas juntas, reintentos incluidos): 18 tickers x 2 intentos = 36.
MAX_LLAMADAS_LLM_DIA = int(os.getenv("MAX_LLAMADAS_LLM_DIA") or 40)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/penny_stocks.db")

# --- Fase 2: NLP y ML predictivo ---

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_MODEL_VERSION = "finbert-prosusai-v1"

# Target de XGBoost: 1 si el cierre sube > TARGET_PCT_THRESHOLD en TARGET_HORIZON_DAYS dias.
TARGET_PCT_THRESHOLD = 0.05
TARGET_HORIZON_DAYS = 3

# Ventanas de indicadores tecnicos.
RSI_WINDOW = 14
SMA_WINDOW = 10
VOLUME_RELATIVE_WINDOW = 20

# Umbrales de probabilidad para el reporte prescriptivo (ver src/reports/business_rules.py).
UMBRAL_SENAL_ALERTA = 0.65
UMBRAL_SENAL_MONITOREAR = 0.40
RSI_SOBREVENTA = 30
RSI_SOBRECOMPRA = 70
VOLUMEN_RELATIVO_SPIKE = 2.0
# Freno: con una volatilidad diaria (desv. estandar de 10 dias) mayor a esta, los indicadores suelen estar
# distorsionados por un split/reverse split o un evento extremo; la regla no emite "senal de alerta".
VOLATILIDAD_DIARIA_MAXIMA = 0.25

# Perfil de riesgo del usuario (reporte prescriptivo). Reglas fijas y educativas, no asesoria personalizada:
#   umbral_alerta: puntaje minimo para "senal de alerta" en ese perfil.
#   posicion_max: parte maxima del capital especulativo en un solo ticker.
#   perdida_max: salida sugerida (stop-loss) si el precio cae este porcentaje desde la entrada.
PERFILES_RIESGO = {
    "Conservador": {"umbral_alerta": 0.75, "posicion_max": 0.02, "perdida_max": 0.05},
    "Moderado": {"umbral_alerta": 0.65, "posicion_max": 0.05, "perdida_max": 0.10},
    "Agresivo": {"umbral_alerta": 0.65, "posicion_max": 0.10, "perdida_max": 0.15},
}
PERFIL_RIESGO_DEFECTO = "Moderado"
