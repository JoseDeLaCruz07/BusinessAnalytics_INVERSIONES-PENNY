# Onboarding para colaboradores

Guía para que alguien nuevo pueda clonar, configurar y correr este proyecto desde cero.

## 1. Acceso al repositorio

El repo es público para leer y clonar. Para subir cambios, el dueño (`UnnicD951`) debe agregarte como colaborador:
GitHub → repo `Penny_Stock` → **Settings** → **Collaborators** → **Add people**.

## 2. Clonar

```bash
git clone https://github.com/UnnicD951/Penny_Stock.git
cd Penny_Stock
```

Todo el trabajo vigente está en `main`.

## 3. Verifica tu versión de Python

```bash
python --version
```

Necesitas **Python 3.11 o 3.12**. Si tienes una versión más nueva (3.13+), varias
librerías de este proyecto (pandas, numpy, torch) todavía no tienen binarios
precompilados para Windows y la instalación va a fallar tratando de compilar
desde cero.

Si te pasa eso: instala Python 3.11 desde https://www.python.org/downloads/release/python-3119/
(no reemplaza tu versión actual, quedan las dos instaladas en paralelo) y usa
`py -3.11` en el siguiente paso en vez de `python`.

## 4. Entorno virtual + dependencias

```bash
python -m venv .venv          # o: py -3.11 -m venv .venv
```

Activar:
- Windows (PowerShell): `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

Si PowerShell da error de "execution policy" al activar, corre una vez:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```
Tarda varios minutos (instala transformers, torch, xgboost, streamlit, etc.).

## Vía rápida: ver el mismo dashboard que el resto del equipo

Si solo quieres **ver** el dashboard con los datos reales (sin correr ingesta ni entrenar), basta con:

1. Haber hecho los pasos 1 a 4 (acceso, clonar, Python 3.11/3.12, venv + `pip install -r requirements.txt`).
2. `cp .env.example .env` (Windows: `copy .env.example .env`) y poner en `.env` el `DATABASE_URL` de la base compartida (Supabase), que te pasa el dueño del repo por un canal privado:
   ```
   DATABASE_URL=postgresql://...
   ```
   `FINNHUB_API_KEY` **no** es necesaria para ver el dashboard, solo para ingestar datos.
3. `streamlit run app.py` y abrir http://localhost:8501.

El modelo viene en el repo (`models/xgboost_latest.json`) y los datos desde la base compartida, así que verás los mismos tickers, gráficos y predicciones. El dashboard guarda en la base las predicciones que visualiza (una por ticker/modelo/día), y los datos se cachean 10 minutos.

Los pasos 5 y 6 de abajo solo aplican si quieres tu propia base de datos o regenerar los datos.

## 5. Tu propia API key de Finnhub (solo si vas a ingestar datos)

**No uses la key de otra persona.** Es gratis, regístrate en https://finnhub.io
con tu propio email.

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edita `.env` y completa:
```
FINNHUB_API_KEY=tu_key_aqui
```
Este archivo nunca se sube a git (está en `.gitignore`) — cada persona tiene el suyo.

## 6. Correr el pipeline

Puedes empezar con datos sintéticos (sin necesitar la API key) para ver el
dashboard funcionando ya:
```bash
python -m src.ingestion.seed_synthetic_data --days 150
python -m src.models.feature_engineering
python -m src.models.train_xgboost
streamlit run app.py
```

O con datos reales (requiere `FINNHUB_API_KEY`):
```bash
python -m src.ingestion.ingest_market_data --days 730
python -m src.ingestion.ingest_text_data
python -m src.models.sentiment_finbert
python -m src.models.feature_engineering
python -m src.models.train_xgboost
python -m src.models.evaluate_models
streamlit run app.py
```

Ver el README principal para el detalle de cada script y los supuestos de
negocio (definición de penny stock, variable objetivo, umbrales de señales).

## 7. Antes de agregar tickers nuevos

El mercado de penny stocks rota mucho (delistings frecuentes). Antes de sumar
un ticker a `DEFAULT_TICKERS` (`src/config.py`), valídalo:
```bash
python -m src.ingestion.verificar_tickers SIMBOLO1 SIMBOLO2 SIMBOLO3
```

## Lo que NO necesitas hacer

- No copies la carpeta `data/` de otra persona — se regenera corriendo los
  scripts, y nunca se sube a git (son datos locales/regenerables).
- No pidas ni uses el `.env` de otra persona — saca tu propia key de Finnhub.
