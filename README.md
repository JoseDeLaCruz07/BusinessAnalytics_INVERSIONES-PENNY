# Penny Stocks Analytics

Sistema de analítica de negocio (descriptiva, predictiva y prescriptiva) para penny stocks, construido siguiendo la metodología **CRISP-ML(Q)**. Ver el prompt original del proyecto para el alcance completo por fases.

## Estado actual: Fase 1-5 implementadas, en producción

### Fase 1 — Data Engineering
- Modelo relacional (`src/db.py`, SQLAlchemy) con las 8 tablas del esquema: `tickers`, `precios_ohlcv`, `ratios_financieros`, `textos`, `sentimiento`, `features`, `predicciones`, `recomendaciones`. Funciona igual sobre SQLite (dev, por defecto) o PostgreSQL/Supabase (producción) según `DATABASE_URL`.
- `src/ingestion/ingest_market_data.py`: OHLCV vía `yfinance` + perfil/ratios vía `finnhub-python`. Filtra/marca penny stocks (precio de cierre < $5).
- `src/ingestion/ingest_sec_filings.py`: catalizadores desde **SEC EDGAR** (gratis, sin API key, requiere `SEC_USER_AGENT` con tu nombre y correo). Guarda en `filings_sec` los 8-K/6-K (clasificados por código de item: acuerdos, adquisiciones, dilución, deslistado, etc.), las participaciones mayores al 5% (13D/13G) y las ofertas de acciones. El tipo de evento se clasifica por reglas; no se infiere la dirección del precio, por eso los acuerdos y adquisiciones quedan neutrales. Se muestra en el Descriptivo como "Catalizadores (SEC)".
- `src/ingestion/ingest_text_data.py`: noticias vía **Finnhub News API** + mensajes vía la **API pública de StockTwits** (sin API key, texto crudo real que pasa por FinBERT igual que las noticias). Se evaluaron y se descartaron dos alternativas: Finnhub Social Sentiment (requiere plan de pago) y la API de Reddit (requiere aprobación manual que en la práctica no se consigue).
- `src/processing/clean_pipeline.py`: limpieza de nulos, recorte de outliers (IQR por ticker) y formatos de fecha en datos numéricos; limpieza de URLs/espacios y descarte de ruido en el corpus de texto.
- `src/ingestion/seed_synthetic_data.py`: genera datos **ficticios** (precios random walk, ratios y titulares de noticia inventados) para los `DEFAULT_TICKERS`, sin necesidad de API keys ni red. Sirve para probar el pipeline y, más adelante, el dashboard, mientras no corras la ingesta real. **No usar sus resultados como si fueran datos de mercado reales.**

### Fase 2 — Model Engineering

- `src/models/sentiment_finbert.py`: clasifica cada texto pendiente en `textos` con **FinBERT (ProsusAI/finbert)** vía `transformers`, guarda score + etiqueta (positivo/negativo/neutral) en `sentimiento`. Descarga ~440MB la primera vez que corre.
- `src/models/feature_engineering.py`: calcula RSI(14), media móvil(10), volumen relativo(20), precio relativo a su media (`precio_vs_sma`) y volatilidad (`vol_10d`, `vol_20d`, `rango_5d`) con la librería `ta`/pandas, y combina con el sentimiento diario agregado por ticker (si `sentiment_finbert.py` no se ha corrido, usa 0.0/neutral para no bloquear el pipeline). Guarda en `features` y exporta `data/features_export.csv`.
- `src/models/train_xgboost.py`: entrena un `XGBClassifier` con validación cruzada k-fold (accuracy, precision, recall, f1, matriz de confusión) y feature importance. Guarda el modelo versionado por fecha en `data/models/xgboost_YYYYMMDD.json` y un reporte en `data/models/reporte_YYYYMMDD.json`.

**Variable objetivo (definición de negocio confirmada contigo):** clasificación binaria — `1` si el precio de cierre sube más de **5%** en los próximos **3 días de trading**, `0` en caso contrario. Configurable en `src/config.py` (`TARGET_PCT_THRESHOLD`, `TARGET_HORIZON_DAYS`).

El pipeline corre en producción contra datos reales (Supabase), reentrenado automáticamente por `.github/workflows/actualizar_datos.yml` (ver Fase 5 más abajo).

### Fase 3 — ML Quality Assurance

- `src/models/evaluate_models.py`:
  - **XGBoost**: holdout temporal honesto (por cada ticker, el 20% de fechas más recientes se reserva como test y nunca se usa para entrenar) — más riguroso que el k-fold de `train_xgboost.py`, que mezcla fechas entre train/test. Reporta accuracy/precision/recall/f1/matriz de confusión/feature importance sobre ese holdout.
  - **FinBERT**: accuracy contra `tests/fixtures/finbert_validation_set.csv`, 18 frases financieras en inglés etiquetadas a mano (positivo/negativo/neutral), y lista de casos mal clasificados con su score.
  - **Checklist de calidad de datos**: completitud (nulos en precios/ratios), consistencia (precios ≤ 0), trazabilidad (textos sin URL), reproducibilidad (todas las dependencias de `requirements.txt` con versión fija).
  - Guarda todo en `data/models/evaluacion_YYYYMMDD.json`.

Detecta correctamente los casos mal clasificados de FinBERT y calcula el checklist sobre la DB real.

### Fase 4 — Dashboard (Streamlit)

- `app.py`: dashboard con selector de ticker + filtro de periodo (30 días / 90 días / 6 meses / 1 año / todo) en la barra lateral, gráficos interactivos (Plotly: zoom, hover, rango deslizante) y tres pestañas:
  - **Descriptivo**: "Panorama del mercado" con un scatter de todos los tickers (X = sentimiento promedio de los últimos 30 días, Y = probabilidad de suba del modelo XGBoost, tamaño = market cap; el cuadrante positivo en verde y el ticker elegido resaltado) y una tabla desplegable con los indicadores que influyen en el P/E (EPS, P/E, P/B, market cap). Debajo, el "Detalle" del ticker elegido: KPIs, precio de cierre, volumen, evolución del sentimiento, sección "Que dicen las noticias" (textos de Finnhub y StockTwits con su sentimiento), "Catalizadores (SEC)" e historial de ratios financieros. Funciona en modo claro y oscuro.
  - **Predictivo**: probabilidad de suba del último modelo XGBoost (gauge con los mismos umbrales de la señal prescriptiva), gráfico "Por qué esta probabilidad" (aporte de cada feature a la predicción de ese ticker, vía `pred_contribs` de XGBoost), catalizadores recientes de la SEC (últimos 30 días, por categoría), "Horizonte de precio" (zona objetivo a 3 días y estimación a 1 mes como rango de volatilidad histórica de 80% y 50% — **es un rango, no una proyección de precio**: XGBoost es un clasificador binario, no un pronosticador) y "Confiabilidad del modelo" (AUC, precisión, recall, tasa base y mejora vs. azar medidos en el 20% más reciente de cada ticker, entrenando solo con fechas anteriores; `evaluar_holdout` en `src/models/train_xgboost.py`).
  - **Prescriptivo**: señal de la regla de negocio (`señal de alerta` / `monitorear` / `sin señal`, `src/reports/business_rules.py`) con su justificación; **Asesor IA** (resumen, a favor, riesgos y qué vigilar, con nivel de confianza) escrito por Gemini; historial de predicciones pasadas con acierto/fallo (guarda una predicción por ticker/modelo/día en `predicciones`/`recomendaciones`); y señales de todos los tickers seguidos.
- `src/reports/asesor_ia.py`: genera una vez al día (en el workflow) un texto por ticker con **Gemini** y lo guarda en `recomendaciones_ia`; el dashboard solo lo lee. El LLM solo redacta: recibe como hechos el puntaje del modelo y su explicación, el rango a 1 mes, los catalizadores de la SEC, el sentimiento agregado y la confiabilidad real del modelo (AUC, tasa base); la confianza "alta" se rebaja a "media" si el AUC es menor a 0.65. Requiere `GEMINI_API_KEY` (gratis en aistudio.google.com/apikey); el modelo se cambia con `GEMINI_MODEL` y `--listar-modelos` muestra los que permite tu clave. `--dry-run` imprime el prompt sin llamar a la API.
- `src/reports/business_rules.py`: reglas auditables y separadas del modelo — umbrales de probabilidad (`UMBRAL_SENAL_ALERTA=0.65`, `UMBRAL_SENAL_MONITOREAR=0.40`) + contexto técnico (RSI en sobreventa/sobrecompra, spike de volumen, sentimiento fuerte). Todo configurable en `src/config.py`.

Desplegado en Streamlit Community Cloud contra Supabase en producción (ver Fase 5).

## Supuestos de negocio (regla 5 del prompt original)

- **Penny stock** = acción con precio de cierre < **$5 USD** (`PENNY_STOCK_PRICE_THRESHOLD` en `src/config.py`). No se descarta del histórico si sube de precio; solo se marca en el log.
- Tickers monitoreados (`DEFAULT_TICKERS` en `src/config.py`): `SNDL, CTRM, PDSB, RETO, BNGO, KOPN, ABUS, EOSE, UPXI, BATL, SLND, LCID, VRRM, EGHT, PTLO, WOOF, JBI, TBLA`. Configurables por `--tickers` en cada script o editando `src/config.py`; validar reemplazos con `python -m src.ingestion.verificar_tickers`.
- Fuentes de texto: noticias (Finnhub) + mensajes de StockTwits.
- Limpieza de outliers de precio: recorte (clip) por IQR con factor 3 sobre `close`, por ticker — no se elimina la fila, se acota el valor.
- Ratios `pe_ratio`/`pb_ratio` negativos o cero se marcan como `NA` (no tienen sentido de negocio) en vez de descartar la fila completa.
- **Variable objetivo de XGBoost**: `1` si `close` sube > 5% en 3 días de trading siguientes, `0` si no (`TARGET_PCT_THRESHOLD`, `TARGET_HORIZON_DAYS` en `src/config.py`).
- Ventanas de indicadores técnicos: RSI(14), SMA(10), volumen relativo sobre promedio móvil de 20 días (`RSI_WINDOW`, `SMA_WINDOW`, `VOLUME_RELATIVE_WINDOW`).
- **Features del modelo** (`FEATURE_COLS` en `train_xgboost.py`): `rsi`, `volumen_relativo`, `sentimiento_agregado`, `precio_vs_sma`, `vol_10d`, `vol_20d`, `rango_5d`, con XGBoost de poca profundidad (`MODELO_PARAMS`). Elegidas con `experimentos.py`: sin la media móvil en dólares y con volatilidad, el AUC pasó de 0.52 a ~0.60 (evaluación en 4 tramos consecutivos con purga). La volatilidad indica *cuándo* son probables los movimientos grandes, no hacia dónde.
- **Esquema**: `init_db()` agrega solas las columnas nuevas a tablas ya existentes (sin tocar datos), también en Supabase.
- Sentimiento agregado diario = promedio del score de FinBERT firmado por etiqueta (positivo=+score, negativo=-score, neutral=0). Días sin noticias quedan en 0.0 (neutral), no se descartan filas.
- **Señales prescriptivas**: `señal de alerta` si prob ≥ 65%, `monitorear` si prob ≥ 40%, `sin señal` si no (`UMBRAL_SENAL_ALERTA`, `UMBRAL_SENAL_MONITOREAR` en `src/config.py`). RSI ≤ 30 = sobreventa, RSI ≥ 70 = sobrecompra, volumen ≥ 2x el promedio = spike — se mencionan como contexto, no cambian el tipo de señal por sí solos.

## Cómo correr Fase 1

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Opción A — datos sintéticos (sin API key, para probar el pipeline ya)

```bash
python -m src.ingestion.seed_synthetic_data         # datos ficticios -> DB
python -m src.processing.clean_pipeline             # limpieza -> data/*_clean.csv
```

### Opción B — datos reales (requiere FINNHUB_API_KEY)

```bash
cp .env.example .env   # completa FINNHUB_API_KEY

python -m src.ingestion.ingest_market_data          # OHLCV + ratios -> DB + data/*.csv
python -m src.ingestion.ingest_text_data            # noticias -> DB + data/textos_export.csv
python -m src.ingestion.ingest_sec_filings          # catalizadores SEC (requiere SEC_USER_AGENT en .env)
python -m src.processing.clean_pipeline             # limpieza -> data/*_clean.csv
```

La base de datos SQLite queda en `data/penny_stocks.db` en ambos casos (comparten el mismo esquema; si corres primero la sintética y luego quieres datos reales, borra `data/penny_stocks.db` antes para no mezclarlos).

## Cómo correr Fase 2 (después de Fase 1)

```bash
python -m src.models.sentiment_finbert        # requiere red a huggingface.co (descarga FinBERT la 1ra vez)
python -m src.models.feature_engineering      # RSI/SMA/volumen relativo + sentimiento -> tabla features
python -m src.models.train_xgboost            # entrena, valida k-fold, guarda modelo en data/models/
```

Si aún no corriste `sentiment_finbert.py`, `feature_engineering.py` igual funciona (deja el sentimiento en neutral) para no bloquearte.

## Cómo correr Fase 3 (después de Fase 2)

```bash
python -m src.models.evaluate_models          # requiere red a huggingface.co (evalua FinBERT tambien)
```

Genera `data/models/evaluacion_YYYYMMDD.json` con las tres evaluaciones (XGBoost, FinBERT, calidad de datos).

## Cómo correr Fase 4 (dashboard)

```bash
streamlit run app.py
```

Necesita al menos: datos en `precios_ohlcv` (Fase 1) y un modelo entrenado (Fase 2; el repo trae el último en `models/xgboost_latest.json`). Para ver los mismos datos que el resto del equipo, ver `ONBOARDING.md` (solo requiere el `DATABASE_URL` compartido). Si no hay `features`/sentimiento, la pestaña Predictivo avisa qué falta en vez de fallar.

## Estructura del proyecto

```
penny-stocks-analytics/
├── data/                  # datos crudos y procesados (gitignored)
├── src/
│   ├── config.py          # tickers, umbrales de negocio, credenciales
│   ├── db.py               # modelo relacional (SQLAlchemy)
│   ├── ingestion/          # ingest_market_data.py, ingest_text_data.py, ingest_sec_filings.py
│   ├── processing/         # clean_pipeline.py
│   ├── models/             # Fase 2/3: sentiment_finbert.py, feature_engineering.py, train_xgboost.py, evaluate_models.py
│   └── reports/            # Fase 4: business_rules.py
├── app.py                 # dashboard Streamlit (Fase 4)
├── tests/
│   └── fixtures/           # finbert_validation_set.csv (set etiquetado a mano)
├── .github/workflows/      # actualizar_datos.yml: ingesta y reentrenamiento automatico
├── requirements.txt
├── .env.example
└── README.md
```

## Fase 5 — Deployment (Streamlit Community Cloud + Supabase)

`app.py` ya soporta Postgres (además de SQLite) vía `DATABASE_URL`, y lee credenciales tanto de `.env` (local) como de `st.secrets` (Streamlit Cloud) — ver el bloque al inicio de `app.py` que copia `st.secrets` a variables de entorno antes de importar `src.config`.

### 1. Crear la base de datos en Supabase (gratis)

1. Crea una cuenta en [supabase.com](https://supabase.com) y un proyecto nuevo.
2. En **Project Settings → Database → Connection string**, copia la URI (formato `postgresql://postgres:[PASSWORD]@[HOST]:5432/postgres`).
3. En tu `.env` local, reemplaza `DATABASE_URL` por esa URI (guarda tu SQLite local aparte si quieres conservarla — son bases distintas).
4. Vuelve a correr el pipeline completo (Fases 1-3) apuntando a Supabase: `ingest_market_data.py` → `ingest_text_data.py` → `sentiment_finbert.py` → `feature_engineering.py` → `train_xgboost.py`. Esto crea las tablas en Supabase automáticamente (mismo esquema que SQLite) y las llena con datos reales.

### 2. Desplegar el dashboard

1. Verifica que `main` esté pusheada a GitHub (ya lo está).
2. Ve a [share.streamlit.io](https://share.streamlit.io), entra con tu cuenta de GitHub.
3. **New app** → selecciona el repo `Penny_Stock`, la rama `main`, y el archivo principal `app.py`.
4. En **Advanced settings → Secrets**, pega (formato TOML):
   ```toml
   FINNHUB_API_KEY = "tu_key_real"
   DATABASE_URL = "postgresql://postgres:...@....supabase.co:5432/postgres"
   ```
5. Deploy. La primera build puede tardar varios minutos (instala `torch`/`transformers`, aunque `app.py` no los usa directamente — solo los necesitan los scripts de ingesta que corres localmente).
6. Te da una URL pública tipo `https://tu-app.streamlit.app` — cualquiera con el link puede verla, no requiere login.

### 3. Mantener los datos actualizados (automático con GitHub Actions)

El dashboard en la nube solo **lee** de Supabase, no ingesta nada por sí solo. Para que
los datos y el modelo no queden viejos, `.github/workflows/actualizar_datos.yml` corre
automáticamente de **lunes a viernes a las 21:30 UTC** (después del cierre del mercado
de EE.UU.) y hace todo el pipeline contra Supabase: ingesta de mercado y texto →
limpieza → sentimiento (FinBERT) → feature engineering → reentrenar XGBoost → publicar
el modelo nuevo en `models/` con un commit automático.

**Configuración necesaria (una sola vez), para que el workflow pueda escribir a Supabase:**

1. Entrá al repo en GitHub → **Settings** → **Secrets and variables** → **Actions**.
2. Click en **New repository secret** y creá estos:
   - `FINNHUB_API_KEY` → tu API key de Finnhub (la misma que usás localmente).
   - `DATABASE_URL` → la cadena de conexión de Supabase (Session pooler), formato
     `postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   - `GEMINI_API_KEY` → clave gratuita de Gemini (opcional: sin ella solo falla el paso del asesor IA). Opcional también la *variable* de repositorio `GEMINI_MODEL`.
   - `SEC_USER_AGENT` → tu nombre y correo, ej. `Penny Stock Analytics tu_correo@dominio.com` (la SEC lo exige; sin él, solo ese paso falla y el resto del pipeline sigue).
3. Listo. El workflow ya está en `.github/workflows/actualizar_datos.yml` — no hace
   falta tocar nada más. Podés probarlo a mano en cualquier momento desde la pestaña
   **Actions** del repo → "Actualizar datos y reentrenar modelo" → **Run workflow**.

Si el modelo cambió (hubo datos nuevos que movieron los pesos), el workflow hace commit
y push de `models/xgboost_latest.json` + `models/xgboost_latest.version.txt` con el
usuario `github-actions[bot]`; Streamlit Cloud redeploya solo al detectar el push. Si no
hubo cambios en el modelo, no genera commit vacío.

### 4. Actualizar el modelo entrenado en la nube manualmente

Si preferís reentrenar a mano en vez de esperar al workflow automático:
- `data/models/xgboost_YYYYMMDD.json` — histórico local, gitignored (no se sube).
- `models/xgboost_latest.json` + `models/xgboost_latest.version.txt` — **sí se versiona en git**, es lo que usa el dashboard desplegado.

```bash
python -m src.models.train_xgboost
git add models/xgboost_latest.json models/xgboost_latest.version.txt
git commit -m "Actualizar modelo entrenado"
git push
```

### Docker

```bash
docker build -t penny-stocks .
docker run --rm -p 8501:8501 --env-file .env penny-stocks   # http://localhost:8501
```

## Backtesting, evidencias y reglas de riesgo

- `python -m src.models.backtesting --solo-lectura`: backtesting **fuera de muestra** (mismos 4 tramos walk-forward con purga que la
  confiabilidad del modelo). Compara "senal de alerta" y "top 3 puntaje" contra comprar y mantener, con 0.1% de comision y 0.2% de
  deslizamiento por lado, caida maxima y metricas de clasificacion contra la linea base "siempre no sube". Las ventanas con saltos
  diarios > 60% (splits sin ajustar) se reportan aparte. Guarda `docs/evidencias/backtest_YYYYMMDD.json` y `.png`.
- `python -m src.reports.evidencias_datos --solo-lectura`: conteos por tabla, tamano de la base, chequeos de calidad y un ejemplo
  real de la entrada del LLM. Guarda `docs/evidencias/evidencias_datos_YYYYMMDD.json`.
- `--solo-lectura` usa `SUPABASE_URL_LECTURA` en una conexion donde Postgres rechaza escrituras (`DATABASE_URL` puede seguir en SQLite).
- **Freno de volatilidad** (`VOLATILIDAD_DIARIA_MAXIMA` en `src/config.py`): con volatilidad diaria de 10 dias > 25% no se emite
  "senal de alerta" (suele ser un reverse split sin ajustar).
- **Perfil de riesgo** (`PERFILES_RIESGO`): conservador / moderado / agresivo cambia el umbral de alerta y muestra posicion maxima
  y stop-loss sugeridos en el Prescriptivo (reglas fijas, educativas).
- **Validacion del asesor IA**: `validar_respuesta` en `asesor_ia.py` busca cifras que no estan en los hechos y contradicciones
  (puntaje vs. tasa base, puesto, compra con "sin senal"); reintenta una vez y, si persiste, publica con confianza baja y un aviso.

## Calidad de datos y monitoreo

- **Precios ajustados por splits** (`ingest_market_data.py`): se descarga con `auto_adjust=True` desde la primera fecha guardada
  hasta hoy inclusive. Si algun cierre guardado difiere mas de 1% (`UMBRAL_REESCRITURA`) del que devuelve Yahoo, se reemplaza el
  historial completo del ticker (un split posterior cambio la escala). Los saltos diarios > 60% quedan avisados en el log.
  El feature engineering recalcula solo las filas que cambian.
- **Limpieza compartida** (`limpiar_ohlcv` en `clean_pipeline.py`): la usan el feature engineering, el entrenamiento y el backtest
  (precios vacios o <= 0, high < low, duplicados). No recorta outliers: un +50% real es lo que el modelo tiene que aprender.
- **Registro diario de predicciones** (`src/reports/predicciones_diarias.py`): paso del workflow que guarda el puntaje y la senal
  de los 18 tickers cada dia, sin depender de que alguien abra el dashboard.
- **Monitoreo del modelo** (`models/monitoreo_modelo.csv`, versionado): una fila por reentrenamiento con AUC walk-forward, lift y
  si se publico. Con AUC < 0.52 (`AUC_MINIMO_PUBLICAR`) el modelo nuevo no reemplaza al publicado; con AUC < 0.55 (`AUC_AVISO`)
  se publica con advertencia y el Predictivo muestra un aviso. El Predictivo grafica la evolucion del AUC y el ultimo backtest.
- **Limite diario del LLM** (`MAX_LLAMADAS_LLM_DIA`, por defecto 40): contador por dia en la tabla `uso_llm`, cuenta reintentos
  y corridas repetidas del workflow; al llegar al tope los tickers restantes conservan su recomendacion anterior.
