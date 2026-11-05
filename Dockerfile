# Imagen para correr el dashboard (y los scripts del pipeline) igual que en Streamlit Cloud.
#   docker build -t penny-stocks .
#   docker run --rm -p 8501:8501 --env-file .env penny-stocks
# Las claves van en .env (nunca dentro de la imagen); ver .env.example.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch se instala en su version CPU (mas liviana); lo necesita solo FinBERT en el pipeline, no el dashboard.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
