"""Modelo relacional (SQLAlchemy declarative); mismo esquema para SQLite y Postgres/Supabase via DATABASE_URL."""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from src.config import DATABASE_URL


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False, index=True)
    nombre = Column(String(255))
    sector = Column(String(120))
    fecha_alta = Column(DateTime, default=dt.datetime.utcnow)

    precios = relationship("PrecioOHLCV", back_populates="ticker")
    ratios = relationship("RatioFinanciero", back_populates="ticker")
    textos = relationship("Texto", back_populates="ticker")
    filings = relationship("FilingSEC", back_populates="ticker")
    features = relationship("Feature", back_populates="ticker")
    predicciones = relationship("Prediccion", back_populates="ticker")


class PrecioOHLCV(Base):
    __tablename__ = "precios_ohlcv"
    __table_args__ = (UniqueConstraint("ticker_id", "fecha", name="uq_precio_ticker_fecha"),)

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volumen = Column(Integer)

    ticker = relationship("Ticker", back_populates="precios")


class RatioFinanciero(Base):
    __tablename__ = "ratios_financieros"
    __table_args__ = (UniqueConstraint("ticker_id", "fecha", name="uq_ratio_ticker_fecha"),)

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    eps = Column(Float)
    shares_outstanding = Column(Float)

    ticker = relationship("Ticker", back_populates="ratios")


class Texto(Base):
    __tablename__ = "textos"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fuente = Column(String(20), nullable=False)  # 'noticia' | 'stocktwits'
    fecha = Column(DateTime, nullable=False)
    contenido = Column(Text, nullable=False)
    url = Column(String(500))

    ticker = relationship("Ticker", back_populates="textos")
    sentimiento = relationship("Sentimiento", back_populates="texto", uselist=False)


class FilingSEC(Base):
    __tablename__ = "filings_sec"
    __table_args__ = (UniqueConstraint("ticker_id", "accession", name="uq_filing_ticker_accession"),)

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    accession = Column(String(25), nullable=False)
    fecha = Column(DateTime, nullable=False)
    formulario = Column(String(20), nullable=False)
    items = Column(String(120))  # codigos de item del 8-K, ej. "1.01,9.01"
    categoria = Column(String(40), nullable=False)
    signo = Column(String(10), nullable=False)  # positivo | negativo | neutral
    descripcion = Column(String(300), nullable=False)
    url = Column(String(500))

    ticker = relationship("Ticker", back_populates="filings")


class RecomendacionIA(Base):
    __tablename__ = "recomendaciones_ia"
    __table_args__ = (UniqueConstraint("ticker_id", "fecha", name="uq_recomendacion_ia_ticker_fecha"),)

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    modelo_llm = Column(String(60), nullable=False)
    senal = Column(String(30), nullable=False)  # senal de la regla de negocio, el ancla auditable
    puntaje = Column(Float, nullable=False)  # puntaje de XGBoost al generar el texto
    resumen = Column(Text, nullable=False)
    a_favor = Column(Text, nullable=False)  # JSON: lista de textos
    riesgos = Column(Text, nullable=False)  # JSON
    vigilar = Column(Text, nullable=False)  # JSON
    confianza = Column(String(10), nullable=False)  # baja | media | alta
    # Opcionales (init_db las agrega solas a la tabla existente; las filas viejas quedan en NULL):
    accion_sugerida = Column(String(30))  # vigilar de cerca | esperar confirmacion | evitar (acotada por la senal)
    justificacion_accion = Column(Text)
    alternativas = Column(Text)  # JSON: lista de textos
    noticias_clave = Column(Text)  # JSON: [{id, fecha, texto, url}] citados por el LLM y verificados
    creada_en = Column(DateTime, default=dt.datetime.utcnow)

    ticker = relationship("Ticker")


class UsoLLM(Base):
    """Contador de consultas al LLM por dia: hace cumplir MAX_LLAMADAS_LLM_DIA aunque el workflow se corra varias veces."""

    __tablename__ = "uso_llm"

    fecha = Column(Date, primary_key=True)
    llamadas = Column(Integer, nullable=False, default=0)


class Sentimiento(Base):
    __tablename__ = "sentimiento"

    id = Column(Integer, primary_key=True)
    texto_id = Column(Integer, ForeignKey("textos.id"), nullable=False)
    score = Column(Float, nullable=False)
    etiqueta = Column(String(20), nullable=False)  # positivo | negativo | neutral
    modelo_version = Column(String(50), nullable=False)

    texto = relationship("Texto", back_populates="sentimiento")


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("ticker_id", "fecha", name="uq_feature_ticker_fecha"),)

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    rsi = Column(Float)
    media_movil = Column(Float)
    volumen_relativo = Column(Float)
    sentimiento_agregado = Column(Float)
    precio_vs_sma = Column(Float)  # close / media_movil - 1
    vol_10d = Column(Float)  # desviacion estandar del retorno diario, 10 dias
    vol_20d = Column(Float)
    rango_5d = Column(Float)  # promedio 5 dias de (high - low) / close

    ticker = relationship("Ticker", back_populates="features")


class Prediccion(Base):
    __tablename__ = "predicciones"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    modelo_version = Column(String(50), nullable=False)
    prediccion = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    features_importantes = Column(Text)  # JSON serializado

    ticker = relationship("Ticker", back_populates="predicciones")
    recomendaciones = relationship("Recomendacion", back_populates="prediccion")


class Recomendacion(Base):
    __tablename__ = "recomendaciones"

    id = Column(Integer, primary_key=True)
    prediccion_id = Column(Integer, ForeignKey("predicciones.id"), nullable=False)
    tipo_senal = Column(String(30), nullable=False)
    justificacion = Column(Text, nullable=False)
    fecha_generacion = Column(DateTime, default=dt.datetime.utcnow)

    prediccion = relationship("Prediccion", back_populates="recomendaciones")


_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=_engine)


def init_db() -> None:
    """Crea las tablas que falten y agrega a las existentes las columnas nuevas (dev SQLite o Postgres ya provisionado)."""
    Base.metadata.create_all(_engine)
    _agregar_columnas_faltantes()


def _agregar_columnas_faltantes() -> None:
    """create_all no altera tablas existentes: se agregan solo columnas opcionales, sin tocar datos."""
    try:
        inspector = inspect(_engine)
        with _engine.begin() as conn:
            for tabla in Base.metadata.sorted_tables:
                if not inspector.has_table(tabla.name):
                    continue
                existentes = {c["name"] for c in inspector.get_columns(tabla.name)}
                for col in tabla.columns:
                    if col.name not in existentes and col.nullable and col.default is None:
                        tipo = col.type.compile(dialect=_engine.dialect)
                        conn.execute(text(f'ALTER TABLE "{tabla.name}" ADD COLUMN "{col.name}" {tipo}'))
    except Exception:
        logger.exception("No se pudieron agregar columnas nuevas; revisa los permisos del usuario de la base")


def get_session():
    return SessionLocal()


def get_or_create_ticker(session, symbol: str, nombre: str | None = None, sector: str | None = None) -> Ticker:
    ticker = session.query(Ticker).filter_by(symbol=symbol).first()
    if ticker is None:
        ticker = Ticker(symbol=symbol, nombre=nombre, sector=sector)
        session.add(ticker)
        session.commit()
        session.refresh(ticker)
    return ticker
