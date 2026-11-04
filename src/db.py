"""Modelo relacional del proyecto (SQLAlchemy declarative).

Mismo esquema para SQLite local (desarrollo) y PostgreSQL/Supabase (produccion),
controlado por DATABASE_URL en .env.
"""
from __future__ import annotations

import datetime as dt

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
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from src.config import DATABASE_URL


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
    fuente = Column(String(20), nullable=False)  # 'reddit' | 'noticia'
    fecha = Column(DateTime, nullable=False)
    contenido = Column(Text, nullable=False)
    url = Column(String(500))

    ticker = relationship("Ticker", back_populates="textos")
    sentimiento = relationship("Sentimiento", back_populates="texto", uselist=False)


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
    """Crea todas las tablas si no existen (dev SQLite o Postgres ya provisionado)."""
    Base.metadata.create_all(_engine)


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