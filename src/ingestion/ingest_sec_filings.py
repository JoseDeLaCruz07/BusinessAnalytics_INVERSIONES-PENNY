"""Fase 1 - Ingesta de catalizadores: filings de la SEC (EDGAR), gratis y sin API key.

Guarda 8-K/6-K (con su item), participaciones >5% (13D/13G) y ofertas de acciones.
Requiere SEC_USER_AGENT en .env (la SEC lo exige y bloquea sin el).

Uso:
    python -m src.ingestion.ingest_sec_filings
    python -m src.ingestion.ingest_sec_filings --tickers SNDL BATL --days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

import requests

from src.config import DEFAULT_TICKERS, SEC_USER_AGENT
from src.db import FilingSEC, get_or_create_ticker, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URL_TICKERS = "https://www.sec.gov/files/company_tickers.json"
URL_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
URL_ARCHIVO = "https://data.sec.gov/submissions/{nombre}"
URL_DOCUMENTO = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_sin_guiones}/{documento}"
PAUSA_SEGUNDOS = 0.2  # el limite de la SEC es 10 consultas por segundo

# Item de 8-K -> (categoria, signo, descripcion). El item dice el tipo de evento, no su direccion:
# un 1.01 puede ser un contrato comercial o un financiamiento, por eso queda neutral.
ITEMS_8K = {
    "1.01": ("Acuerdo", "neutral", "Acuerdo material"),
    "1.02": ("Acuerdo", "negativo", "Termino de acuerdo material"),
    "1.03": ("Quiebra", "negativo", "Quiebra o concurso"),
    "2.01": ("Adquisicion", "neutral", "Adquisicion o venta de activos"),
    "2.02": ("Resultados", "neutral", "Resultados financieros"),
    "2.03": ("Deuda", "neutral", "Nueva obligacion financiera"),
    "2.04": ("Deuda", "negativo", "Aceleracion de deuda"),
    "2.05": ("Reestructuracion", "negativo", "Costos de reestructuracion"),
    "2.06": ("Deterioro", "negativo", "Deterioro de activos"),
    "3.01": ("Deslistado", "negativo", "Aviso de deslistado"),
    "3.02": ("Dilucion", "negativo", "Venta de acciones no registradas"),
    "4.01": ("Auditoria", "neutral", "Cambio de auditor"),
    "4.02": ("Auditoria", "negativo", "Estados financieros no confiables"),
    "5.01": ("Control", "neutral", "Cambio de control"),
    "5.02": ("Directivos", "neutral", "Cambio de directivos"),
    "7.01": ("Comunicado", "neutral", "Divulgacion Reg FD"),
    "8.01": ("Otros eventos", "neutral", "Otros eventos"),
}
# Cuando un 8-K trae varios items manda el primero de esta lista (el mas relevante); 9.01 solo son anexos.
PRIORIDAD_ITEMS = ["3.01", "1.03", "4.02", "1.02", "3.02", "2.04", "2.05", "2.06", "1.01", "2.01", "5.01",
                   "2.03", "2.02", "4.01", "5.02", "7.01", "8.01"]

OFERTAS = ("S-1", "S-3", "F-1", "F-3")


def clasificar(formulario: str, items: str) -> tuple[str, str, str] | None:
    """(categoria, signo, descripcion) del filing, o None si no es un catalizador que sigamos."""
    f = formulario.upper().strip()
    if f == "8-K":
        codigos = [c.strip() for c in (items or "").split(",") if c.strip() in ITEMS_8K]
        if not codigos:
            return None
        codigos.sort(key=PRIORIDAD_ITEMS.index)
        categoria, signo, _ = ITEMS_8K[codigos[0]]
        return categoria, signo, " / ".join(ITEMS_8K[c][2] for c in codigos)
    if f == "6-K":
        return "Comunicado", "neutral", "Comunicado de emisor extranjero"
    if f.startswith(("SC 13D", "SC 13G", "SCHEDULE 13D", "SCHEDULE 13G")):
        return "Participacion", "neutral", "Participacion accionaria mayor al 5%"
    if f.split("/")[0] in OFERTAS or f.startswith("424B"):
        return "Dilucion", "negativo", "Oferta de acciones (posible dilucion)"
    return None


def filas_desde_submissions(data: dict, desde: dt.date, bloque: dict | None = None) -> list[dict]:
    """Convierte el JSON de submissions de un emisor en filas listas para guardar.

    `bloque` es el dict de columnas paralelas a leer; por defecto el de filings recientes.
    """
    cik = int(data["cik"])
    r = bloque if bloque is not None else data.get("filings", {}).get("recent", {})
    filas = []
    for i, accession in enumerate(r.get("accessionNumber", [])):
        fecha_txt = (r.get("acceptanceDateTime") or [None] * (i + 1))[i] or r["filingDate"][i]
        fecha = dt.datetime.fromisoformat(fecha_txt.replace("Z", "+00:00")).replace(tzinfo=None)
        if fecha.date() < desde:
            continue
        formulario = r["form"][i]
        items = (r.get("items") or [""] * (i + 1))[i]
        clase = clasificar(formulario, items)
        if clase is None:
            continue
        documento = (r.get("primaryDocument") or [""] * (i + 1))[i] or f"{accession}-index.htm"
        filas.append(
            {
                "accession": accession,
                "fecha": fecha,
                "formulario": formulario,
                "items": items or None,
                "categoria": clase[0],
                "signo": clase[1],
                "descripcion": clase[2],
                "url": URL_DOCUMENTO.format(
                    cik=cik, acc_sin_guiones=accession.replace("-", ""), documento=documento
                ),
            }
        )
    return filas


def archivos_antiguos(data: dict, desde: dt.date) -> list[str]:
    """Nombres de los archivos extra con filings mas viejos que los recientes y que caen dentro de la ventana."""
    return [
        a["name"]
        for a in data.get("filings", {}).get("files", [])
        if dt.date.fromisoformat(a["filingTo"]) >= desde
    ]


def _get_json(sesion: requests.Session, url: str) -> dict:
    resp = sesion.get(url, timeout=20)
    resp.raise_for_status()
    time.sleep(PAUSA_SEGUNDOS)
    return resp.json()


def run(tickers: list[str], days: int) -> None:
    if not SEC_USER_AGENT:
        logger.error(
            "SEC_USER_AGENT no esta configurada: la SEC exige identificarse. "
            "Agregala a .env, ej. SEC_USER_AGENT=Penny Stock Analytics tu_correo@dominio.com"
        )
        sys.exit(1)

    init_db()
    session = get_session()
    sesion_http = requests.Session()
    sesion_http.headers.update({"User-Agent": SEC_USER_AGENT, "Accept": "application/json"})
    desde = dt.date.today() - dt.timedelta(days=days)

    mapa_cik = {v["ticker"].upper(): v["cik_str"] for v in _get_json(sesion_http, URL_TICKERS).values()}

    for symbol in tickers:
        cik = mapa_cik.get(symbol.upper())
        if cik is None:
            logger.warning("%s no figura en el listado de la SEC; se omite", symbol)
            continue
        logger.info("Descargando filings de la SEC para %s (CIK %s)", symbol, cik)
        try:
            data = _get_json(sesion_http, URL_SUBMISSIONS.format(cik=int(cik)))
            filas = filas_desde_submissions(data, desde)
            for nombre in archivos_antiguos(data, desde):
                extra = _get_json(sesion_http, URL_ARCHIVO.format(nombre=nombre))
                filas += filas_desde_submissions(data, desde, bloque=extra)
        except Exception:
            logger.exception("Fallo SEC EDGAR para %s", symbol)
            continue

        ticker_obj = get_or_create_ticker(session, symbol=symbol)
        existentes = {a for (a,) in session.query(FilingSEC.accession).filter_by(ticker_id=ticker_obj.id)}
        nuevas = list({f["accession"]: f for f in filas if f["accession"] not in existentes}.values())
        session.add_all(FilingSEC(ticker_id=ticker_obj.id, **f) for f in nuevas)
        session.commit()
        logger.info("%s: %d filings relevantes, %d nuevos", symbol, len(filas), len(nuevas))

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de filings de la SEC (EDGAR)")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run(args.tickers, args.days)
