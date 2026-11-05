"""Pruebas de la clasificacion y el parseo de filings de la SEC (sin red, con JSON en el formato de EDGAR)."""
import datetime as dt

from src.ingestion.ingest_sec_filings import archivos_antiguos, clasificar, filas_desde_submissions

SUBMISSIONS = {
    "cik": "1045810",
    "filings": {
        "recent": {
            "accessionNumber": ["0001-24-000003", "0001-24-000002", "0001-24-000001", "0001-23-000009", "0001-24-000004"],
            "filingDate": ["2024-05-03", "2024-05-02", "2024-05-01", "2023-01-01", "2024-04-30"],
            "acceptanceDateTime": [
                "2024-05-03T21:05:00.000Z", "2024-05-02T13:00:00.000Z", "2024-05-01T20:10:00.000Z",
                "2023-01-01T10:00:00.000Z", "2024-04-30T09:00:00.000Z",
            ],
            "form": ["8-K", "SC 13D", "424B5", "8-K", "10-Q"],
            "items": ["1.01,9.01", "", "", "2.02", ""],
            "primaryDocument": ["a8k.htm", "d13.htm", "p.htm", "old.htm", "q.htm"],
        }
    },
}


def test_8k_prioriza_el_item_mas_relevante_e_ignora_anexos():
    assert clasificar("8-K", "9.01,3.02,8.01") == (
        "Dilucion", "negativo", "Venta de acciones no registradas / Otros eventos"
    )
    assert clasificar("8-K", "9.01") is None


def test_formularios_sin_interes_se_descartan():
    assert clasificar("10-Q", "") is None
    assert clasificar("4", "") is None


def test_participacion_y_ofertas():
    assert clasificar("SCHEDULE 13D/A", "")[0] == "Participacion"
    assert clasificar("424B5", "")[1] == "negativo"
    assert clasificar("S-3/A", "")[0] == "Dilucion"


def test_filas_filtra_por_fecha_y_arma_la_url():
    filas = filas_desde_submissions(SUBMISSIONS, dt.date(2024, 4, 1))
    assert [f["accession"] for f in filas] == ["0001-24-000003", "0001-24-000002", "0001-24-000001"]
    assert filas[0]["categoria"] == "Acuerdo"
    assert filas[0]["fecha"] == dt.datetime(2024, 5, 3, 21, 5)
    assert filas[0]["url"] == "https://www.sec.gov/Archives/edgar/data/1045810/000124000003/a8k.htm"


def test_archivos_antiguos_solo_los_que_caen_en_la_ventana():
    data = {"filings": {"files": [
        {"name": "CIK1-submissions-001.json", "filingTo": "2024-01-10"},
        {"name": "CIK1-submissions-002.json", "filingTo": "2022-03-01"},
    ]}}
    assert archivos_antiguos(data, dt.date(2023, 6, 1)) == ["CIK1-submissions-001.json"]


def test_bloque_antiguo_se_lee_igual_que_el_reciente():
    bloque = SUBMISSIONS["filings"]["recent"]
    assert filas_desde_submissions(SUBMISSIONS, dt.date(2023, 1, 1), bloque=bloque)[-1]["accession"] == "0001-23-000009"
