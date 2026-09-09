"""API tests.

Each test gets its own temporary SQLite file and a database seeded through the
real pipeline, so these exercise the actual stack rather than mocks of it.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DOCUMENT_STORAGE", str(tmp_path / "storage"))

    import src.api

    importlib.reload(src.api)

    from src.db import init_db
    from src.llm import MockProvider
    from src.pipeline import InvoicePipeline
    from src.repository import InvoiceRepository

    init_db(src.api.engine)
    pipeline = InvoicePipeline(provider=MockProvider(), use_cache=False)
    with src.api.SessionFactory() as session:
        repo = InvoiceRepository(session)
        for name in ("invoice_clean_pkr.pdf", "invoice_bad_math.pdf"):
            repo.save(pipeline.process(f"samples/{name}"))

    with TestClient(src.api.app) as c:
        yield c


def first_queued(client):
    items = client.get("/api/queue").json()
    assert items, "seed data should leave something needing review"
    return items[0]


# ------------------------------------------------------------------- reads

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_stats_shape(client):
    body = client.get("/api/stats").json()
    assert body["documents"] == 2
    assert "straight_through_rate" in body


def test_queue_is_worst_first(client):
    items = client.get("/api/queue").json()
    scores = [i["overall_confidence"] for i in items]
    assert scores == sorted(scores)


def test_detail_includes_everything_the_reviewer_needs(client):
    detail = client.get(f"/api/extractions/{first_queued(client)['id']}").json()
    for key in ("confidences", "issues", "history", "line_items", "filename", "model"):
        assert key in detail, key
    assert detail["history"], "the extraction event should already be audited"


def test_unknown_extraction_is_404(client):
    assert client.get("/api/extractions/9999").status_code == 404


# ------------------------------------------------------------- corrections

def test_correction_updates_and_audits(client):
    eid = first_queued(client)["id"]
    body = client.patch(
        f"/api/extractions/{eid}",
        json={"field": "invoice_number", "value": "CORRECTED-1", "actor": "aymen"},
    ).json()

    assert body["invoice_number"] == "CORRECTED-1"
    last = body["history"][-1]
    assert last["action"] == "corrected"
    assert last["actor"] == "aymen"
    assert last["new_value"] == "CORRECTED-1"


def test_numeric_correction_is_coerced(client):
    eid = first_queued(client)["id"]
    body = client.patch(
        f"/api/extractions/{eid}", json={"field": "total_amount", "value": "1,234.50"}
    ).json()
    assert float(body["total_amount"]) == 1234.50


def test_bad_number_is_rejected_with_a_useful_message(client):
    eid = first_queued(client)["id"]
    r = client.patch(
        f"/api/extractions/{eid}", json={"field": "total_amount", "value": "abc"}
    )
    assert r.status_code == 422
    assert "not a number" in r.json()["detail"]


def test_bad_date_says_what_format_is_expected(client):
    eid = first_queued(client)["id"]
    r = client.patch(
        f"/api/extractions/{eid}", json={"field": "invoice_date", "value": "14-03-2026"}
    )
    assert r.status_code == 422
    assert "YYYY-MM-DD" in r.json()["detail"]


def test_non_editable_field_is_refused(client):
    """The API must not let a client write to arbitrary columns."""
    eid = first_queued(client)["id"]
    r = client.patch(
        f"/api/extractions/{eid}", json={"field": "overall_confidence", "value": "1.0"}
    )
    assert r.status_code == 422


def test_clearing_a_field_stores_null(client):
    eid = first_queued(client)["id"]
    body = client.patch(
        f"/api/extractions/{eid}", json={"field": "buyer_name", "value": ""}
    ).json()
    assert body["buyer_name"] is None


# --------------------------------------------------------------- decisions

def test_approve_clears_it_from_the_queue(client):
    eid = first_queued(client)["id"]
    assert client.post(f"/api/extractions/{eid}/approve", json={"actor": "aymen"}).status_code == 200
    assert eid not in [i["id"] for i in client.get("/api/queue").json()]


def test_reject_requires_a_reason(client):
    eid = first_queued(client)["id"]
    r = client.post(f"/api/extractions/{eid}/reject", json={"actor": "a", "note": "  "})
    assert r.status_code == 422
    assert "reason" in r.json()["detail"].lower()


def test_reject_with_a_reason_records_it(client):
    eid = first_queued(client)["id"]
    body = client.post(
        f"/api/extractions/{eid}/reject", json={"actor": "aymen", "note": "wrong supplier"}
    ).json()
    assert body["status"] == "rejected"
    assert "wrong supplier" in body["history"][-1]["note"]


# ----------------------------------------------------------------- vendors

def test_vendors_expose_learned_conventions(client):
    rows = client.get("/api/vendors").json()
    assert rows
    assert {"name", "date_convention", "invoice_count"} <= set(rows[0])


# ---------------------------------------------------------------- document

def test_missing_original_explains_itself(client):
    """CLI-processed documents were never copied into storage. The error should
    say so rather than returning a bare 404."""
    detail = client.get(f"/api/extractions/{first_queued(client)['id']}").json()
    r = client.get(f"/api/documents/{detail['document_id']}/file")
    assert r.status_code == 404
    assert "re-upload" in r.json()["detail"]


# ------------------------------------------------------------------ upload

def test_upload_extracts_and_queues(client, monkeypatch):
    import src.llm

    monkeypatch.setattr(src.llm, "get_provider", lambda *a, **k: src.llm.MockProvider())
    with open("samples/invoice_usd_notax.pdf", "rb") as fh:
        r = client.post("/api/upload", files={"file": ("inv.pdf", fh, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["duplicate"] is False


def test_uploading_a_known_document_costs_no_extraction(client, monkeypatch):
    import src.llm

    monkeypatch.setattr(src.llm, "get_provider", lambda *a, **k: src.llm.MockProvider())
    for _ in range(2):
        with open("samples/invoice_ambiguous_date.pdf", "rb") as fh:
            r = client.post("/api/upload", files={"file": ("x.pdf", fh, "application/pdf")})
    assert r.json()["duplicate"] is True


def test_unsupported_file_type_is_refused(client):
    r = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
