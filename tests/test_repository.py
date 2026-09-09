"""Phase 2 tests: persistence, deduplication, audit trail, vendor learning.

Runs against an in-memory SQLite database, so no setup and no cleanup.
"""

from datetime import date
from decimal import Decimal

import pytest

from src.db import DateConvention, Document, Extraction, Vendor, init_db, make_engine, make_session_factory
from src.llm import MockProvider
from src.pipeline import InvoicePipeline
from src.repository import CONVENTION_THRESHOLD, InvoiceRepository
from src.schemas import ReviewStatus


@pytest.fixture
def session():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    with make_session_factory(engine)() as s:
        yield s


@pytest.fixture
def repo(session):
    return InvoiceRepository(session)


CLEAN = {
    "invoice_number": "INV-2026-0412",
    "invoice_date": "2026-03-05",  # 05/03 — swappable, unlike 14/03
    "vendor_name": "Zenith Textiles (Pvt) Ltd",
    "vendor_tax_id": "4210398-7",
    "currency": "PKR",
    "subtotal": "133300.00",
    "tax_amount": "23994.00",
    "total_amount": "157294.00",
    "line_items": [
        {"description": "Cotton fabric", "quantity": 120, "unit_price": 850, "line_total": 102000},
        {"description": "Polyester lining", "quantity": 80, "unit_price": 310, "line_total": 24800},
        {"description": "Packaging", "quantity": 1, "unit_price": 6500, "line_total": 6500},
    ],
    "_field_confidence": {"invoice_number": 1.0, "total_amount": 1.0},
}


def run(canned=None, path="samples/invoice_clean_pkr.pdf"):
    pipeline = InvoicePipeline(provider=MockProvider(canned=canned or CLEAN), use_cache=False)
    return pipeline.process(path)


# ------------------------------- persistence -------------------------------

def test_save_persists_everything(repo, session):
    outcome = repo.save(run())
    assert not outcome.is_duplicate

    e = session.get(Extraction, outcome.extraction.id)
    assert e.invoice_number == "INV-2026-0412"
    assert e.total_amount == Decimal("157294.00")
    assert e.invoice_date == date(2026, 3, 5)
    assert len(e.line_items) == 3
    assert e.line_items[0].description == "Cotton fabric"
    assert e.confidences, "field confidences should be stored for the review UI"
    assert e.model and e.prompt_version, "provenance is required for Phase 4"


def test_money_survives_as_decimal_not_float(repo, session):
    """Floats corrupt currency. 0.1 + 0.2 != 0.3 and an accountant will notice."""
    outcome = repo.save(run())
    total = session.get(Extraction, outcome.extraction.id).total_amount
    assert isinstance(total, Decimal)
    assert total == Decimal("157294.00")


# ----------------------------- deduplication ------------------------------

def test_same_content_is_not_stored_twice(repo, session):
    first = repo.save(run())
    second = repo.save(run())

    assert not first.is_duplicate
    assert second.is_duplicate
    assert second.previous_extraction_id == first.extraction.id
    assert session.query(Document).count() == 1


def test_dedup_is_by_content_not_filename(repo, session, tmp_path):
    """The same invoice forwarded under a different name must still be caught —
    paying twice is the expensive failure this prevents."""
    import shutil

    copy = tmp_path / "FW_invoice_copy.pdf"
    shutil.copy("samples/invoice_clean_pkr.pdf", copy)

    repo.save(run())
    second = repo.save(run(path=str(copy)))

    assert second.is_duplicate
    assert session.query(Document).count() == 1


def test_different_documents_are_both_stored(repo, session):
    repo.save(run())
    repo.save(run(path="samples/invoice_usd_notax.pdf"))
    assert session.query(Document).count() == 2


# ------------------------------ audit trail -------------------------------

def test_extraction_is_audited_on_save(repo):
    outcome = repo.save(run())
    entries = repo.history(outcome.extraction.id)
    assert [e.action for e in entries] == ["extracted"]


def test_correction_records_before_and_after(repo):
    outcome = repo.save(run())
    repo.correct_field(
        outcome.extraction.id, "total_amount", Decimal("157294.50"),
        actor="aymen", note="checked against the PDF",
    )
    entry = repo.history(outcome.extraction.id)[-1]
    assert entry.action == "corrected"
    assert entry.actor == "aymen"
    assert entry.field == "total_amount"
    assert entry.old_value == "157294.00"
    assert entry.new_value == "157294.50"


def test_approval_is_recorded_and_status_changes(repo, session):
    outcome = repo.save(run())
    repo.approve(outcome.extraction.id, actor="aymen")
    assert session.get(Extraction, outcome.extraction.id).status == "approved"
    assert repo.history(outcome.extraction.id)[-1].action == "approved"


def test_rejection_requires_and_keeps_a_reason(repo):
    outcome = repo.save(run())
    repo.reject(outcome.extraction.id, actor="aymen", reason="wrong supplier entirely")
    entry = repo.history(outcome.extraction.id)[-1]
    assert entry.action == "rejected"
    assert "wrong supplier" in entry.note


def test_history_is_append_only(repo):
    outcome = repo.save(run())
    repo.correct_field(outcome.extraction.id, "invoice_number", "INV-999", actor="a")
    repo.correct_field(outcome.extraction.id, "buyer_name", "Someone Else", actor="b")
    repo.approve(outcome.extraction.id, actor="c")
    assert [e.action for e in repo.history(outcome.extraction.id)] == [
        "extracted", "corrected", "corrected", "approved",
    ]


# ---------------------------- vendor learning -----------------------------

def test_vendor_created_and_counted(repo, session):
    repo.save(run())
    repo.save(run(path="samples/invoice_bad_math.pdf"))
    zenith = session.query(Vendor).filter_by(name="Zenith Textiles (Pvt) Ltd").one()
    assert zenith.invoice_count == 2
    assert zenith.tax_id == "4210398-7"


def test_day_month_swap_teaches_the_convention(repo, session):
    outcome = repo.save(run())
    # Model read 05/03/2026 as 5 March; the human says it was 3 May.
    # Day and month have traded places — that is the informative signal.
    repo.correct_field(
        outcome.extraction.id, "invoice_date", date(2026, 5, 3), actor="aymen"
    )
    vendor = session.query(Vendor).filter_by(name="Zenith Textiles (Pvt) Ltd").one()
    assert vendor.date_convention == DateConvention.MONTH_FIRST
    assert vendor.convention_confirmations == 1


def test_unrelated_date_edit_teaches_nothing(repo, session):
    """Only a day/month swap reveals ordering. Any other fix is a different
    error and must not be mistaken for evidence about the vendor."""
    outcome = repo.save(run())
    repo.correct_field(outcome.extraction.id, "invoice_date", date(2025, 7, 22), actor="aymen")
    vendor = session.query(Vendor).filter_by(name="Zenith Textiles (Pvt) Ltd").one()
    assert vendor.date_convention == DateConvention.UNKNOWN


def test_one_confirmation_is_not_enough_to_trust(repo):
    """A single observation is a data point, not a pattern."""
    outcome = repo.save(run())
    repo.correct_field(outcome.extraction.id, "invoice_date", date(2026, 5, 3), actor="a")
    assert repo.known_date_convention("Zenith Textiles (Pvt) Ltd") is None


def test_repeated_confirmations_become_trusted(repo, session):
    paths = [
        "samples/invoice_clean_pkr.pdf",
        "samples/invoice_usd_notax.pdf",
        "samples/invoice_bad_math.pdf",
    ]
    assert CONVENTION_THRESHOLD <= len(paths)

    for path in paths:
        outcome = repo.save(run(path=path))
        repo.correct_field(
            outcome.extraction.id, "invoice_date", date(2026, 5, 3), actor="aymen"
        )

    vendor = session.query(Vendor).filter_by(name="Zenith Textiles (Pvt) Ltd").one()
    assert vendor.convention_confirmations >= CONVENTION_THRESHOLD
    assert repo.known_date_convention("Zenith Textiles (Pvt) Ltd") == DateConvention.MONTH_FIRST


def test_unknown_vendor_has_no_convention(repo):
    assert repo.known_date_convention("Never Seen Ltd") is None
    assert repo.known_date_convention(None) is None


# --------------------------------- reads ----------------------------------

def test_review_queue_is_worst_first(repo):
    repo.save(run(canned={**CLEAN, "total_amount": "999999.00", "line_items": []}))
    repo.save(run(canned=CLEAN, path="samples/invoice_usd_notax.pdf"))

    queue = repo.review_queue()
    confidences = [e.overall_confidence for e in queue]
    assert confidences == sorted(confidences), "lowest confidence must surface first"


def test_stats_reports_straight_through_rate(repo):
    repo.save(run())
    repo.save(run(canned={**CLEAN, "subtotal": "1.00", "line_items": []},
                  path="samples/invoice_bad_math.pdf"))
    st = repo.stats()
    assert st["documents"] == 2
    assert 0.0 <= st["straight_through_rate"] <= 1.0


def test_correcting_unknown_field_is_rejected(repo):
    outcome = repo.save(run())
    with pytest.raises(ValueError):
        repo.correct_field(outcome.extraction.id, "not_a_column", "x", actor="a")


def test_failed_extraction_can_still_be_recorded(repo):
    """A document that failed to parse should still leave a trace, or you have
    no way to find out what the system choked on."""
    result = run(canned={"line_items": []})
    result.status = ReviewStatus.NEEDS_REVIEW
    outcome = repo.save(result)
    assert outcome.extraction.id is not None


def test_unswappable_date_correction_teaches_nothing(repo, session):
    """14/03 cannot be a swap — month 14 does not exist — so correcting it
    reveals nothing about ordering. Guards against false learning."""
    outcome = repo.save(run(canned={**CLEAN, "invoice_date": "2026-03-14"}))
    repo.correct_field(outcome.extraction.id, "invoice_date", date(2026, 3, 15), actor="a")
    vendor = session.query(Vendor).filter_by(name="Zenith Textiles (Pvt) Ltd").one()
    assert vendor.date_convention == DateConvention.UNKNOWN


def test_known_hashes_enables_skipping_before_extraction(repo):
    """The expensive part is the API call. Duplicates must be identifiable
    from the hash alone, before any extraction happens."""
    from src.reader import sha256_of

    outcome = repo.save(run())
    digest = sha256_of("samples/invoice_clean_pkr.pdf")
    other = sha256_of("samples/invoice_usd_notax.pdf")

    known = repo.known_hashes([digest, other])
    assert known == {digest: outcome.extraction.id}


def test_known_hashes_handles_empty_input(repo):
    assert repo.known_hashes([]) == {}
