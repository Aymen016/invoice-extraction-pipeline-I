"""Tests that run with no API key and no network.

This is the point of MockProvider: CI can exercise read -> prompt -> parse
-> validate -> score on every push, for free.
"""

import json
from datetime import date
from decimal import Decimal

import pytest

from src.confidence import overall_confidence, score_extraction
from src.llm import MockProvider
from src.pipeline import InvoicePipeline
from src.schemas import InvoiceExtraction, ReviewStatus
from src.validation import validate


# --------------------------- coercion ---------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1,250.00", Decimal("1250.00")),
    ("PKR 133,300.00", Decimal("133300.00")),
    ("$6,900", Decimal("6900")),
    ("(450.00)", Decimal("-450.00")),   # accounting negative
    ("", None),
    ("n/a", None),
    (None, None),
])
def test_money_coercion(raw, expected):
    inv = InvoiceExtraction(total_amount=raw)
    assert inv.total_amount == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026-03-14", date(2026, 3, 14)),
    ("14/03/2026", date(2026, 3, 14)),
    ("14 March 2026", date(2026, 3, 14)),
    ("March 14, 2026", date(2026, 3, 14)),
    ("not a date", None),
])
def test_date_coercion(raw, expected):
    assert InvoiceExtraction(invoice_date=raw).invoice_date == expected


def test_currency_symbol_mapping():
    assert InvoiceExtraction(currency="$").currency == "USD"
    assert InvoiceExtraction(currency="rs").currency == "PKR"
    assert InvoiceExtraction(currency="pkr").currency == "PKR"


# --------------------------- validation ---------------------------

def test_arithmetic_mismatch_is_caught():
    inv = InvoiceExtraction(
        subtotal="100500.00", tax_amount="18090.00", total_amount="121000.00"
    )
    errors = validate(inv)
    assert any("Arithmetic mismatch" in e for e in errors)


def test_correct_arithmetic_passes():
    inv = InvoiceExtraction(
        subtotal="133300.00", tax_amount="23994.00", total_amount="157294.00"
    )
    assert validate(inv) == []


def test_rounding_tolerance():
    inv = InvoiceExtraction(subtotal="100.00", tax_amount="18.00", total_amount="118.01")
    assert validate(inv) == []  # 1 paisa drift is fine


def test_due_before_invoice_date():
    inv = InvoiceExtraction(invoice_date="2026-03-14", due_date="2026-02-01")
    assert any("precedes" in e for e in validate(inv))


def test_implausible_tax_rate():
    inv = InvoiceExtraction(subtotal="1000", tax_amount="900", total_amount="1900")
    assert any("implausible" in e for e in validate(inv))


def test_line_items_must_sum_to_subtotal():
    inv = InvoiceExtraction(
        subtotal="1000",
        line_items=[
            {"description": "A", "quantity": 2, "unit_price": 100, "line_total": 200},
            {"description": "B", "quantity": 1, "unit_price": 300, "line_total": 300},
        ],
    )
    assert any("Line items sum" in e for e in validate(inv))


# --------------------------- confidence ---------------------------

SOURCE = """
Invoice No: INV-2026-0412
Invoice Date: 14/03/2026
From: Zenith Textiles (Pvt) Ltd
Grand Total: PKR 157,294.00
"""


def test_grounded_value_scores_high():
    inv = InvoiceExtraction(invoice_number="INV-2026-0412")
    score = score_extraction(inv, SOURCE, {"invoice_number": 1.0})[0]
    assert score.grounding_score == 1.0
    assert score.final_confidence == 1.0


def test_hallucinated_value_scores_low_despite_model_confidence():
    """The whole reason grounding exists: the model claims certainty about
    a number that is nowhere in the document."""
    inv = InvoiceExtraction(total_amount="999999.00")
    score = score_extraction(inv, SOURCE, {"total_amount": 1.0})[0]
    assert score.grounding_score == 0.0
    assert score.final_confidence <= 0.4


def test_number_grounding_ignores_comma_formatting():
    inv = InvoiceExtraction(total_amount="157294.00")
    score = score_extraction(inv, SOURCE, {})[0]
    assert score.grounding_score == 1.0


def test_critical_fields_dominate_overall_score():
    inv = InvoiceExtraction(vendor_address="somewhere unfindable", invoice_number="INV-2026-0412")
    scores = score_extraction(inv, SOURCE, {})
    # invoice_number is grounded and weighted 3x; a bad address shouldn't sink it
    assert overall_confidence(scores) > 0.5


# --------------------------- pipeline ---------------------------

def test_pipeline_end_to_end_offline(tmp_path):
    canned = {
        "invoice_number": "INV-2026-0412",
        "invoice_date": "2026-03-14",
        "vendor_name": "Zenith Textiles (Pvt) Ltd",
        "currency": "PKR",
        "subtotal": "133300.00",
        "tax_amount": "23994.00",
        "total_amount": "157294.00",
        "line_items": [],
        "_field_confidence": {"invoice_number": 1.0, "total_amount": 1.0},
    }
    pipeline = InvoicePipeline(provider=MockProvider(canned=canned), use_cache=False)
    result = pipeline.process("samples/invoice_clean_pkr.pdf")

    assert result.status == ReviewStatus.AUTO_APPROVED
    assert result.extraction.total_amount == Decimal("157294.00")
    assert result.validation_errors == []
    assert result.overall_confidence > 0.85
    assert result.text_source == "pdf_text"


def test_bad_arithmetic_routes_to_review():
    canned = {
        "invoice_number": "PO-77401",
        "invoice_date": "2026-01-05",
        "vendor_name": "Highland Components Ltd",
        "subtotal": "100500.00",
        "tax_amount": "18090.00",
        "total_amount": "121000.00",
        "line_items": [],
        "_field_confidence": {},
    }
    pipeline = InvoicePipeline(provider=MockProvider(canned=canned), use_cache=False)
    result = pipeline.process("samples/invoice_bad_math.pdf")

    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert any("Arithmetic mismatch" in e for e in result.validation_errors)


def test_malformed_llm_output_fails_gracefully():
    """A model that returns garbage should produce a FAILED result,
    not an exception that kills the batch."""
    class Broken(MockProvider):
        def _complete(self, prompt, temperature):
            return "I'm sorry, I cannot help with that."

    pipeline = InvoicePipeline(provider=Broken(), use_cache=False)
    results = pipeline.process_many(["samples/invoice_clean_pkr.pdf"])
    assert results[0].status == ReviewStatus.FAILED


def test_ocr_path_used_for_images():
    pipeline = InvoicePipeline(provider=MockProvider(canned={"line_items": []}), use_cache=False)
    result = pipeline.process("samples/invoice_scanned.jpg")
    assert result.text_source == "image_ocr"


def test_cache_prevents_second_call():
    calls = {"n": 0}

    class Counting(MockProvider):
        def _complete(self, prompt, temperature):
            calls["n"] += 1
            return '{"invoice_number": "X", "line_items": []}'

    provider = Counting()
    pipeline = InvoicePipeline(provider=provider, use_cache=True)
    pipeline.process("samples/invoice_clean_pkr.pdf")
    first = calls["n"]
    result = pipeline.process("samples/invoice_clean_pkr.pdf")
    assert calls["n"] == first  # no second API call
    assert result.cached is True


# --------------------------- environment ---------------------------

def test_missing_tesseract_gives_actionable_message(monkeypatch):
    """A missing system binary should produce install instructions,
    not a wrapper library's stack trace."""
    import shutil

    from src.reader import MissingOCREngine, _require_tesseract

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(MissingOCREngine) as exc:
        _require_tesseract()
    assert "apt install tesseract-ocr" in str(exc.value)


def test_fatal_error_aborts_batch_instead_of_repeating():
    """A bad model name fails identically on every document. The batch must
    stop at the first one, not emit N copies of the same traceback."""
    from src.llm import FatalLLMError

    attempts = {"n": 0}

    class BadConfig(MockProvider):
        def _complete(self, prompt, temperature):
            attempts["n"] += 1
            raise FatalLLMError("Model 'nonexistent-model' is not available")

    pipeline = InvoicePipeline(provider=BadConfig(), use_cache=False)
    with pytest.raises(FatalLLMError):
        pipeline.process_many(["samples/invoice_clean_pkr.pdf"] * 5)
    assert attempts["n"] == 1  # stopped after the first, not all five


def test_fatal_errors_are_not_retried():
    """Exponential backoff is for rate limits. Config errors must not retry."""
    from src.llm import FatalLLMError

    attempts = {"n": 0}

    class BadConfig(MockProvider):
        def _complete(self, prompt, temperature):
            attempts["n"] += 1
            raise FatalLLMError("bad key")

    with pytest.raises(FatalLLMError):
        BadConfig().complete("prompt", use_cache=False)
    assert attempts["n"] == 1


def test_rate_limits_are_retried():
    """The opposite case: transient errors should back off and retry."""
    from src.llm import LLMError, RateLimited

    attempts = {"n": 0}

    class Flaky(MockProvider):
        def _complete(self, prompt, temperature):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimited("429")
            return '{"invoice_number": "OK", "line_items": []}'

    import src.llm as llm_mod
    original = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: None  # don't actually wait in tests
    try:
        text, _ = Flaky().complete("prompt", use_cache=False)
    finally:
        llm_mod.time.sleep = original
    assert attempts["n"] == 3
    assert "OK" in text


def test_thinking_config_dropped_on_generic_400():
    """Google rejects thinkingConfig with a generic 'invalid argument' message
    that never names the field. The retry must not depend on parsing it."""
    from src.llm import GeminiProvider

    calls = []

    class FakeResponse:
        def __init__(self, status, text):
            self.status_code = status
            self.text = text

        def json(self):
            return json.loads(self.text)

    provider = GeminiProvider(model="test-model", api_key="fake")

    import src.llm as llm_mod

    def fake_post(url, **kwargs):
        config = kwargs["json"]["generationConfig"]
        calls.append("thinkingConfig" in config)
        if "thinkingConfig" in config:
            return FakeResponse(400, '{"error":{"message":"Request contains an invalid argument."}}')
        return FakeResponse(200, json.dumps(
            {"candidates": [{"content": {"parts": [{"text": '{"invoice_number":"X"}'}]}}]}
        ))

    original = llm_mod.requests.post
    llm_mod.requests.post = fake_post
    try:
        result = provider._complete("prompt", 0.0)
    finally:
        llm_mod.requests.post = original

    assert calls == [True, False]      # tried with, retried without
    assert "invoice_number" in result
    assert provider._thinking_supported is False


# ----------------- grounding limits found by real model output -----------------

def test_currency_grounded_against_symbol_not_iso_code():
    """The model is told to emit ISO codes, so 'USD' never literally appears in
    a document printing '$'. Grounding must not punish correct normalisation."""
    src_text = "Invoice SW-88213\nSubtotal: $6,900.00\nTotal Due: $6,900.00"
    inv = InvoiceExtraction(currency="USD")
    score = score_extraction(inv, src_text, {})[0]
    assert score.grounding_score == 1.0


def test_currency_grounded_for_rupee_variants():
    src_text = "Subtotal: Rs 92,500.00\nSales Tax: Rs 16,650.00"
    inv = InvoiceExtraction(currency="PKR")
    assert score_extraction(inv, src_text, {})[0].grounding_score == 1.0


def test_wrong_currency_still_scores_zero():
    src_text = "Total: $6,900.00"
    inv = InvoiceExtraction(currency="JPY")
    assert score_extraction(inv, src_text, {})[0].grounding_score == 0.0


def test_ambiguous_numeric_date_is_capped():
    """05/01/2026 could be 5 Jan or 1 May. Grounding cannot tell — the same
    digits are present either way — so it must not report high confidence."""
    from src.confidence import AMBIGUOUS_DATE_CEILING

    src_text = "Purchase Order PO-77401\nDate: 05/01/2026"
    for parsed in ("2026-05-01", "2026-01-05"):
        inv = InvoiceExtraction(invoice_date=parsed)
        score = score_extraction(inv, src_text, {"invoice_date": 1.0})[0]
        assert score.grounding_score <= AMBIGUOUS_DATE_CEILING, parsed


def test_unambiguous_numeric_date_still_scores_high():
    """14/03/2026 has a day above 12, so the ordering resolves itself."""
    src_text = "Invoice Date: 14/03/2026"
    inv = InvoiceExtraction(invoice_date="2026-03-14")
    assert score_extraction(inv, src_text, {})[0].grounding_score > 0.9


def test_written_month_is_never_ambiguous():
    src_text = "Invoice Date: 5 January 2026"
    inv = InvoiceExtraction(invoice_date="2026-01-05")
    assert score_extraction(inv, src_text, {})[0].grounding_score > 0.9


def test_same_day_and_month_is_not_ambiguous():
    """03/03/2026 reads identically either way, so there is nothing to flag."""
    src_text = "Date: 03/03/2026"
    inv = InvoiceExtraction(invoice_date="2026-03-03")
    assert score_extraction(inv, src_text, {})[0].grounding_score > 0.9


def test_ambiguous_date_routes_to_review_with_explanation():
    canned = {
        "invoice_number": "PO-77401",
        "invoice_date": "2026-05-01",
        "vendor_name": "Highland Components Ltd",
        "total_amount": "121000.00",
        "line_items": [],
        "_field_confidence": {},
    }
    pipeline = InvoicePipeline(provider=MockProvider(canned=canned), use_cache=False)
    result = pipeline.process("samples/invoice_bad_math.pdf")
    assert result.status == ReviewStatus.NEEDS_REVIEW
    assert any("ambiguous numeric date" in e for e in result.validation_errors)
