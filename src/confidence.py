"""Confidence scoring.

An LLM's self-reported confidence is weak evidence — models are cheerfully
certain about things they hallucinated. So we combine it with a second,
independent signal:

    GROUNDING — does the extracted value literally appear in the source text?

If the model says the total is 45,300 and "45,300" is nowhere in the document,
that is a fabrication regardless of how confident the model claims to be.
Grounding is cheap, deterministic, and catches the failure mode that matters.

    final = 0.4 * model_confidence + 0.6 * grounding

Grounding is weighted higher on purpose. Tune these weights in Phase 4 once
you have labelled data to tune against — do not trust my numbers, measure them.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Optional

from .schemas import FieldConfidence, InvoiceExtraction

MODEL_WEIGHT = 0.4
GROUNDING_WEIGHT = 0.6

# Fields that must be right for the row to be worth anything.
CRITICAL_FIELDS = {"invoice_number", "vendor_name", "total_amount", "invoice_date"}

# Dates get reformatted to ISO, so literal string matching is meaningless
# for them. We ground them on their component parts instead.
DATE_FIELDS = {"invoice_date", "due_date"}

# Currency is normalised to an ISO code, so "USD" will not appear in a document
# that prints "$". Ground it against the symbols it could have come from.
CURRENCY_ALIASES = {
    "USD": ["usd", "$", "us$", "dollar"],
    "PKR": ["pkr", "rs", "rs.", "₨", "rupee"],
    "GBP": ["gbp", "£", "pound"],
    "EUR": ["eur", "€", "euro"],
    "AED": ["aed", "dirham"],
    "INR": ["inr", "₹", "rs", "rupee"],
}

# A numeric date where both leading parts are <= 12 (e.g. 05/01/2026) cannot be
# resolved from the document alone. Grounding cannot detect a day/month swap —
# the same three numbers are present either way — so we cap confidence instead
# and let a human decide.
AMBIGUOUS_DATE_CEILING = 0.55


def score_extraction(
    extraction: InvoiceExtraction,
    source_text: str,
    model_confidences: Optional[dict[str, float]] = None,
) -> list[FieldConfidence]:
    model_confidences = model_confidences or {}
    normalised = _normalise(source_text)
    scores: list[FieldConfidence] = []

    for field in InvoiceExtraction.model_fields:
        if field == "line_items":
            continue
        value = getattr(extraction, field)
        if value is None:
            continue

        model_conf = float(model_confidences.get(field, 0.7))
        model_conf = max(0.0, min(1.0, model_conf))
        grounding = _grounding_score(field, value, normalised, source_text)

        scores.append(
            FieldConfidence(
                field=field,
                value=str(value),
                model_confidence=round(model_conf, 3),
                grounding_score=round(grounding, 3),
                final_confidence=round(
                    MODEL_WEIGHT * model_conf + GROUNDING_WEIGHT * grounding, 3
                ),
            )
        )
    return scores


def overall_confidence(scores: list[FieldConfidence]) -> float:
    """Weighted toward the critical fields. A perfect vendor_address does not
    compensate for a wrong total_amount."""
    if not scores:
        return 0.0
    total = weight_sum = 0.0
    for s in scores:
        w = 3.0 if s.field in CRITICAL_FIELDS else 1.0
        total += s.final_confidence * w
        weight_sum += w
    return round(total / weight_sum, 3)


def missing_critical_fields(extraction: InvoiceExtraction) -> list[str]:
    return sorted(f for f in CRITICAL_FIELDS if getattr(extraction, f) is None)


# --------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip everything that varies between the document and the
    model's output: whitespace, commas, currency symbols."""
    text = text.lower()
    text = re.sub(r"[,\s₨$£€]", "", text)
    return text


def _grounding_score(field: str, value: Any, normalised_source: str, raw_source: str) -> float:
    if field in DATE_FIELDS:
        return _date_grounding(value, raw_source)

    if field == "currency":
        return _currency_grounding(value, raw_source)

    if isinstance(value, Decimal):
        return _number_grounding(value, normalised_source)

    text = _normalise(str(value))
    if not text:
        return 0.0
    if text in normalised_source:
        return 1.0

    # Partial credit: multi-word values (addresses, company names) often get
    # lightly reformatted. Score by token overlap rather than all-or-nothing.
    tokens = [t for t in re.split(r"\W+", str(value).lower()) if len(t) > 2]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in normalised_source)
    return hits / len(tokens)


def _currency_grounding(value: Any, raw_source: str) -> float:
    """The model is asked to output an ISO code, so a literal match is the wrong
    test. Accept any symbol or spelling the code could legitimately derive from."""
    code = str(value).strip().upper()
    source = raw_source.lower()
    for alias in CURRENCY_ALIASES.get(code, [code.lower()]):
        if alias in source:
            return 1.0
    return 0.0


def _number_grounding(value: Decimal, normalised_source: str) -> float:
    """Try several renderings, because 1250 may appear as 1250, 1250.00 or 1,250.00
    (commas already stripped by _normalise)."""
    candidates = {
        str(value),
        f"{value:.2f}",
        f"{value:f}".rstrip("0").rstrip("."),
        str(int(value)) if value == value.to_integral_value() else str(value),
    }
    return 1.0 if any(c and c in normalised_source for c in candidates) else 0.0


def _date_grounding(value, raw_source: str) -> float:
    """A date reformatted to ISO won't string-match, so we check that the day,
    month and year each appear somewhere.

    Important limitation, handled explicitly below: this cannot detect a
    day/month swap. Given "05/01/2026" the numbers 05, 01 and 2026 are present
    whether the model read it as 5 January or 1 May. When the source date is
    genuinely ambiguous we cap the score rather than pretend to have verified it.
    """
    source = raw_source.lower()

    if _source_date_is_ambiguous(value, raw_source):
        return AMBIGUOUS_DATE_CEILING
    year = str(value.year)
    if year not in source and year[-2:] not in source:
        return 0.0

    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    month_name = month_names[value.month - 1]
    month_hit = (
        month_name in source
        or month_name[:3] in source
        or f"{value.month:02d}" in source
        or f"/{value.month}/" in source
    )
    day_hit = f"{value.day:02d}" in source or str(value.day) in source

    return 0.4 + 0.3 * month_hit + 0.3 * day_hit


def _source_date_is_ambiguous(value, raw_source: str) -> bool:
    """True when the document renders this date numerically with both the day
    and month <= 12, making day-first vs month-first undecidable.

    A written month ("14 March 2026") is never ambiguous, and a day above 12
    ("14/03/2026") resolves itself — only the genuinely undecidable case counts.
    """
    if value.day > 12 or value.month > 12:
        return False

    pattern = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
    for a, b, _year in pattern.findall(raw_source):
        pair = {int(a), int(b)}
        if pair == {value.day, value.month} and int(a) <= 12 and int(b) <= 12:
            # Both orderings produce a valid date from these same digits.
            return value.day != value.month
    return False
