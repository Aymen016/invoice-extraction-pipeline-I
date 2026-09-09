"""Deterministic validation.

These rules cost nothing to run and catch errors no confidence score will.
An invoice is a document that has to add up — that arithmetic is free
ground truth, so use it.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from .schemas import InvoiceExtraction

# Currency rounding means totals rarely match to the cent. Allow a small drift.
TOLERANCE = Decimal("0.02")


def validate(extraction: InvoiceExtraction) -> list[str]:
    """Returns a list of human-readable problems. Empty list means clean."""
    errors: list[str] = []
    e = extraction

    # Does subtotal + tax equal total?
    if e.subtotal is not None and e.total_amount is not None:
        expected = e.subtotal + (e.tax_amount or Decimal(0))
        drift = abs(expected - e.total_amount)
        if drift > TOLERANCE:
            errors.append(
                f"Arithmetic mismatch: subtotal {e.subtotal} + tax "
                f"{e.tax_amount or 0} = {expected}, but total says {e.total_amount} "
                f"(off by {drift})"
            )

    # Do the line items add up to the subtotal?
    line_totals = [li.line_total for li in e.line_items if li.line_total is not None]
    if line_totals and e.subtotal is not None:
        summed = sum(line_totals, Decimal(0))
        if abs(summed - e.subtotal) > TOLERANCE:
            errors.append(
                f"Line items sum to {summed} but subtotal says {e.subtotal}"
            )

    # Does each line item's own arithmetic hold?
    for i, li in enumerate(e.line_items, start=1):
        if li.quantity is not None and li.unit_price is not None and li.line_total is not None:
            expected = li.quantity * li.unit_price
            if abs(expected - li.line_total) > TOLERANCE:
                errors.append(
                    f"Line {i} ({li.description[:30]}): {li.quantity} x {li.unit_price} "
                    f"= {expected}, but line total says {li.line_total}"
                )

    # Date sanity
    if e.invoice_date and e.due_date and e.due_date < e.invoice_date:
        errors.append(f"Due date {e.due_date} precedes invoice date {e.invoice_date}")

    if e.invoice_date:
        if e.invoice_date > date.today() + timedelta(days=2):
            errors.append(f"Invoice date {e.invoice_date} is in the future")
        if e.invoice_date < date.today() - timedelta(days=365 * 10):
            errors.append(f"Invoice date {e.invoice_date} is over 10 years old — likely misparsed")

    # Value sanity
    if e.total_amount is not None and e.total_amount < 0:
        errors.append(f"Negative total: {e.total_amount}")

    if e.tax_amount is not None and e.subtotal is not None and e.subtotal > 0:
        rate = e.tax_amount / e.subtotal
        if rate > Decimal("0.5"):
            errors.append(f"Tax is {rate:.0%} of subtotal — implausible, check the mapping")

    if e.currency and len(e.currency) != 3:
        errors.append(f"Currency '{e.currency}' is not a 3-letter ISO code")

    return errors
