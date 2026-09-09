"""Extraction schemas.

These Pydantic models are the contract between the LLM and everything
downstream. The LLM is asked to emit JSON matching InvoiceExtraction;
if it does not validate, we retry. Nothing untyped reaches the database.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ReviewStatus(str, Enum):
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class LineItem(BaseModel):
    """A single row from the invoice's line-item table."""

    description: str = Field(description="Item or service description")
    quantity: Optional[Decimal] = Field(default=None, description="Units ordered")
    unit_price: Optional[Decimal] = Field(default=None, description="Price per unit")
    line_total: Optional[Decimal] = Field(default=None, description="quantity * unit_price")

    @field_validator("quantity", "unit_price", "line_total", mode="before")
    @classmethod
    def _clean_number(cls, v):
        return _parse_money(v)


class InvoiceExtraction(BaseModel):
    """The fields we pull from every invoice or purchase order.

    Every field is Optional. A missing field is a real, expected outcome —
    it is not an error. Forcing the model to invent a value is far worse
    than recording a null and flagging it for review.
    """

    invoice_number: Optional[str] = None
    invoice_date: Optional[date] = None
    due_date: Optional[date] = None

    vendor_name: Optional[str] = None
    vendor_tax_id: Optional[str] = Field(
        default=None, description="NTN, GST, VAT or equivalent registration number"
    )
    vendor_address: Optional[str] = None

    buyer_name: Optional[str] = None

    currency: Optional[str] = Field(default=None, description="ISO 4217 code, e.g. PKR, USD")
    subtotal: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None

    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("subtotal", "tax_amount", "total_amount", mode="before")
    @classmethod
    def _clean_number(cls, v):
        return _parse_money(v)

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def _clean_date(cls, v):
        return _parse_date(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _clean_currency(cls, v):
        if not v:
            return None
        code = str(v).strip().upper()
        symbols = {"$": "USD", "£": "GBP", "€": "EUR", "₨": "PKR", "RS": "PKR", "RS.": "PKR"}
        return symbols.get(code, code[:3] if len(code) >= 3 else code)


class FieldConfidence(BaseModel):
    """Per-field trust score, 0.0 to 1.0."""

    field: str
    value: Optional[str] = None
    model_confidence: float = Field(ge=0.0, le=1.0)
    grounding_score: float = Field(
        ge=0.0, le=1.0, description="Did this value literally appear in the source text?"
    )
    final_confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Everything Phase 1 produces for one document."""

    source_file: str
    source_sha256: str
    extraction: InvoiceExtraction
    confidences: list[FieldConfidence] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    status: ReviewStatus = ReviewStatus.NEEDS_REVIEW
    overall_confidence: float = 0.0
    text_source: str = Field(default="", description="'pdf_text' or 'ocr'")
    model: str = ""
    elapsed_seconds: float = 0.0
    cached: bool = False

    def confidence_for(self, field: str) -> Optional[FieldConfidence]:
        return next((c for c in self.confidences if c.field == field), None)


# --------------------------------------------------------------------------
# Coercion helpers. LLMs return "1,250.00", "Rs 1250", "$1,250" and "1250".
# We normalise all of it before Pydantic sees it.
# --------------------------------------------------------------------------

def _parse_money(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float, Decimal)):
        return Decimal(str(v))
    s = str(v).strip()
    for junk in ["Rs.", "Rs", "PKR", "USD", "$", "£", "€", "₨", ",", " "]:
        s = s.replace(junk, "")
    if s.startswith("(") and s.endswith(")"):  # accounting negatives
        s = "-" + s[1:-1]
    if not s or s in {"-", "."}:
        return None
    try:
        return Decimal(s)
    except Exception:
        return None


def _parse_date(v):
    if v is None or v == "":
        return None
    if isinstance(v, date):
        return v
    s = str(v).strip()
    from datetime import datetime

    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
        "%Y/%m/%d", "%d.%m.%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
