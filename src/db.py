"""Database schema.

SQLite by default so the project runs with no setup; Postgres by changing
DATABASE_URL. Nothing here is SQLite-specific — no dialect-only column types,
no raw SQL — so the swap is a connection string and nothing else.

Design notes worth knowing:

* Documents and extractions are separate tables. One document can be extracted
  many times (new model, new prompt version, human correction), and you want
  the history, not an overwrite. `is_current` marks the live row.

* `sha256` is unique on documents. That is the deduplication mechanism: the
  same invoice emailed twice cannot be entered twice.

* Nothing is ever deleted. Corrections append to audit_log. For anything
  touching money, "who changed this and when" is not optional.
"""

from __future__ import annotations

import enum
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# Money: 18 digits, 2 decimal places. Never float — 0.1 + 0.2 != 0.3 and an
# accountant will notice.
Money = Numeric(18, 2)
Qty = Numeric(18, 4)


class DateConvention(str, enum.Enum):
    """How a vendor writes numeric dates. Learned from confirmed corrections."""

    UNKNOWN = "unknown"
    DAY_FIRST = "day_first"      # 05/01/2026 = 5 January
    MONTH_FIRST = "month_first"  # 05/01/2026 = 1 May


class Document(Base):
    """A physical file. Unique by content, not by filename."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(500))
    byte_size: Mapped[Optional[int]] = mapped_column(Integer)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    text_source: Mapped[Optional[str]] = mapped_column(String(20))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    extractions: Mapped[list["Extraction"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Extraction.id"
    )

    @property
    def current_extraction(self) -> Optional["Extraction"]:
        return next((e for e in self.extractions if e.is_current), None)


class Vendor(Base):
    """Accumulated knowledge about a supplier.

    This is what makes the system improve with use. The first ambiguous date
    from a vendor goes to a human; once confirmed, the convention is recorded
    and every later invoice from them resolves automatically.
    """

    __tablename__ = "vendors"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300), unique=True, index=True)
    tax_id: Mapped[Optional[str]] = mapped_column(String(50))
    default_currency: Mapped[Optional[str]] = mapped_column(String(3))

    date_convention: Mapped[DateConvention] = mapped_column(
        Enum(DateConvention), default=DateConvention.UNKNOWN
    )
    # How many human confirmations back the convention. One is a data point;
    # three is a pattern. The repository will not trust a single observation.
    convention_confirmations: Mapped[int] = mapped_column(Integer, default=0)

    invoice_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Extraction(Base):
    """One attempt at reading one document."""

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    vendor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("vendors.id"), index=True)

    # Provenance: which model and which prompt produced this. Without both,
    # the Phase 4 accuracy table cannot be reproduced.
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(20))

    status: Mapped[str] = mapped_column(String(30), index=True)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    invoice_number: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    invoice_date: Mapped[Optional[date]] = mapped_column(Date)
    due_date: Mapped[Optional[date]] = mapped_column(Date)
    vendor_name: Mapped[Optional[str]] = mapped_column(String(300))
    vendor_tax_id: Mapped[Optional[str]] = mapped_column(String(50))
    vendor_address: Mapped[Optional[str]] = mapped_column(Text)
    buyer_name: Mapped[Optional[str]] = mapped_column(String(300))
    currency: Mapped[Optional[str]] = mapped_column(String(3))
    subtotal: Mapped[Optional[Decimal]] = mapped_column(Money)
    tax_amount: Mapped[Optional[Decimal]] = mapped_column(Money)
    total_amount: Mapped[Optional[Decimal]] = mapped_column(Money)

    elapsed_seconds: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="extractions")
    vendor: Mapped[Optional["Vendor"]] = relationship()
    line_items: Mapped[list["LineItemRow"]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan", order_by="LineItemRow.position"
    )
    confidences: Mapped[list["FieldConfidenceRow"]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan"
    )
    issues: Mapped[list["ValidationIssue"]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan"
    )
    audit_entries: Mapped[list["AuditEntry"]] = relationship(
        back_populates="extraction", cascade="all, delete-orphan", order_by="AuditEntry.id"
    )


class LineItemRow(Base):
    __tablename__ = "line_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id"), index=True)
    position: Mapped[int] = mapped_column(Integer, default=0)

    description: Mapped[str] = mapped_column(Text)
    quantity: Mapped[Optional[Decimal]] = mapped_column(Qty)
    unit_price: Mapped[Optional[Decimal]] = mapped_column(Money)
    line_total: Mapped[Optional[Decimal]] = mapped_column(Money)

    extraction: Mapped["Extraction"] = relationship(back_populates="line_items")


class FieldConfidenceRow(Base):
    """Per-field scores, persisted so the review UI can highlight weak fields
    and so Phase 4 can correlate confidence against actual correctness."""

    __tablename__ = "field_confidences"
    __table_args__ = (UniqueConstraint("extraction_id", "field"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id"), index=True)

    field: Mapped[str] = mapped_column(String(60))
    value: Mapped[Optional[str]] = mapped_column(Text)
    model_confidence: Mapped[float] = mapped_column(Float)
    grounding_score: Mapped[float] = mapped_column(Float)
    final_confidence: Mapped[float] = mapped_column(Float)

    extraction: Mapped["Extraction"] = relationship(back_populates="confidences")


class ValidationIssue(Base):
    __tablename__ = "validation_issues"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id"), index=True)
    message: Mapped[str] = mapped_column(Text)

    extraction: Mapped["Extraction"] = relationship(back_populates="issues")


class AuditEntry(Base):
    """Append-only history. Every state change lands here.

    For a system that writes financial records, being able to answer "who
    approved this, when, and what did they change" is the difference between
    a demo and something a business can actually adopt.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    extraction_id: Mapped[int] = mapped_column(ForeignKey("extractions.id"), index=True)

    action: Mapped[str] = mapped_column(String(40))  # extracted | corrected | approved | rejected
    actor: Mapped[str] = mapped_column(String(100), default="system")
    field: Mapped[Optional[str]] = mapped_column(String(60))
    old_value: Mapped[Optional[str]] = mapped_column(Text)
    new_value: Mapped[Optional[str]] = mapped_column(Text)
    note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    extraction: Mapped["Extraction"] = relationship(back_populates="audit_entries")


# --------------------------------------------------------------------------

DEFAULT_URL = "sqlite:///invoices.db"


def make_engine(url: Optional[str] = None, echo: bool = False):
    import os

    url = url or os.getenv("DATABASE_URL", DEFAULT_URL)
    kwargs = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        # Needed when the FastAPI backend in Phase 3 serves requests on threads.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def make_session_factory(engine=None):
    return sessionmaker(bind=engine or make_engine(), expire_on_commit=False, future=True)


def init_db(engine=None) -> None:
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
