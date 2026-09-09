"""Repository: everything that touches the database lives behind this.

Three responsibilities that are easy to get wrong and worth isolating:

  1. DEDUPLICATION by content hash. The same invoice arrives twice — forwarded
     email, re-scanned, uploaded by two people. Filenames differ; content does
     not. Paying an invoice twice is the expensive failure this prevents.

  2. AUDIT TRAIL. Every approval and correction appends a row. Nothing is
     overwritten in place.

  3. VENDOR LEARNING. When a human resolves an ambiguous date, we record which
     convention that implied. After enough confirmations the system resolves
     the same vendor's dates on its own, and the review queue shrinks over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    AuditEntry,
    DateConvention,
    Document,
    Extraction,
    FieldConfidenceRow,
    LineItemRow,
    ValidationIssue,
    Vendor,
)
from .schemas import ExtractionResult

logger = logging.getLogger(__name__)

# One human confirmation is a data point; several are a pattern. Below this we
# record observations but keep sending the vendor's dates to review.
CONVENTION_THRESHOLD = 3


@dataclass
class SaveOutcome:
    extraction: Extraction
    is_duplicate: bool
    previous_extraction_id: Optional[int] = None


class InvoiceRepository:
    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------ pre-check

    def known_hashes(self, hashes: list[str]) -> dict[str, int]:
        """Which of these content hashes are already stored, and as which
        extraction. Lets the caller skip extraction entirely for documents it
        has already seen, instead of paying for an API call to rediscover it.
        """
        if not hashes:
            return {}
        rows = self.session.scalars(
            select(Document).where(Document.sha256.in_(hashes))
        )
        out = {}
        for doc in rows:
            current = doc.current_extraction
            out[doc.sha256] = current.id if current else 0
        return out

    # ---------------------------------------------------------------- save

    def save(self, result: ExtractionResult, actor: str = "system") -> SaveOutcome:
        """Persist an extraction, deduplicating on document content hash."""
        from pathlib import Path

        existing_doc = self.session.scalar(
            select(Document).where(Document.sha256 == result.source_sha256)
        )

        if existing_doc is not None:
            current = existing_doc.current_extraction
            logger.info(
                "Duplicate content: %s matches document #%s (%s)",
                Path(result.source_file).name, existing_doc.id, existing_doc.filename,
            )
            return SaveOutcome(
                extraction=current,
                is_duplicate=True,
                previous_extraction_id=current.id if current else None,
            )

        doc = Document(
            sha256=result.source_sha256,
            filename=Path(result.source_file).name,
            text_source=result.text_source,
        )
        try:
            doc.byte_size = Path(result.source_file).stat().st_size
        except OSError:
            pass
        self.session.add(doc)
        self.session.flush()

        extraction = self._build_extraction(doc, result)
        self.session.add(extraction)
        self.session.flush()

        self.session.add(
            AuditEntry(
                extraction_id=extraction.id,
                action="extracted",
                actor=actor,
                note=f"{result.model} / prompt {extraction.prompt_version} / "
                     f"{result.status.value} @ {result.overall_confidence:.2f}",
            )
        )
        self.session.commit()
        return SaveOutcome(extraction=extraction, is_duplicate=False)

    def _build_extraction(self, doc: Document, result: ExtractionResult) -> Extraction:
        from .prompts import PROMPT_VERSION

        e = result.extraction
        vendor = self._upsert_vendor(e.vendor_name, e.vendor_tax_id, e.currency)

        extraction = Extraction(
            document_id=doc.id,
            vendor_id=vendor.id if vendor else None,
            model=result.model,
            prompt_version=PROMPT_VERSION,
            status=result.status.value,
            overall_confidence=result.overall_confidence,
            elapsed_seconds=result.elapsed_seconds,
            invoice_number=e.invoice_number,
            invoice_date=e.invoice_date,
            due_date=e.due_date,
            vendor_name=e.vendor_name,
            vendor_tax_id=e.vendor_tax_id,
            vendor_address=e.vendor_address,
            buyer_name=e.buyer_name,
            currency=e.currency,
            subtotal=e.subtotal,
            tax_amount=e.tax_amount,
            total_amount=e.total_amount,
        )
        extraction.line_items = [
            LineItemRow(
                position=i,
                description=li.description,
                quantity=li.quantity,
                unit_price=li.unit_price,
                line_total=li.line_total,
            )
            for i, li in enumerate(e.line_items)
        ]
        extraction.confidences = [
            FieldConfidenceRow(
                field=c.field,
                value=c.value,
                model_confidence=c.model_confidence,
                grounding_score=c.grounding_score,
                final_confidence=c.final_confidence,
            )
            for c in result.confidences
        ]
        extraction.issues = [ValidationIssue(message=m) for m in result.validation_errors]
        return extraction

    def _upsert_vendor(
        self, name: Optional[str], tax_id: Optional[str], currency: Optional[str]
    ) -> Optional[Vendor]:
        if not name:
            return None
        clean = name.strip()
        vendor = self.session.scalar(select(Vendor).where(Vendor.name == clean))
        if vendor is None:
            vendor = Vendor(name=clean, tax_id=tax_id, default_currency=currency)
            self.session.add(vendor)
            self.session.flush()
        vendor.invoice_count += 1
        if tax_id and not vendor.tax_id:
            vendor.tax_id = tax_id
        if currency and not vendor.default_currency:
            vendor.default_currency = currency
        return vendor

    # ------------------------------------------------------------- review

    def correct_field(
        self, extraction_id: int, field: str, new_value, actor: str, note: str = ""
    ) -> Extraction:
        """Apply a human correction and record it."""
        extraction = self.session.get(Extraction, extraction_id)
        if extraction is None:
            raise ValueError(f"No extraction #{extraction_id}")
        if not hasattr(extraction, field):
            raise ValueError(f"Not a correctable field: {field}")

        old_value = getattr(extraction, field)
        setattr(extraction, field, new_value)

        self.session.add(
            AuditEntry(
                extraction_id=extraction_id,
                action="corrected",
                actor=actor,
                field=field,
                old_value=str(old_value) if old_value is not None else None,
                new_value=str(new_value) if new_value is not None else None,
                note=note or None,
            )
        )

        # A corrected date is the signal that teaches us the vendor's convention.
        if field in ("invoice_date", "due_date") and isinstance(new_value, date):
            self._learn_date_convention(extraction, old_value, new_value)

        self.session.commit()
        return extraction

    def approve(self, extraction_id: int, actor: str, note: str = "") -> Extraction:
        extraction = self.session.get(Extraction, extraction_id)
        if extraction is None:
            raise ValueError(f"No extraction #{extraction_id}")
        extraction.status = "approved"
        self.session.add(
            AuditEntry(
                extraction_id=extraction_id, action="approved", actor=actor, note=note or None
            )
        )
        self.session.commit()
        return extraction

    def reject(self, extraction_id: int, actor: str, reason: str) -> Extraction:
        extraction = self.session.get(Extraction, extraction_id)
        if extraction is None:
            raise ValueError(f"No extraction #{extraction_id}")
        extraction.status = "rejected"
        self.session.add(
            AuditEntry(
                extraction_id=extraction_id, action="rejected", actor=actor, note=reason
            )
        )
        self.session.commit()
        return extraction

    # ---------------------------------------------------- vendor learning

    def _learn_date_convention(
        self, extraction: Extraction, old_value, new_value: date
    ) -> None:
        """If a human swapped day and month, that reveals how this vendor writes
        dates. Record it; once enough confirmations agree, trust it."""
        if extraction.vendor_id is None or not isinstance(old_value, date):
            return
        # Only a day/month swap is informative. Any other edit is a different
        # kind of error and tells us nothing about ordering.
        if not (old_value.day == new_value.month and old_value.month == new_value.day):
            return
        if old_value == new_value:
            return

        vendor = self.session.get(Vendor, extraction.vendor_id)
        if vendor is None:
            return

        # The model was told to prefer day-first. A correction away from what it
        # produced means this vendor writes month-first.
        implied = DateConvention.MONTH_FIRST

        if vendor.date_convention == implied:
            vendor.convention_confirmations += 1
        else:
            vendor.date_convention = implied
            vendor.convention_confirmations = 1

        logger.info(
            "Vendor '%s' date convention: %s (%d confirmation(s))",
            vendor.name, implied.value, vendor.convention_confirmations,
        )

    def known_date_convention(self, vendor_name: Optional[str]) -> Optional[DateConvention]:
        """What we can rely on for this vendor, or None if not yet established."""
        if not vendor_name:
            return None
        vendor = self.session.scalar(select(Vendor).where(Vendor.name == vendor_name.strip()))
        if vendor is None or vendor.date_convention == DateConvention.UNKNOWN:
            return None
        if vendor.convention_confirmations < CONVENTION_THRESHOLD:
            return None
        return vendor.date_convention

    # -------------------------------------------------------------- reads

    def review_queue(self, limit: int = 50) -> list[Extraction]:
        """Lowest confidence first — the worst extractions need eyes soonest."""
        return list(
            self.session.scalars(
                select(Extraction)
                .where(Extraction.status == "needs_review", Extraction.is_current.is_(True))
                .order_by(Extraction.overall_confidence)
                .limit(limit)
            )
        )

    def history(self, extraction_id: int) -> list[AuditEntry]:
        return list(
            self.session.scalars(
                select(AuditEntry)
                .where(AuditEntry.extraction_id == extraction_id)
                .order_by(AuditEntry.id)
            )
        )

    def stats(self) -> dict:
        from sqlalchemy import func

        rows = self.session.execute(
            select(Extraction.status, func.count(Extraction.id))
            .where(Extraction.is_current.is_(True))
            .group_by(Extraction.status)
        ).all()
        by_status = {status: count for status, count in rows}
        total = sum(by_status.values())
        auto = by_status.get("auto_approved", 0) + by_status.get("approved", 0)
        return {
            "documents": self.session.scalar(select(func.count(Document.id))) or 0,
            "vendors": self.session.scalar(select(func.count(Vendor.id))) or 0,
            "by_status": by_status,
            "total": total,
            "straight_through_rate": round(auto / total, 3) if total else 0.0,
        }
