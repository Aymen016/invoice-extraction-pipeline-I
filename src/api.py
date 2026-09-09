"""FastAPI backend for the review queue.

The API's job is narrow: let a human see what the extractor was unsure about,
fix it, and sign off. Every mutating endpoint records who did it — the audit
trail from Phase 2 is the point, not an afterthought.

    uvicorn src.api:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Document, Extraction, Vendor, init_db, make_engine, make_session_factory
from .repository import InvoiceRepository
from .schemas import ReviewStatus

logger = logging.getLogger(__name__)

# Where uploaded documents are kept so the UI can show the original alongside
# the extracted fields. Reviewing figures without the source is guesswork.
STORAGE = Path(os.getenv("DOCUMENT_STORAGE", "storage"))

engine = make_engine()
SessionFactory = make_session_factory(engine)

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(engine)
    STORAGE.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Invoice review",
    description="Human review queue for extracted invoice data.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Vite dev server. Tighten this for any real deployment.
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session():
    with SessionFactory() as session:
        yield session


def get_repo(session: Session = Depends(get_session)) -> InvoiceRepository:
    return InvoiceRepository(session)


# ------------------------------------------------------------------ schemas

class LineItemOut(BaseModel):
    description: str
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    line_total: Optional[Decimal] = None


class ConfidenceOut(BaseModel):
    field: str
    value: Optional[str] = None
    model_confidence: float
    grounding_score: float
    final_confidence: float


class AuditOut(BaseModel):
    action: str
    actor: str
    field: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    note: Optional[str] = None
    created_at: str


class QueueItem(BaseModel):
    """Just enough to render the queue list — the detail view fetches the rest."""

    id: int
    filename: str
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    total_amount: Optional[Decimal] = None
    currency: Optional[str] = None
    overall_confidence: float
    issue_count: int
    status: str


class ExtractionDetail(BaseModel):
    id: int
    document_id: int
    filename: str
    status: str
    overall_confidence: float
    model: str
    prompt_version: str
    text_source: Optional[str] = None

    invoice_number: Optional[str] = None
    invoice_date: Optional[date] = None
    due_date: Optional[date] = None
    vendor_name: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    vendor_address: Optional[str] = None
    buyer_name: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None

    line_items: list[LineItemOut] = Field(default_factory=list)
    confidences: list[ConfidenceOut] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    history: list[AuditOut] = Field(default_factory=list)


class CorrectionIn(BaseModel):
    field: str
    value: Optional[str] = None
    actor: str = "reviewer"
    note: str = ""


class DecisionIn(BaseModel):
    actor: str = "reviewer"
    note: str = ""


# --------------------------------------------------------------- serialising

EDITABLE_FIELDS = {
    "invoice_number": str,
    "invoice_date": date,
    "due_date": date,
    "vendor_name": str,
    "vendor_tax_id": str,
    "vendor_address": str,
    "buyer_name": str,
    "currency": str,
    "subtotal": Decimal,
    "tax_amount": Decimal,
    "total_amount": Decimal,
}


def _coerce(field: str, raw: Optional[str]) -> Any:
    """Turn a form string into the column's real type.

    Rejecting bad input here, with a message naming the field, is far kinder
    than letting SQLAlchemy raise something opaque three layers down.
    """
    if raw is None or raw == "":
        return None
    kind = EDITABLE_FIELDS[field]
    if kind is str:
        return raw.strip()
    if kind is Decimal:
        try:
            return Decimal(raw.replace(",", "").strip())
        except InvalidOperation:
            raise HTTPException(422, f"'{raw}' is not a number ({field})")
    if kind is date:
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            raise HTTPException(422, f"Dates must be YYYY-MM-DD, got '{raw}' ({field})")
    return raw


def _detail(extraction: Extraction, repo: InvoiceRepository) -> ExtractionDetail:
    doc = extraction.document
    return ExtractionDetail(
        id=extraction.id,
        document_id=doc.id,
        filename=doc.filename,
        status=extraction.status,
        overall_confidence=extraction.overall_confidence,
        model=extraction.model,
        prompt_version=extraction.prompt_version,
        text_source=doc.text_source,
        invoice_number=extraction.invoice_number,
        invoice_date=extraction.invoice_date,
        due_date=extraction.due_date,
        vendor_name=extraction.vendor_name,
        vendor_tax_id=extraction.vendor_tax_id,
        vendor_address=extraction.vendor_address,
        buyer_name=extraction.buyer_name,
        currency=extraction.currency,
        subtotal=extraction.subtotal,
        tax_amount=extraction.tax_amount,
        total_amount=extraction.total_amount,
        line_items=[
            LineItemOut(
                description=li.description,
                quantity=li.quantity,
                unit_price=li.unit_price,
                line_total=li.line_total,
            )
            for li in extraction.line_items
        ],
        confidences=[
            ConfidenceOut(
                field=c.field,
                value=c.value,
                model_confidence=c.model_confidence,
                grounding_score=c.grounding_score,
                final_confidence=c.final_confidence,
            )
            for c in extraction.confidences
        ],
        issues=[i.message for i in extraction.issues],
        history=[
            AuditOut(
                action=a.action,
                actor=a.actor,
                field=a.field,
                old_value=a.old_value,
                new_value=a.new_value,
                note=a.note,
                created_at=a.created_at.isoformat(),
            )
            for a in repo.history(extraction.id)
        ],
    )


# ------------------------------------------------------------------- routes

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/stats")
def stats(repo: InvoiceRepository = Depends(get_repo)) -> dict:
    return repo.stats()


@app.get("/api/queue", response_model=list[QueueItem])
def queue(
    limit: int = Query(50, ge=1, le=200),
    repo: InvoiceRepository = Depends(get_repo),
) -> list[QueueItem]:
    """Worst confidence first — the reviewer's time goes where it's needed."""
    return [
        QueueItem(
            id=e.id,
            filename=e.document.filename,
            vendor_name=e.vendor_name,
            invoice_number=e.invoice_number,
            total_amount=e.total_amount,
            currency=e.currency,
            overall_confidence=e.overall_confidence,
            issue_count=len(e.issues),
            status=e.status,
        )
        for e in repo.review_queue(limit=limit)
    ]


@app.get("/api/extractions/{extraction_id}", response_model=ExtractionDetail)
def get_extraction(
    extraction_id: int,
    session: Session = Depends(get_session),
    repo: InvoiceRepository = Depends(get_repo),
) -> ExtractionDetail:
    extraction = session.get(Extraction, extraction_id)
    if extraction is None:
        raise HTTPException(404, f"No extraction #{extraction_id}")
    return _detail(extraction, repo)


@app.patch("/api/extractions/{extraction_id}", response_model=ExtractionDetail)
def correct(
    extraction_id: int,
    body: CorrectionIn,
    session: Session = Depends(get_session),
    repo: InvoiceRepository = Depends(get_repo),
) -> ExtractionDetail:
    if body.field not in EDITABLE_FIELDS:
        raise HTTPException(
            422, f"'{body.field}' is not editable. Allowed: {sorted(EDITABLE_FIELDS)}"
        )
    if session.get(Extraction, extraction_id) is None:
        raise HTTPException(404, f"No extraction #{extraction_id}")

    value = _coerce(body.field, body.value)
    extraction = repo.correct_field(
        extraction_id, body.field, value, actor=body.actor, note=body.note
    )
    return _detail(extraction, repo)


@app.post("/api/extractions/{extraction_id}/approve", response_model=ExtractionDetail)
def approve(
    extraction_id: int,
    body: DecisionIn,
    session: Session = Depends(get_session),
    repo: InvoiceRepository = Depends(get_repo),
) -> ExtractionDetail:
    if session.get(Extraction, extraction_id) is None:
        raise HTTPException(404, f"No extraction #{extraction_id}")
    return _detail(repo.approve(extraction_id, actor=body.actor, note=body.note), repo)


@app.post("/api/extractions/{extraction_id}/reject", response_model=ExtractionDetail)
def reject(
    extraction_id: int,
    body: DecisionIn,
    session: Session = Depends(get_session),
    repo: InvoiceRepository = Depends(get_repo),
) -> ExtractionDetail:
    if session.get(Extraction, extraction_id) is None:
        raise HTTPException(404, f"No extraction #{extraction_id}")
    if not body.note.strip():
        raise HTTPException(422, "Rejecting requires a reason.")
    return _detail(repo.reject(extraction_id, actor=body.actor, reason=body.note), repo)


@app.get("/api/documents/{document_id}/file")
def document_file(document_id: int, session: Session = Depends(get_session)):
    """Serve the original so the reviewer can check figures against the page."""
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, f"No document #{document_id}")
    path = STORAGE / f"{doc.sha256}{Path(doc.filename).suffix}"
    if not path.exists():
        raise HTTPException(
            404,
            f"The file for '{doc.filename}' is not in storage. Documents processed "
            f"through the CLI before the API existed were not copied here — "
            f"re-upload it to view the original.",
        )
    return FileResponse(path, filename=doc.filename)


@app.get("/api/vendors")
def vendors(session: Session = Depends(get_session)) -> list[dict]:
    """Shows what the system has learned. The date convention column is the
    interesting one — it fills in as reviewers correct swapped dates."""
    rows = session.scalars(select(Vendor).order_by(Vendor.invoice_count.desc()))
    return [
        {
            "name": v.name,
            "tax_id": v.tax_id,
            "default_currency": v.default_currency,
            "invoice_count": v.invoice_count,
            "date_convention": v.date_convention.value,
            "convention_confirmations": v.convention_confirmations,
        }
        for v in rows
    ]


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    repo: InvoiceRepository = Depends(get_repo),
) -> dict:
    """Upload a document, extract it, and queue it for review."""
    from .llm import get_provider
    from .pipeline import InvoicePipeline
    from .reader import sha256_of

    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}:
        raise HTTPException(415, f"Cannot read '{suffix}' files. Upload a PDF or an image.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        digest = sha256_of(tmp_path)

        # Check before extracting. A hash is free; an API call is not.
        known = repo.known_hashes([digest])
        if digest in known:
            return {
                "duplicate": True,
                "extraction_id": known[digest],
                "message": "This document is already in the system.",
            }

        STORAGE.mkdir(parents=True, exist_ok=True)
        stored = STORAGE / f"{digest}{suffix}"
        shutil.copy(tmp_path, stored)

        provider = get_provider(
            os.getenv("LLM_PROVIDER", "gemini"), os.getenv("LLM_MODEL")
        )
        result = InvoicePipeline(provider=provider).process(tmp_path)
        result.source_file = file.filename or stored.name

        outcome = repo.save(result, actor="upload")
        return {
            "duplicate": False,
            "extraction_id": outcome.extraction.id,
            "status": result.status.value,
            "confidence": result.overall_confidence,
            "issues": result.validation_errors,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
