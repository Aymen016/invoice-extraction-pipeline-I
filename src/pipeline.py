"""The Phase 1 pipeline: a document goes in, a scored ExtractionResult comes out.

    read -> prompt -> LLM -> parse JSON -> validate schema
         -> check arithmetic -> score confidence -> route

Routing is the whole point of the product. Everything above the threshold
posts straight to the database. Everything below goes to a human. Getting
that threshold right is what turns "a chatbot that reads invoices" into
"a system a business can actually run on".
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .confidence import (
    AMBIGUOUS_DATE_CEILING,
    missing_critical_fields,
    overall_confidence,
    score_extraction,
)
from .llm import BaseProvider, FatalLLMError, get_provider
from .prompts import build_extraction_prompt
from .reader import read_document
from .schemas import ExtractionResult, InvoiceExtraction, ReviewStatus
from .validation import validate

logger = logging.getLogger(__name__)

# Tuned in Phase 4 against labelled data. Starting point only.
AUTO_APPROVE_THRESHOLD = 0.85


class InvoicePipeline:
    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        auto_approve_threshold: float = AUTO_APPROVE_THRESHOLD,
        use_cache: bool = True,
        on_progress=None,
    ):
        self.provider = provider or get_provider("gemini")
        self.threshold = auto_approve_threshold
        self.use_cache = use_cache
        # Called as on_progress(index, total, path, stage). Without it a long
        # batch is indistinguishable from a hang.
        self.on_progress = on_progress

    def process(self, path: str | Path) -> ExtractionResult:
        started = time.perf_counter()
        doc = read_document(path)

        prompt = build_extraction_prompt(doc.text)
        raw, cached = self.provider.complete(prompt, temperature=0.0, use_cache=self.use_cache)

        payload = _parse_json(raw)
        model_confs = payload.pop("_field_confidence", {}) or {}

        try:
            extraction = InvoiceExtraction.model_validate(payload)
        except ValidationError as exc:
            logger.error("Schema validation failed for %s", path)
            return ExtractionResult(
                source_file=str(path),
                source_sha256=doc.sha256,
                extraction=InvoiceExtraction(),
                validation_errors=[f"Schema validation failed: {exc.error_count()} errors"],
                status=ReviewStatus.FAILED,
                text_source=doc.source,
                model=self.provider.model,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                cached=cached,
            )

        confidences = score_extraction(extraction, doc.text, model_confs)
        errors = validate(extraction)

        missing = missing_critical_fields(extraction)
        if missing:
            errors.append(f"Missing critical field(s): {', '.join(missing)}")

        # A date the document renders ambiguously (05/01/2026) cannot be
        # resolved by grounding, so say so plainly instead of letting a capped
        # confidence score quietly decide.
        for field in ("invoice_date", "due_date"):
            conf = next((c for c in confidences if c.field == field), None)
            if conf and conf.grounding_score <= AMBIGUOUS_DATE_CEILING:
                errors.append(
                    f"{field} '{conf.value}' read from an ambiguous numeric date — "
                    f"day-first and month-first both fit. Confirm against the document."
                )

        score = overall_confidence(confidences)

        # A clean arithmetic check is strong evidence. Reward it.
        if not errors and extraction.total_amount is not None and extraction.subtotal is not None:
            score = min(1.0, score + 0.05)

        if errors:
            status = ReviewStatus.NEEDS_REVIEW
        elif score >= self.threshold:
            status = ReviewStatus.AUTO_APPROVED
        else:
            status = ReviewStatus.NEEDS_REVIEW

        return ExtractionResult(
            source_file=str(path),
            source_sha256=doc.sha256,
            extraction=extraction,
            confidences=confidences,
            validation_errors=errors,
            status=status,
            overall_confidence=round(score, 3),
            text_source=doc.source,
            model=self.provider.model,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            cached=cached,
        )

    def process_many(self, paths: list[str | Path]) -> list[ExtractionResult]:
        """Per-document errors are recorded and the batch continues.

        Configuration errors are different: a bad model name or rejected key
        will fail identically on every remaining document, so we stop at the
        first one instead of emitting the same traceback N times.
        """
        results = []
        total = len(paths)
        for i, p in enumerate(paths, start=1):
            if self.on_progress:
                self.on_progress(i, total, p)
            try:
                results.append(self.process(p))
            except FatalLLMError:
                raise
            except Exception as exc:
                logger.exception("Failed on %s", p)
                results.append(
                    ExtractionResult(
                        source_file=str(p),
                        source_sha256="",
                        extraction=InvoiceExtraction(),
                        validation_errors=[f"{type(exc).__name__}: {exc}"],
                        status=ReviewStatus.FAILED,
                    )
                )
        return results


def _parse_json(raw: str) -> dict:
    """Models sometimes wrap JSON in markdown fences despite instructions."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the outermost braces.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
