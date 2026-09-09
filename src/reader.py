"""Turn a PDF or image file into plain text.

Strategy, in order of preference:
  1. If the PDF has a real text layer, use it. Fast, free, perfectly accurate.
  2. If not (a scan or a photo), rasterise and run Tesseract OCR.

Most B2B invoices arrive as digital PDFs with a text layer, so path 1
handles the majority for free. Never OCR a document that does not need it.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# Below this many characters we assume the "text layer" is junk
# (page numbers, a stray watermark) and fall back to OCR.
MIN_USEFUL_CHARS = 120


class MissingOCREngine(RuntimeError):
    """Raised when a scanned document needs OCR but Tesseract isn't installed."""


@dataclass
class ReadResult:
    text: str
    source: str  # "pdf_text" | "ocr" | "image_ocr"
    page_count: int
    sha256: str


def sha256_of(path: str | Path) -> str:
    """Content hash. This is our cache key — identical file, identical result."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_document(path: str | Path) -> ReadResult:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such document: {path}")

    digest = sha256_of(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        return ReadResult(_ocr_image(path), "image_ocr", 1, digest)

    if suffix != ".pdf":
        raise ValueError(f"Unsupported file type: {suffix}")

    text, pages = _extract_pdf_text(path)
    if len(text.strip()) >= MIN_USEFUL_CHARS:
        return ReadResult(text, "pdf_text", pages, digest)

    logger.info("Thin text layer (%d chars) — falling back to OCR", len(text.strip()))
    return ReadResult(_ocr_pdf(path), "ocr", pages, digest)


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    """Layout-preserving extraction. Layout matters enormously for invoices —
    a line-item table collapsed into one string is much harder for the LLM."""
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text(layout=True) or ""
            parts.append(f"--- PAGE {i} ---\n{compact(page_text)}")
    return "\n\n".join(parts), page_count


def compact(text: str) -> str:
    """Collapse layout padding without destroying column structure.

    layout=True pads every line to full page width, which on a simple invoice
    means ~70% of the characters are spaces you are paying tokens for. Runs of
    3+ spaces become a tab-like separator (preserving "this is a column break")
    and blank-line runs collapse. Typically cuts token count by half or more.
    """
    lines = []
    blank_run = 0
    for line in text.splitlines():
        line = re.sub(r" {3,}", "   ", line.rstrip())
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        lines.append(line)
    return "\n".join(lines).strip()


def _ocr_pdf(path: Path) -> str:
    import pdfplumber
    import pytesseract

    _require_tesseract()
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # 200 DPI is the sweet spot: enough for 8pt invoice text,
            # not so much that OCR crawls.
            image = page.to_image(resolution=200).original
            parts.append(f"--- PAGE {i} ---\n{pytesseract.image_to_string(image)}")
    return "\n\n".join(parts)


def _ocr_image(path: Path) -> str:
    import pytesseract
    from PIL import Image

    _require_tesseract()
    with Image.open(path) as img:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img)


def _require_tesseract() -> None:
    """Fail with an actionable message rather than a raw wrapper traceback.

    pytesseract is only a Python wrapper — the OCR engine is a separate
    system binary, and forgetting to install it is the single most common
    setup problem with this project.
    """
    import shutil

    if shutil.which("tesseract") is None:
        raise MissingOCREngine(
            "Tesseract OCR is not installed or not on PATH.\n"
            "  Ubuntu/WSL : sudo apt install tesseract-ocr\n"
            "  macOS      : brew install tesseract\n"
            "  Windows    : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Digital PDFs with a text layer do not need it — only scans and photos do."
        )
