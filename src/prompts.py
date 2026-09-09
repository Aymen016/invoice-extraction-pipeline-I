"""Prompts live in their own module and carry a version string.

Reason: in Phase 4 you will run the eval harness against PROMPT_V1, change
something, run it against PROMPT_V2, and put both accuracy numbers in the
README. That table is the most credible thing on the repo. It only works
if prompts are versioned artefacts rather than strings buried in a function.
"""

PROMPT_VERSION = "v1"

EXTRACTION_PROMPT = """You are an invoice data extraction system. Extract structured data from the document text below.

Return ONLY a JSON object. No markdown fences, no commentary.

Required shape:
{{
  "invoice_number": string | null,
  "invoice_date": "YYYY-MM-DD" | null,
  "due_date": "YYYY-MM-DD" | null,
  "vendor_name": string | null,
  "vendor_tax_id": string | null,
  "vendor_address": string | null,
  "buyer_name": string | null,
  "currency": "ISO 4217 code" | null,
  "subtotal": number | null,
  "tax_amount": number | null,
  "total_amount": number | null,
  "line_items": [
    {{"description": string, "quantity": number | null, "unit_price": number | null, "line_total": number | null}}
  ],
  "_field_confidence": {{"field_name": 0.0-1.0, ...}}
}}

Rules:
- If a field is not present in the document, return null. Never guess, never
  infer, never invent a plausible value. A null is a correct answer.
- Copy values verbatim from the document. Do not reformat numbers except to
  strip currency symbols and thousands separators.
- Dates must be ISO format (YYYY-MM-DD). If the format is ambiguous between
  day-first and month-first, prefer day-first and lower that field's confidence.
- vendor_name is who is being PAID. buyer_name is who is PAYING. Getting these
  backwards is the single most common failure — check which address sits under
  "Bill To".
- Give every field you populate an entry in "_field_confidence": 1.0 when the
  label is explicit and unambiguous, lower when you had to interpret layout,
  below 0.6 when you are genuinely unsure.

=== DOCUMENT TEXT ===
{document_text}
"""


def build_extraction_prompt(document_text: str, max_chars: int = 60_000) -> str:
    if len(document_text) > max_chars:
        # Invoice fields cluster at the top and bottom; the middle of a long
        # line-item table is the safest thing to drop.
        head = document_text[: max_chars // 2]
        tail = document_text[-max_chars // 2 :]
        document_text = f"{head}\n\n[... middle truncated ...]\n\n{tail}"
    return EXTRACTION_PROMPT.format(document_text=document_text)
