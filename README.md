# Invoice Extraction Pipeline

Turns invoices and purchase orders — PDFs or phone photos — into validated
database rows. Extractions the system is confident about post straight through.
Everything else is routed to a human for review.

> **Status: Phase 1 of 5.** Extraction core is complete and tested.
> Database, review UI, eval harness, and deployment still to come.

---

## Why this exists

Small businesses pay someone to retype invoices into Excel. That person is
slow, expensive, and makes mistakes. Full automation is not the answer either —
an LLM that silently gets a total wrong is worse than no automation at all.

So this system does something narrower and more useful: it extracts what it
can, **scores how much it trusts each field**, and only auto-approves what
passes both a confidence threshold and a set of deterministic arithmetic
checks. The human sees the 20% that needs judgement instead of 100% of the
typing.

---

## Quick start

```bash
git clone <your-repo-url> && cd invoice-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python make_samples.py                              # generate test invoices
python cli.py samples/ --provider mock              # run offline, no API key
```

That last command runs the entire pipeline with zero setup and zero cost.
When you want real extraction:

```bash
cp .env.example .env               # add your key from https://aistudio.google.com/apikey
python cli.py --list-models        # see what your key can actually use
python cli.py samples/
```

**On model names:** Google retires model IDs periodically, and a retired ID
returns HTTP 404. `LLM_MODEL` in `.env` is a default, not a promise —
`--list-models` queries the API and is the only reliable source of truth for
your key. The pipeline treats 404/401/403 as fatal and stops the batch
immediately rather than failing identically on every document.

### Tesseract (required only for scanned documents)

`pytesseract` is just a Python wrapper — the OCR engine is a separate system
binary. Digital PDFs with a text layer never touch it, so skip this if you are
only processing born-digital invoices.

```bash
sudo apt install tesseract-ocr     # Ubuntu / WSL
brew install tesseract             # macOS
tesseract --version                # verify
```

Windows: https://github.com/UB-Mannheim/tesseract/wiki

---

## How it works

```
document (PDF / JPG)
      │
      ▼
  reader.py ──── text layer present? ── yes ──► pdfplumber (free, exact)
      │                                └─ no ──► Tesseract OCR
      ▼
  compaction ─── strips layout padding, ~85% fewer tokens
      │
      ▼
  prompts.py ─── versioned prompt (v1)
      │
      ▼
  llm.py ─────── disk cache → Gemini → exponential backoff on 429
      │
      ▼
  schemas.py ─── Pydantic validation + type coercion
      │
      ├─► validation.py ── does subtotal + tax = total? do line items sum?
      │
      └─► confidence.py ── model self-report × grounding check
      │
      ▼
  pipeline.py ── AUTO_APPROVED | NEEDS_REVIEW | FAILED
```

### The two ideas that matter

**1. Grounding beats self-reported confidence.**
Models are cheerfully certain about values they invented. So every extracted
field is checked against the source: *does this string actually appear in the
document?* If the model says the total is 45,300 and "45,300" is nowhere on
the page, that is a fabrication no matter how confident it claims to be.

```
final_confidence = 0.4 × model_confidence + 0.6 × grounding
```

**What grounding cannot do.** It compares strings, so it is blind to any field
the model was asked to *transform*. Two cases found by running real model output
against the sample set:

- *Currency.* The model emits ISO codes, so `USD` never appears in a document
  printing `$`. Fixed by grounding against the symbols a code could derive from.
- *Dates.* Given `05/01/2026`, the digits 05, 01 and 2026 are present whether the
  model read 5 January or 1 May — a day/month swap is undetectable. Real runs
  showed the same model reading one document day-first and another month-first.
  Fixed by detecting undecidable dates and routing them to a human rather than
  reporting a confidence the check cannot justify.

Knowing where a signal fails is worth as much as the signal.

**2. Invoices have to add up, and that arithmetic is free ground truth.**
No LLM call needed to know that `subtotal + tax` should equal `total`.
Deterministic checks catch a whole class of errors that confidence scoring
never will, and they cost nothing to run.

---

## Cost

| Component | Cost |
|---|---|
| LLM (Gemini Flash free tier) | $0 |
| OCR (Tesseract, local) | $0 |
| PDF text extraction | $0 |
| **Development total** | **$0** |

Content-hash caching means re-running the same document never re-calls the API.
This matters enormously in development, where you will process the same test
set hundreds of times.

> **Privacy note:** Google may use free-tier inputs to improve its models.
> Do not send real client documents through a free-tier key — use a paid key
> or Vertex AI for anything containing real customer data.

---

## Usage

```bash
python cli.py samples/                        # process a folder
python cli.py samples/ --db                   # and save to sqlite:///invoices.db
python cli.py --stats                         # database summary
python cli.py invoice.pdf --json out/         # write structured JSON
python cli.py samples/ --threshold 0.9        # stricter auto-approval
python cli.py samples/ --provider mock        # offline
python cli.py samples/ --no-cache -v          # force fresh calls, verbose
```

As a library:

```python
from src.pipeline import InvoicePipeline

result = InvoicePipeline().process("invoice.pdf")

print(result.status)                          # ReviewStatus.AUTO_APPROVED
print(result.extraction.total_amount)         # Decimal('157294.00')
print(result.overall_confidence)              # 0.93
print(result.validation_errors)               # []
```

---

## Tests

```bash
pytest                # 83 tests, no API key, no network
```

`pytest.ini` sets `pythonpath = .` and a root `conftest.py` mirrors it, so
`from src...` resolves whether you run `pytest`, `python -m pytest`, or your
IDE's runner.

The `MockProvider` lets CI exercise the full path — read, prompt, parse,
validate, score — on every push without spending a token.

---

## Sample documents

`make_samples.py` generates deliberately varied invoices, because a uniform
test set gives you a flatteringly wrong accuracy number:

| File | What it tests |
|---|---|
| `invoice_clean_pkr.pdf` | Happy path, PKR, tax adds up |
| `invoice_usd_notax.pdf` | Different labels, no tax line, USD |
| `invoice_bad_math.pdf` | Arithmetic error — validation must catch it |
| `invoice_ambiguous_date.pdf` | `03/04/2026` — March or April? |
| `invoice_scanned.jpg` | No text layer, forces the OCR path |

---

## Review UI (Phase 3)

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
./run.sh                    # API on :8000, UI on :5173
```

Three panes: the queue on the left ordered worst-confidence-first, the original
document in the middle, the extracted values on the right.

Each field carries a mark in the left margin — a tick where the value was found
verbatim in the document, a query mark where it was only partly matched, a flag
where it could not be located at all. The reviewer's eye goes to the flags. It
is the same convention a bookkeeper uses with a coloured pencil, which is what
this screen replaces.

Corrections save on blur and are audited immediately. `j`/`k` move through the
queue, `e` jumps to the first queried field, `a` approves, `x` rejects. Someone
clearing forty invoices should never need the mouse.

Rejection requires a reason; the API returns 422 without one.

---

## Storage (Phase 2)

SQLite by default, Postgres by setting `DATABASE_URL` — no dialect-specific
code, so the swap is a connection string.

**Deduplication is by content hash, not filename.** The same invoice forwarded
under a new name, or re-scanned, is still the same invoice. Paying one twice is
the expensive mistake this prevents.

The hash check runs *before* extraction, not after. Discovering a duplicate at
write time still costs you the API call that produced it; checking first costs
a hash. Point the CLI at a folder repeatedly and only genuinely new documents
are ever sent to the model.

**The audit log is append-only.** Every approval, rejection and field correction
records who, when, and the value before and after. Nothing is overwritten.

**The system learns from corrections.** When a reviewer fixes a date and the
day and month have traded places, that reveals how the vendor writes dates.
After three consistent confirmations the convention is trusted and that
vendor's invoices stop reaching the review queue. Corrections that are not
swaps teach nothing and are ignored — an ordinary typo fix is not evidence
about ordering.

Money is stored as `Numeric(18,2)`, never float.

---

## Roadmap

- [x] **Phase 1** — Extraction core, confidence scoring, validation, tests
- [x] **Phase 2** — Persistence, deduplication, audit trail, vendor learning
- [x] **Phase 3** — FastAPI backend + React review UI
- [ ] **Phase 4** — Eval harness on 50 labelled invoices, field-level accuracy table
- [ ] **Phase 5** — Docker, GitHub Actions, deployment

---

## Licence

MIT
