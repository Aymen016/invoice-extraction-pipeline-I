#!/usr/bin/env python3
"""Run the extraction pipeline from the command line.

    python cli.py samples/                     # real Gemini call
    python cli.py samples/ --provider mock     # offline, no API key
    python cli.py samples/inv.pdf --json out/  # write JSON per document
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.llm import FatalLLMError, LLMError, get_provider
from src.pipeline import AUTO_APPROVE_THRESHOLD, InvoicePipeline
from src.schemas import ExtractionResult, ReviewStatus

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)

STATUS_STYLE = {
    ReviewStatus.AUTO_APPROVED: (GREEN, "AUTO-APPROVED"),
    ReviewStatus.NEEDS_REVIEW: (YELLOW, "NEEDS REVIEW"),
    ReviewStatus.FAILED: (RED, "FAILED"),
}


def collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED)


def render(result: ExtractionResult) -> None:
    colour, label = STATUS_STYLE[result.status]
    name = Path(result.source_file).name

    print(f"\n{BOLD}{name}{RESET}")
    print(f"  {colour}{label}{RESET}  confidence {result.overall_confidence:.2f}"
          f"  {DIM}[{result.text_source} · {result.model}"
          f"{' · cached' if result.cached else ''} · {result.elapsed_seconds}s]{RESET}")

    e = result.extraction
    rows = [
        ("Invoice #", e.invoice_number),
        ("Date", e.invoice_date),
        ("Vendor", e.vendor_name),
        ("Buyer", e.buyer_name),
        ("Currency", e.currency),
        ("Subtotal", e.subtotal),
        ("Tax", e.tax_amount),
        ("Total", e.total_amount),
    ]
    for field_label, value in rows:
        if value is None:
            print(f"  {DIM}{field_label:<12} —{RESET}")
            continue
        key = {
            "Invoice #": "invoice_number", "Date": "invoice_date",
            "Vendor": "vendor_name", "Buyer": "buyer_name",
            "Currency": "currency", "Subtotal": "subtotal",
            "Tax": "tax_amount", "Total": "total_amount",
        }[field_label]
        conf = result.confidence_for(key)
        suffix = f"  {DIM}({conf.final_confidence:.2f}){RESET}" if conf else ""
        print(f"  {field_label:<12} {value}{suffix}")

    if e.line_items:
        print(f"  {DIM}Line items   {len(e.line_items)}{RESET}")

    for err in result.validation_errors:
        print(f"  {RED}!{RESET} {err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract structured data from invoices.")
    ap.add_argument("target", type=Path, nargs="?", help="A file or a folder of documents")
    ap.add_argument("--list-models", action="store_true",
                    help="Show which models this API key can use, then exit")
    ap.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "gemini"),
                    choices=["gemini", "mock"])
    ap.add_argument("--model", default=os.getenv("LLM_MODEL"))
    ap.add_argument("--threshold", type=float, default=AUTO_APPROVE_THRESHOLD)
    ap.add_argument("--json", type=Path, metavar="DIR", help="Write one JSON file per document")
    ap.add_argument("--timeout", type=int, default=90, help="Seconds to wait per API call")
    ap.add_argument("--db", nargs="?", const=os.getenv("DATABASE_URL", "sqlite:///invoices.db"),
                    metavar="URL", help="Persist results (default sqlite:///invoices.db)")
    ap.add_argument("--stats", action="store_true", help="Show database summary and exit")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # -v must show OUR debug output only. pdfminer emits a log line per PDF
    # token — thousands of them per page — which buries anything useful.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.verbose:
        logging.getLogger("src").setLevel(logging.DEBUG)
    for noisy in ("pdfminer", "PIL", "urllib3", "pdfplumber"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_models:
        try:
            models = get_provider("gemini", args.model).list_models()
        except (FatalLLMError, LLMError) as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 2
        print(f"{BOLD}Models available to your key{RESET}\n")
        for m in sorted(models, key=lambda x: x["name"]):
            limit = f"{m['input_limit']:,} tok" if m["input_limit"] else ""
            print(f"  {m['name']:<38} {DIM}{limit}{RESET}")
        print(f"\n{DIM}Set the one you want as LLM_MODEL in .env{RESET}")
        return 0

    if args.stats:
        from src.db import init_db, make_engine, make_session_factory
        from src.repository import InvoiceRepository

        engine = make_engine(args.db)
        init_db(engine)
        with make_session_factory(engine)() as session:
            st = InvoiceRepository(session).stats()
        print(f"\n{BOLD}Database{RESET}")
        print(f"  Documents            {st['documents']}")
        print(f"  Vendors              {st['vendors']}")
        for status, count in sorted(st["by_status"].items()):
            print(f"  {status:<20} {count}")
        print(f"  Straight-through     {st['straight_through_rate']:.0%}")
        return 0

    if args.target is None:
        ap.error("a target file or folder is required (or use --list-models)")

    files = collect(args.target)
    if not files:
        print(f"No supported documents found at {args.target}", file=sys.stderr)
        return 1

    # Deduplicate BEFORE extracting. A content hash is free; an API call is not.
    skipped: list[tuple[Path, int]] = []
    if args.db:
        from src.db import init_db, make_engine, make_session_factory
        from src.reader import sha256_of
        from src.repository import InvoiceRepository

        engine = make_engine(args.db)
        init_db(engine)
        digests = {f: sha256_of(f) for f in files}
        with make_session_factory(engine)() as session:
            known = InvoiceRepository(session).known_hashes(list(digests.values()))
        if known:
            skipped = [(f, known[d]) for f, d in digests.items() if d in known]
            files = [f for f in files if digests[f] not in known]
            for path, prior in skipped:
                print(f"  {DIM}already processed: {path.name} "
                      f"(extraction #{prior}) — skipping{RESET}")

    if not files:
        print(f"\n{DIM}Nothing new to process. "
              f"{len(skipped)} document(s) already in the database.{RESET}")
        return 0

    def progress(i, total, path):
        # \r + flush keeps it on one line and appears immediately, so a slow
        # API call looks like work rather than a freeze.
        print(f"\r  [{i}/{total}] {Path(path).name[:45]:<45}", end="", flush=True)

    pipeline = InvoicePipeline(
        provider=get_provider(args.provider, args.model, timeout=args.timeout),
        auto_approve_threshold=args.threshold,
        use_cache=not args.no_cache,
        on_progress=progress,
    )

    model_label = args.model or os.getenv("LLM_MODEL", "default")
    print(f"Processing {len(files)} document(s) with {args.provider} ({model_label})...")
    try:
        results = pipeline.process_many(files)
    except FatalLLMError as exc:
        print(f"\n{RED}Configuration problem — batch stopped.{RESET}\n{exc}", file=sys.stderr)
        return 2

    print("\r" + " " * 60 + "\r", end="")  # clear the progress line

    for r in results:
        render(r)

    if args.db:
        from src.repository import InvoiceRepository

        saved = duplicates = 0
        with make_session_factory(engine)() as session:
            repo = InvoiceRepository(session)
            for r in results:
                if r.status == ReviewStatus.FAILED:
                    continue
                outcome = repo.save(r)
                if outcome.is_duplicate:
                    duplicates += 1
                    print(f"  {DIM}skipped duplicate: {Path(r.source_file).name} "
                          f"(already extraction #{outcome.previous_extraction_id}){RESET}")
                else:
                    saved += 1
        note = f"{saved} saved"
        if duplicates:
            note += f", {duplicates} duplicate(s) caught at write time"
        if skipped:
            note += f", {len(skipped)} skipped before extraction"
        print(f"\n{DIM}Database: {note}{RESET}")

    if args.json:
        args.json.mkdir(parents=True, exist_ok=True)
        for r in results:
            out = args.json / f"{Path(r.source_file).stem}.json"
            out.write_text(r.model_dump_json(indent=2))
        print(f"\n{DIM}JSON written to {args.json}/{RESET}")

    counts = {s: sum(1 for r in results if r.status == s) for s in ReviewStatus}
    total = len(results)
    auto = counts[ReviewStatus.AUTO_APPROVED]
    print(f"\n{BOLD}Summary{RESET}")
    print(f"  {GREEN}{auto} auto-approved{RESET} · "
          f"{YELLOW}{counts[ReviewStatus.NEEDS_REVIEW]} need review{RESET} · "
          f"{RED}{counts[ReviewStatus.FAILED]} failed{RESET}")
    if total:
        print(f"  Straight-through rate: {auto / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
