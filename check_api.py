#!/usr/bin/env python3
"""Isolate the API call from the pipeline.

If `python cli.py samples/` appears to hang, run this. It sends the smallest
possible request and times every stage, so you can tell the difference between
a network problem, a slow model, and a bug in the pipeline.

    python check_api.py
    python check_api.py --model gemini-3.5-flash-lite
"""

import argparse
import json
import os
import sys
import time

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE = "https://generativelanguage.googleapis.com/v1beta"


def stage(label: str):
    print(f"  {label:<42}", end="", flush=True)


def ok(seconds: float, note: str = ""):
    print(f"OK  {seconds:5.2f}s  {note}")


def fail(msg: str):
    print(f"FAILED\n     {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gemini-3.6-flash"))
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    key = os.getenv("GEMINI_API_KEY")
    print("\nGemini API diagnostic\n")

    stage("1. GEMINI_API_KEY present")
    if not key:
        fail("Not set. Check that .env exists and contains GEMINI_API_KEY=...")
        return 1
    ok(0.0, f"({key[:6]}...{key[-4:]})")

    stage("2. Reach googleapis.com")
    t = time.perf_counter()
    try:
        requests.get(f"{BASE}/models", headers={"x-goog-api-key": key}, timeout=20)
        ok(time.perf_counter() - t)
    except requests.Timeout:
        fail("Timed out. Network or firewall is blocking the connection.\n"
             "     On WSL, try: sudo sh -c 'echo nameserver 8.8.8.8 > /etc/resolv.conf'")
        return 1
    except requests.RequestException as exc:
        fail(f"{type(exc).__name__}: {exc}")
        return 1

    stage(f"3. Tiny request to {args.model}")
    t = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE}/models/{args.model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
                "generationConfig": {"maxOutputTokens": 10, "temperature": 0},
            },
            timeout=args.timeout,
        )
    except requests.Timeout:
        fail(f"No response in {args.timeout}s. This model is too slow or unreachable.\n"
             f"     Try: python check_api.py --model gemini-3.5-flash-lite")
        return 1
    elapsed = time.perf_counter() - t

    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text[:250]}")
        return 1
    ok(elapsed)

    print("  4. Realistic invoice-sized request")
    payload = "Invoice No: INV-001\nVendor: Acme Ltd\nTotal: PKR 5,000\n" * 12
    prompt = (
        'Extract the invoice number as JSON: {"invoice_number": string}\n\n' + payload
    )

    # Check 3 sent a bare request and succeeded; check 4 adds three things at
    # once. If it fails we bisect, so the output names the offending parameter
    # instead of leaving you to guess.
    variants = [
        ("JSON mode + no thinking", {
            "maxOutputTokens": 2048, "temperature": 0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        }),
        ("JSON mode only", {
            "maxOutputTokens": 2048, "temperature": 0,
            "responseMimeType": "application/json",
        }),
        ("no thinking only", {
            "maxOutputTokens": 2048, "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        }),
        ("plain", {"maxOutputTokens": 2048, "temperature": 0}),
    ]

    working = None
    for label, config in variants:
        stage(f"   {label}")
        t = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE}/models/{args.model}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
                timeout=args.timeout,
            )
        except requests.Timeout:
            fail(f"timed out after {args.timeout}s")
            continue

        elapsed = time.perf_counter() - t
        if resp.status_code == 200:
            try:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                ok(elapsed, f"-> {text.strip()[:35]}")
            except (KeyError, IndexError):
                ok(elapsed, "(empty response)")
            if working is None:
                working = (label, config, elapsed)
        else:
            try:
                detail = json.loads(resp.text)["error"]["message"]
            except Exception:
                detail = resp.text[:120]
            fail(f"HTTP {resp.status_code}: {detail[:120]}")

    if working is None:
        print("\nEvery variant failed. Try a different model:")
        print("  python check_api.py --model gemini-3.5-flash-lite")
        print("  python check_api.py --model gemini-2.5-flash")
        return 1

    label, config, elapsed = working
    print(f"\nWorking configuration: {label}")
    if "thinkingConfig" not in config:
        print("  This model does not accept thinkingConfig. The pipeline detects")
        print("  that on the first 400 and drops it automatically — no action needed.")
    if "responseMimeType" not in config:
        print("  JSON mode unavailable; the pipeline strips markdown fences anyway.")

    print(f"\nAll checks passed. Roughly {elapsed:.1f}s per document, "
          f"so 5 documents ≈ {elapsed * 5:.0f}s.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
