"""LLM access layer.

Three things matter here and they are all about not wasting your free quota:

  1. Disk cache keyed on (prompt hash, model). During development you will
     re-run the same 50 invoices hundreds of times. Without a cache you burn
     your daily quota before lunch.
  2. Exponential backoff on 429. Rate-limit errors are normal operating
     behaviour on a free tier, not bugs.
  3. A MockProvider so the whole pipeline and the test suite run with no
     API key and no network at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))

# Google retires model IDs periodically. This is a default, not a promise —
# `python cli.py --list-models` is the source of truth for your key.
DEFAULT_GEMINI_MODEL = os.getenv("LLM_MODEL") or "gemini-3.6-flash"


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """Transient. Back off and retry."""


class FatalLLMError(LLMError):
    """Permanent: bad model name, bad key, no permission.

    Retrying cannot help, and neither can the next document. These abort the
    whole batch immediately instead of failing once per file.
    """


class BaseProvider(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def _complete(self, prompt: str, temperature: float) -> str: ...

    def complete(self, prompt: str, temperature: float = 0.0, use_cache: bool = True) -> tuple[str, bool]:
        """Returns (response_text, was_cached)."""
        key = self._cache_key(prompt, temperature)
        if use_cache:
            hit = self._cache_read(key)
            if hit is not None:
                logger.debug("cache hit %s", key[:12])
                return hit, True

        text = self._with_backoff(prompt, temperature)
        if use_cache:
            self._cache_write(key, text)
        return text, False

    def _with_backoff(self, prompt: str, temperature: float, max_attempts: int = 5) -> str:
        delay = 1.0
        last: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._complete(prompt, temperature)
            except FatalLLMError:
                raise  # config problem — retrying is pointless
            except RateLimited as exc:
                last = exc
                if attempt == max_attempts:
                    break
                # Jitter stops parallel workers from retrying in lockstep.
                sleep_for = delay + random.uniform(0, 0.5)
                logger.warning("Rate limited, retrying in %.1fs (attempt %d)", sleep_for, attempt)
                time.sleep(sleep_for)
                delay *= 2
        raise LLMError(f"Gave up after {max_attempts} attempts: {last}")

    def _cache_key(self, prompt: str, temperature: float) -> str:
        blob = f"{self.name}|{self.model}|{temperature}|{prompt}"
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_read(self, key: str) -> Optional[str]:
        p = CACHE_DIR / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["response"]
        except Exception:
            return None

    def _cache_write(self, key: str, text: str) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(json.dumps({"response": text}))


class GeminiProvider(BaseProvider):
    """Google AI Studio. Free tier, no credit card.

    Do not send real client documents through the free tier — Google may use
    free-tier inputs to improve its models. Use a paid key or Vertex AI for
    anything with real customer data in it.
    """

    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        timeout: int = 90,
        disable_thinking: bool = True,
    ):
        self.model = model
        self.timeout = timeout
        # Newer Gemini models reason internally before answering. For copying
        # labelled fields out of a document that is wasted latency and wasted
        # quota, so we switch it off. Set False if accuracy suffers on messy scans.
        self.disable_thinking = disable_thinking
        self._thinking_supported = True
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    def _complete(self, prompt: str, temperature: float) -> str:
        config = {
            "temperature": temperature,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        }
        if self.disable_thinking and self._thinking_supported:
            config["thinkingConfig"] = {"thinkingBudget": 0}

        try:
            resp = requests.post(
                f"{self.BASE}/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise LLMError(
                f"No response from {self.model} within {self.timeout}s. "
                f"Try a faster model (a *-flash-lite variant) or raise --timeout."
            ) from exc

        # Not every model accepts thinkingConfig, and Google's rejection is
        # often the generic "Request contains an invalid argument" rather than
        # anything naming the field. So on ANY 400 while we're sending it,
        # drop it and retry once before treating the error as real.
        if resp.status_code == 400 and config.get("thinkingConfig") is not None:
            logger.info(
                "Model %s rejected the request while thinkingConfig was set — "
                "retrying without it", self.model,
            )
            self._thinking_supported = False
            return self._complete(prompt, temperature)
        if resp.status_code == 429:
            raise RateLimited(resp.text[:300])

        if resp.status_code in (400, 401, 403, 404):
            raise FatalLLMError(self._explain(resp.status_code, resp.text))

        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response shape: {json.dumps(data)[:300]}") from exc

    def _explain(self, status: int, body: str) -> str:
        """Google's error bodies are informative — surface the message and add
        the command that resolves it, rather than dumping raw JSON."""
        try:
            detail = json.loads(body)["error"]["message"]
        except Exception:
            detail = body[:300]

        if status == 404:
            return (
                f"Model '{self.model}' is not available to this API key.\n"
                f"  Google says: {detail}\n"
                f"  Run `python cli.py --list-models` to see what your key can use,\n"
                f"  then set LLM_MODEL in your .env accordingly."
            )
        if status in (401, 403):
            return (
                f"API key rejected ({status}).\n"
                f"  Google says: {detail}\n"
                f"  Check GEMINI_API_KEY in .env — get one at https://aistudio.google.com/apikey"
            )
        return f"Request rejected ({status}): {detail}"

    def list_models(self) -> list[dict]:
        """Ask the API what this key can actually use.

        Model names change and older ones get retired. Never hardcode a list
        in documentation — query it."""
        resp = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": self.api_key},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise FatalLLMError(self._explain(resp.status_code, resp.text))
        out = []
        for m in resp.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append({
                    "name": m["name"].removeprefix("models/"),
                    "display": m.get("displayName", ""),
                    "input_limit": m.get("inputTokenLimit"),
                })
        return out


class MockProvider(BaseProvider):
    """Returns a canned extraction. Lets the pipeline and tests run offline.

    This is not a toy — it means your CI can exercise the full path
    (read -> prompt -> parse -> validate -> score) without an API key
    or a single paid token.
    """

    name = "mock"
    model = "mock-1"

    def __init__(self, canned: Optional[dict] = None):
        self.canned = canned

    def _complete(self, prompt: str, temperature: float) -> str:
        if self.canned is not None:
            return json.dumps(self.canned)
        return json.dumps(_naive_parse(prompt))


def _naive_parse(prompt: str) -> dict:
    """Crude regex 'extraction' so MockProvider echoes the real document
    instead of a fixed fixture. Good enough to prove the plumbing works."""
    import re

    text = prompt.split("=== DOCUMENT TEXT ===")[-1]

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "invoice_number": grab(r"invoice\s*(?:#|no\.?|number)[:\s]*([A-Z0-9\-/]+)"),
        "invoice_date": grab(r"(?:invoice\s*)?date[:\s]*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})"),
        "due_date": grab(r"due\s*date[:\s]*([0-9]{1,4}[-/][0-9]{1,2}[-/][0-9]{1,4})"),
        "vendor_name": grab(r"(?:from|vendor|seller)[:\s]*([A-Za-z0-9 .,&'\-]{3,60})"),
        "vendor_tax_id": grab(r"(?:NTN|GST|VAT|Tax\s*ID)[:\s#]*([A-Z0-9\-]{5,20})"),
        "buyer_name": grab(r"(?:bill\s*to|buyer|to)[:\s]*([A-Za-z0-9 .,&'\-]{3,60})"),
        "currency": grab(r"\b(PKR|USD|EUR|GBP|AED)\b"),
        "subtotal": grab(r"sub\s*total[:\s]*[A-Za-z.$₨]*\s*([0-9,]+\.?[0-9]*)"),
        "tax_amount": grab(r"(?:tax|gst|vat)[^:\n]*[:\s]*[A-Za-z.$₨]*\s*([0-9,]+\.?[0-9]*)"),
        "total_amount": grab(
            r"(?:grand\s+total|amount\s+payable|total\s+due|(?<!sub)\btotal)"
            r"[:\s]*[A-Za-z.$₨]*\s*([0-9,]+\.?[0-9]*)"
        ),
        "line_items": [],
        "_field_confidence": {},
    }


def get_provider(
    name: str = "gemini", model: Optional[str] = None, timeout: int = 90
) -> BaseProvider:
    if name == "mock":
        return MockProvider()
    if name == "gemini":
        return GeminiProvider(model=model or DEFAULT_GEMINI_MODEL, timeout=timeout)
    raise ValueError(f"Unknown provider: {name}")
