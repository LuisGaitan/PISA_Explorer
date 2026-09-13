"""Minimal Gemini API client (REST, stdlib only — no SDK dependency).

The API key is read from, in order:
  1. the GEMINI_API_KEY or GOOGLE_API_KEY environment variable
  2. a `.env` file in the repo root with a line like  GEMINI_API_KEY=...
The `.env` file is gitignored; never commit keys.
"""

import json
import os
import time
import urllib.error
import urllib.request

from .db import REPO_ROOT

DEFAULT_MODEL = os.environ.get("PISA_LLM_MODEL", "gemini-2.5-flash")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


# Per-question call accounting (the agent runs one question at a time under
# a lock, so a module-level tally is safe). reset_stats() before a question,
# stats() after.
_stats = {"calls": 0, "ms": 0.0}


def reset_stats() -> None:
    _stats["calls"], _stats["ms"] = 0, 0.0


def stats() -> dict:
    return {"llm_calls": _stats["calls"], "llm_ms": round(_stats["ms"])}


def _clean_key(raw: str) -> str:
    """Strip whitespace, quotes and a UTF-8 BOM. A key that reaches us via a
    shell pipe or secret manager can carry a leading '\\ufeff' and a trailing
    newline; either one breaks the latin-1-encoded HTTP header."""
    return raw.strip().lstrip("﻿").strip().strip("'\"").strip()


def _load_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            key = _clean_key(os.environ[name])
            if key:
                return key
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = _clean_key(line.split("=", 1)[1])
                if key:
                    return key
    raise LLMError(
        "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
        f"or put GEMINI_API_KEY=<your key> in {env_file}"
    )


RETRY_STATUSES = {429, 500, 502, 503, 504}   # rate-limited / transient upstream
RETRY_DELAYS = (1.0, 3.0)                    # seconds before attempt 2 and 3


def _post_with_retry(request: urllib.request.Request) -> dict:
    """POST once, retrying transient failures (Gemini overload, rate limit,
    network blips) with short backoff — at hundreds of concurrent users a
    single 503 must not become a failed answer."""
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            if e.code not in RETRY_STATUSES or attempt == attempts - 1:
                raise LLMError(f"Gemini API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt == attempts - 1:
                raise LLMError(f"Gemini API unreachable: {e.reason}") from e
        time.sleep(RETRY_DELAYS[attempt])
    raise LLMError("Gemini API: retries exhausted")   # unreachable


def generate(
    prompt: str,
    system: str | None = None,
    json_mode: bool = False,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """One non-streaming generation call; returns the text of the reply."""
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"

    request = urllib.request.Request(
        _ENDPOINT.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": _load_key()},
        method="POST",
    )
    started = time.time()
    try:
        payload = _post_with_retry(request)
    finally:
        _stats["calls"] += 1
        _stats["ms"] += (time.time() - started) * 1000

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected Gemini response shape: {str(payload)[:500]}") from e


def generate_json(prompt: str, system: str | None = None,
                  model: str = DEFAULT_MODEL) -> dict:
    """Generation call that must return a JSON object. Models sometimes emit
    an ARRAY of objects (e.g. one plan per requested measure) — unwrap to the
    first object rather than crashing downstream .get() calls."""
    text = generate(prompt, system=system, json_mode=True, model=model)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Gemini returned invalid JSON: {text[:500]}") from e
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), None)
    if not isinstance(parsed, dict):
        raise LLMError(f"Gemini returned {type(parsed).__name__}, expected a "
                       f"JSON object: {text[:300]}")
    return parsed
