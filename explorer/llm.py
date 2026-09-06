"""Minimal Gemini API client (REST, stdlib only — no SDK dependency).

The API key is read from, in order:
  1. the GEMINI_API_KEY or GOOGLE_API_KEY environment variable
  2. a `.env` file in the repo root with a line like  GEMINI_API_KEY=...
The `.env` file is gitignored; never commit keys.
"""

import json
import os
import urllib.error
import urllib.request

from .db import REPO_ROOT

DEFAULT_MODEL = os.environ.get("PISA_LLM_MODEL", "gemini-2.5-flash")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


def _load_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    return key
    raise LLMError(
        "No Gemini API key found. Set the GEMINI_API_KEY environment variable "
        f"or put GEMINI_API_KEY=<your key> in {env_file}"
    )


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
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"Gemini API error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Gemini API unreachable: {e.reason}") from e

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
