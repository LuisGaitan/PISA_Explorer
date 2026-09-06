"""PISA Explorer web app — local development server and Cloud Run entrypoint.

  python -m explorer.app                # local: http://127.0.0.1:8765
  python -m explorer.app --port 9000

Environment (all optional locally; set them in production):
  PORT               listen port (Cloud Run sets this; default 8765)
  BIND_HOST          0.0.0.0 in containers; default 127.0.0.1 (local only)
  PISA_ACCESS_CODE   if set, every /api/ask must present it (the UI asks once
                     and remembers it) — REQUIRED for any public deployment
  PISA_RATE_LIMIT    questions per session per hour (default 20)
  PISA_GLOBAL_RATE   questions per hour across ALL users (default 200) —
                     the hard cap on Gemini spend
  GEMINI_API_KEY     the LLM key (or .env locally)

Protection model (lessons from the old prototype's open endpoint):
per-session cookies isolate conversation histories; an access code gates the
spending endpoint; per-session and global hourly rate limits bound cost; the
agent's own guards (read-only DB, SELECT-only SQL) bound what a query can do.
"""

import argparse
import json
import os
import secrets
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import Agent

STATIC_DIR = Path(__file__).resolve().parent / "static"

ACCESS_CODE = os.environ.get("PISA_ACCESS_CODE") or None
RATE_LIMIT = int(os.environ.get("PISA_RATE_LIMIT", "20"))
GLOBAL_RATE = int(os.environ.get("PISA_GLOBAL_RATE", "200"))
RATE_WINDOW = 3600.0
MAX_SESSIONS = 500

_agent: Agent | None = None
_lock = threading.Lock()
_sessions: OrderedDict[str, dict] = OrderedDict()   # sid -> {history, asks}
_global_asks: deque = deque()


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def get_session(sid: str) -> dict:
    session = _sessions.get(sid)
    if session is None:
        session = {"history": [], "asks": deque()}
        _sessions[sid] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    else:
        _sessions.move_to_end(sid)
    return session


def rate_ok(session: dict) -> tuple[bool, str]:
    now = time.time()
    for q in (session["asks"], _global_asks):
        while q and now - q[0] > RATE_WINDOW:
            q.popleft()
    if len(session["asks"]) >= RATE_LIMIT:
        return False, f"rate limit reached ({RATE_LIMIT} questions/hour) — try again later"
    if len(_global_asks) >= GLOBAL_RATE:
        return False, "the service is at its hourly capacity — try again later"
    session["asks"].append(now)
    _global_asks.append(now)
    return True, ""


def _denan(value):
    """Recursively replace non-finite floats with None (strict-JSON safe)."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: _denan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_denan(v) for v in value]
    return value


def result_payload(result) -> dict:
    table = None
    if result.table is not None:
        # astype(object) first — on float columns, .where(..., None) alone
        # silently turns None back into NaN, which is invalid JSON in browsers
        clean = result.table.astype(object).where(result.table.notna(), None)
        table = {"columns": list(clean.columns),
                 "rows": clean.to_dict(orient="records")}
    return {"answer": result.answer, "table": table, "plan": result.plan,
            "provenance": result.provenance, "error": result.error}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str,
              extra_headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict, **kw) -> None:
        try:
            body = json.dumps(payload, default=str, allow_nan=False)
        except ValueError:   # a NaN slipped past scrubbing somewhere unusual
            body = json.dumps(_denan(payload), default=str, allow_nan=False)
        self._send(status, body.encode("utf-8"),
                   "application/json; charset=utf-8", **kw)

    def _sid(self) -> tuple[str, dict | None]:
        cookies = self.headers.get("Cookie", "")
        for part in cookies.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "sid" and len(value) >= 16:
                return value, None
        sid = secrets.token_urlsafe(24)
        return sid, {"Set-Cookie": f"sid={sid}; Path=/; HttpOnly; SameSite=Lax"}

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            _, cookie = self._sid()
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8",
                       extra_headers=cookie)
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            # Always consume the request body BEFORE any early response —
            # with HTTP/1.1 keep-alive, unread body bytes corrupt the next
            # request on the same connection.
            length = int(self.headers.get("Content-Length", 0))
            if length > 100_000:
                self.close_connection = True
                self._send_json(413, {"error": "request too large"})
                return
            raw = self.rfile.read(length)
            if ACCESS_CODE and not secrets.compare_digest(
                    self.headers.get("X-Access-Code", ""), ACCESS_CODE):
                self._send_json(401, {"error": "access code required"})
                return
            body = json.loads(raw or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "empty question"})
                return
            if len(question) > 2000:
                self._send_json(400, {"error": "question too long"})
                return

            sid, cookie = self._sid()
            with _lock:
                session = get_session(sid)
                ok, message = rate_ok(session)
                if not ok:
                    self._send_json(429, {"error": message}, extra_headers=cookie)
                    return
                result = get_agent().ask(question, history=session["history"])
                session["history"].append({
                    "question": question,
                    "answer": result.answer,
                    "explanation": (result.plan or {}).get("explanation"),
                })
                del session["history"][:-6]
            self._send_json(200, result_payload(result), extra_headers=cookie)
        except BrokenPipeError:
            pass
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--host", default=os.environ.get("BIND_HOST", "127.0.0.1"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    guard = "access code ON" if ACCESS_CODE else "no access code (local dev)"
    print(f"PISA Explorer on http://{args.host}:{args.port}  "
          f"[{guard}, {RATE_LIMIT}/h per session, {GLOBAL_RATE}/h global]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
