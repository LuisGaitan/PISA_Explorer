"""PISA Explorer web app — local development server and Cloud Run entrypoint.

  python -m explorer.app                # local: http://127.0.0.1:8765
  python -m explorer.app --port 9000

Environment (all optional locally; set them in production):
  PORT                listen port (Cloud Run sets this; default 8765)
  BIND_HOST           0.0.0.0 in containers; default 127.0.0.1 (local only)
  PISA_ADMIN_CODE     unlocks /admin and /api/admin/*; if unset, admin is off
  PISA_RATE_LIMIT     questions per session per hour (default 20)
  PISA_GLOBAL_RATE    questions per hour across ALL users (default 200) —
                      the hard cap on Gemini spend
  PISA_EVENTS_BACKEND firestore | jsonl | auto (default: firestore on Cloud Run)
  GEMINI_API_KEY      the LLM key (or .env locally)

Access: there is no password. The gate asks visitors for the name of their
institution or organization (free text, remembered by the browser), which is
sent with every request as the X-Institution header (percent-encoded so
non-ASCII names survive HTTP) and recorded on each event for the admin
dashboard. Protection model: per-session cookies isolate conversation
histories; per-session and global hourly rate limits bound Gemini spend; the
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
from urllib.parse import unquote

from .agent import Agent
from .events import open_store

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_TYPES = {".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".png": "image/png"}

RATE_LIMIT = int(os.environ.get("PISA_RATE_LIMIT", "20"))
GLOBAL_RATE = int(os.environ.get("PISA_GLOBAL_RATE", "200"))
RATE_WINDOW = 3600.0
MAX_SESSIONS = 500
ADMIN_CODE = os.environ.get("PISA_ADMIN_CODE") or None
INSTITUTION_MAX_CHARS = 80
INSTITUTION_MIN_CHARS = 2

_agent: Agent | None = None
_lock = threading.Lock()          # guards _sessions / rate counters (short holds)
_agent_lock = threading.Lock()    # serializes analyses on this instance
_sessions: OrderedDict[str, dict] = OrderedDict()   # sid -> {history, asks}
_global_asks: deque = deque()
_events = open_store()


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def clean_institution(raw: str | None) -> str | None:
    """Normalize a visitor-supplied institution name (percent-decoded, single
    spaces, capped length); None when it is missing or too short."""
    text = unquote(str(raw or ""))
    text = " ".join(text.replace("\x00", "").split())[:INSTITUTION_MAX_CHARS].strip()
    return text if len(text) >= INSTITUTION_MIN_CHARS else None


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


def result_payload(result, event_id: str | None) -> dict:
    table = None
    if result.table is not None:
        # astype(object) first — on float columns, .where(..., None) alone
        # silently turns None back into NaN, which is invalid JSON in browsers
        clean = result.table.astype(object).where(result.table.notna(), None)
        table = {"columns": list(clean.columns),
                 "rows": clean.to_dict(orient="records")}
    return {"answer": result.answer, "table": table, "plan": result.plan,
            "provenance": result.provenance, "error": result.error,
            "event_id": event_id}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---------- plumbing ----------

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

    def _read_body(self) -> tuple[dict | None, bytes]:
        """Always consume the request body BEFORE any early response — with
        HTTP/1.1 keep-alive, unread body bytes corrupt the next request."""
        length = int(self.headers.get("Content-Length", 0))
        if length > 100_000:
            self.close_connection = True
            return None, b""
        raw = self.rfile.read(length)
        try:
            return json.loads(raw or b"{}"), raw
        except json.JSONDecodeError:
            return {}, raw

    def _institution(self) -> str | None:
        return clean_institution(self.headers.get("X-Institution", ""))

    def _is_admin(self) -> bool:
        return bool(ADMIN_CODE) and secrets.compare_digest(
            self.headers.get("X-Admin-Code", ""), ADMIN_CODE)

    def _serve_static(self, name: str) -> None:
        path = STATIC_DIR / name
        if "/" in name or "\\" in name or not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = STATIC_TYPES.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ctype,
                   extra_headers={"Cache-Control": "no-cache"})

    # ---------- GET ----------

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            _, cookie = self._sid()
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8", extra_headers=cookie)
        elif path == "/admin":
            self._serve_static("admin.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/health":
            self._send(200, b"ok", "text/plain")
        elif path.startswith("/api/admin/events"):
            self._admin_events()
        else:
            self._send(404, b"not found", "text/plain")

    def _admin_events(self) -> None:
        if not self._is_admin():
            self._send_json(401, {"error": "admin code required"})
            return
        query = self.path.partition("?")[2]
        days = 30
        for part in query.split("&"):
            k, _, v = part.partition("=")
            if k == "days" and v.isdigit():
                days = max(1, min(int(v), 365))
        docs = _events.query(days)
        self._send_json(200, {"days": days, "count": len(docs),
                              "backend": _events.name, "events": docs})

    # ---------- POST ----------

    def do_POST(self):
        routes = {"/api/ask": self._ask, "/api/whoami": self._whoami,
                  "/api/event": self._event}
        handler = routes.get(self.path)
        if handler is None:
            self._read_body()
            self._send(404, b"not found", "text/plain")
            return
        try:
            body, _ = self._read_body()
            if body is None:
                self._send_json(413, {"error": "request too large"})
                return
            handler(body)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            self._send_json(500, {"error": str(e)})

    def _whoami(self, body: dict) -> None:
        """The gate: accepts an institution/organization name and echoes the
        normalized form the server will record."""
        institution = clean_institution(
            body.get("institution") or self.headers.get("X-Institution", ""))
        if institution is None:
            self._send_json(401, {"error": "Please enter the name of your "
                                           "institution or organization."})
            return
        self._send_json(200, {"institution": institution, "gated": True})

    def _ask(self, body: dict) -> None:
        institution = self._institution()
        if institution is None:
            self._send_json(401, {"error": "institution required"})
            return
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json(400, {"error": "empty question"})
            return
        if len(question) > 2000:
            self._send_json(400, {"error": "question too long"})
            return

        sid, cookie = self._sid()
        with _lock:                       # session/rate bookkeeping: microseconds
            session = get_session(sid)
            ok, message = rate_ok(session)
            history = list(session["history"])
        if not ok:
            _events.create({"kind": "rate_limited", "session": sid[:12],
                            "institution": institution, "question": question})
            self._send_json(429, {"error": message}, extra_headers=cookie)
            return
        # One analysis at a time per instance: the DuckDB connection is not
        # thread-safe and an all-economies query can need ~1 GB. Throughput
        # comes from Cloud Run instances, not threads (see DEPLOY.md).
        with _agent_lock:
            result = get_agent().ask(question, history=history)
        with _lock:
            session["history"].append({
                "question": question,
                "answer": result.answer,
                "explanation": (result.plan or {}).get("explanation"),
            })
            del session["history"][:-6]
            turn = len(session["history"])
        event_id = _events.create({
            "kind": "ask", "session": sid[:12], "institution": institution,
            "turn": turn,
            **result.analytics(),
        })
        self._send_json(200, result_payload(result, event_id), extra_headers=cookie)

    def _event(self, body: dict) -> None:
        """Browser-side facts about a question: what rendered, feedback,
        exports, device — merged into that question's event document."""
        if self._institution() is None:
            self._send_json(401, {"error": "institution required"})
            return
        event_id = str(body.get("event_id") or "")
        kind = str(body.get("kind") or "")
        if not (event_id.isalnum() and len(event_id) == 32):
            self._send_json(400, {"error": "bad event id"})
            return
        fields: dict = {}
        if kind == "render":
            fields = {"chart": str(body.get("chart") or "none")[:20],
                      "width": int(body.get("width") or 0),
                      "mobile": bool(body.get("mobile")),
                      "ua": str(body.get("ua") or "")[:200]}
        elif kind == "feedback":
            vote = body.get("vote")
            if vote not in ("up", "down"):
                self._send_json(400, {"error": "vote must be up or down"})
                return
            fields = {"vote": vote, "comment": str(body.get("comment") or "")[:1000],
                      "feedback_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        elif kind == "export":
            fields = {"exported": True}
        else:
            self._send_json(400, {"error": "unknown event kind"})
            return
        _events.update(event_id, fields)
        self._send_json(200, {"ok": True})

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--host", default=os.environ.get("BIND_HOST", "127.0.0.1"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    gate = "institution-name gate (no password)"
    print(f"PISA Explorer on http://{args.host}:{args.port}  "
          f"[{gate}, admin {'ON' if ADMIN_CODE else 'off'}, events -> {_events.name}, "
          f"{RATE_LIMIT}/h per session, {GLOBAL_RATE}/h global]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
