"""Local web app for PISA Explorer.

  python -m explorer.app          # serves http://127.0.0.1:8765
  python -m explorer.app --port 9000

Single-user local server (binds 127.0.0.1 only). The agent's DuckDB
connection is serialized behind a lock; the LLM never sees this server —
it only fills analysis plans, as in the CLI.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import Agent

STATIC_DIR = Path(__file__).resolve().parent / "static"

_agent: Agent | None = None
_agent_lock = threading.Lock()


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def result_payload(result) -> dict:
    table = None
    if result.table is not None:
        clean = result.table.where(result.table.notna(), None)
        table = {
            "columns": list(clean.columns),
            "rows": clean.to_dict(orient="records"),
        }
    return {
        "answer": result.answer,
        "table": table,
        "plan": result.plan,
        "provenance": result.provenance,
        "error": result.error,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            html = (STATIC_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "empty question"})
                return
            if len(question) > 2000:
                self._send_json(400, {"error": "question too long"})
                return
            with _agent_lock:
                result = get_agent().ask(question)
            self._send_json(200, result_payload(result))
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):  # quiet the request log
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PISA Explorer running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
