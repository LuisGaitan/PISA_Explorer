"""Usage analytics: one document per question, updated as the browser reports
what rendered and how the tester felt about it.

Backends (chosen automatically, override with PISA_EVENTS_BACKEND):
  firestore  on Cloud Run (K_SERVICE set) — durable across instances;
             collection "events", document id = event id
  jsonl      locally — append-only data/events.jsonl; later lines with the
             same id are merged over earlier ones when read back

Writes never raise into the request path: analytics must not break answers.
"""

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .db import DATA_DIR

MAX_QUERY_DOCS = 5000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def create(self, doc: dict) -> str:
        event_id = doc.setdefault("id", uuid.uuid4().hex)
        doc.setdefault("ts", _now_iso())
        self._append(doc)
        return event_id

    def update(self, event_id: str, fields: dict) -> None:
        self._append({"id": event_id, **fields})

    def query(self, days: int) -> list[dict]:
        if not self.path.exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        merged: dict[str, dict] = {}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("id")
                if not rid:
                    continue
                if rid in merged:
                    merged[rid].update(rec)
                elif rec.get("ts", "") >= cutoff:
                    merged[rid] = rec
        docs = sorted(merged.values(), key=lambda d: d.get("ts", ""), reverse=True)
        return docs[:MAX_QUERY_DOCS]


class FirestoreStore:
    def __init__(self):
        from google.cloud import firestore  # imported lazily: not needed locally
        self._fs = firestore
        self.client = firestore.Client()
        self.col = self.client.collection("events")

    def create(self, doc: dict) -> str:
        event_id = doc.setdefault("id", uuid.uuid4().hex)
        doc.setdefault("ts", _now_iso())
        self.col.document(event_id).set(doc)
        return event_id

    def update(self, event_id: str, fields: dict) -> None:
        self.col.document(event_id).set(fields, merge=True)

    def query(self, days: int) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        q = (self.col.where(filter=self._fs.FieldFilter("ts", ">=", cutoff))
             .order_by("ts", direction=self._fs.Query.DESCENDING)
             .limit(MAX_QUERY_DOCS))
        return [d.to_dict() for d in q.stream()]


class SafeStore:
    """Wraps a backend so analytics failures are logged, never raised."""

    def __init__(self, backend, name: str):
        self.backend = backend
        self.name = name

    def _guard(self, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 — analytics must never break answers
            print(f"[events:{self.name}] {type(e).__name__}: {e}", file=sys.stderr)
            return None

    def create(self, doc: dict) -> str | None:
        return self._guard(self.backend.create, doc)

    def update(self, event_id: str, fields: dict) -> None:
        self._guard(self.backend.update, event_id, fields)

    def query(self, days: int) -> list[dict]:
        return self._guard(self.backend.query, days) or []


def open_store() -> SafeStore:
    backend = os.environ.get("PISA_EVENTS_BACKEND", "auto")
    if backend == "auto":
        backend = "firestore" if os.environ.get("K_SERVICE") else "jsonl"
    if backend == "firestore":
        try:
            return SafeStore(FirestoreStore(), "firestore")
        except Exception as e:  # noqa: BLE001
            print(f"[events] Firestore unavailable ({e}); falling back to jsonl",
                  file=sys.stderr)
    return SafeStore(JsonlStore(DATA_DIR / "events.jsonl"), "jsonl")
