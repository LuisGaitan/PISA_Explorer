"""Replay recorded production questions against the current build and diff.

The event log already holds every question users asked and what the app
answered (route, template, variables, the answer text). Re-asking them
against a new build turns real traffic into a regression suite: a question
that used to route to data and now clarifies, a variable that changed, a
number that disappeared from the answer, or a new error shows up here
before a user sees it.

    python scripts/replay_events.py                      # data/events.jsonl, last 7 days
    python scripts/replay_events.py --days 30 --limit 80
    python scripts/replay_events.py --events export.csv  # the admin page's CSV export
    python scripts/replay_events.py --url https://.../   # ask a running server

Needs GEMINI_API_KEY and (without --url) the local DuckDB. Output: one line
per question with the differences, and data/replay_<timestamp>.json.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from golden_live import ask_http, ask_local, numbers_in   # noqa: E402


def load_events(path: Path, days: int) -> list[dict]:
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["variables"] = [v for v in (r.get("variables") or "").split("|") if v]
        return [r for r in rows if r.get("kind") == "ask"]
    from explorer.events import JsonlStore
    return [e for e in JsonlStore(path).query(days) if e.get("kind") == "ask"]


def diff(old: dict, new: dict) -> list[str]:
    out = []
    if old.get("route") != new.get("route"):
        out.append(f"route {old.get('route')} -> {new.get('route')}")
    old_t, new_t = old.get("template"), (new.get("plan") or {}).get("template")
    if old_t and old_t != new_t:
        out.append(f"template {old_t} -> {new_t}")
    old_vars = set(old.get("variables") or [])
    new_vars = {v["variable"] for v in (new.get("provenance") or {}).get("variables") or []}
    if old_vars and new_vars and old_vars != new_vars:
        out.append(f"variables {sorted(old_vars - new_vars)} -> {sorted(new_vars - old_vars)}")
    old_nums = {round(v, 1) for v, d in numbers_in(old.get("answer") or "") if d or v >= 100}
    new_nums = {round(v, 1) for v, d in numbers_in(new.get("answer") or "") if d or v >= 100}
    lost = sorted(old_nums - new_nums)
    if old_nums and lost:
        out.append(f"numbers no longer in the answer: {lost[:6]}")
    if new.get("error"):
        out.append(f"ERROR now: {new['error'][:120]}")
    elif old.get("error"):
        out.append("fixed: used to error")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(ROOT / "data" / "events.jsonl"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--url")
    args = ap.parse_args()
    events = load_events(Path(args.events), args.days)
    seen, todo = set(), []
    for e in events:                      # newest first; one run per distinct question
        q = (e.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q)
            todo.append(e)
        if len(todo) >= args.limit:
            break
    agent = None
    if not args.url:
        from explorer.agent import Agent
        agent = Agent()
    report, changed = [], 0
    for e in todo:
        t0 = time.time()
        try:
            new = ask_http(args.url, e["question"]) if args.url else ask_local(agent, e["question"])
        except Exception as ex:  # noqa: BLE001
            new = {"answer": "", "error": f"{type(ex).__name__}: {ex}"}
        d = diff(e, new)
        changed += bool(d)
        print(("SAME " if not d else "DIFF ") + f"[{round(time.time() - t0)}s] {e['question'][:90]}"
              + ("" if not d else "\n      " + "; ".join(d)))
        report.append({"question": e["question"], "institution": e.get("institution"), "then": {
            "route": e.get("route"), "template": e.get("template"), "variables": e.get("variables"),
            "answer": e.get("answer"), "error": e.get("error"), "vote": e.get("vote")},
            "now": {"route": new.get("route"), "template": (new.get("plan") or {}).get("template"),
                    "answer": new.get("answer"), "error": new.get("error"), "guards": new.get("guards"),
                    "summary_mode": new.get("summary_mode"), "prose_issues": new.get("prose_issues")},
            "diff": d})
    out = ROOT / "data" / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(report)} questions replayed, {changed} changed. Details: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
