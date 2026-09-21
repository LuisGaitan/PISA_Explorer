"""Run a list of questions through the local engine and save everything a
reviewer needs to judge each answer: route, plan, the first rows of the
table, notes, the app's verified statements, guards, summary mode, timing.

    python scripts/redteam_run.py questions.json out.json [--start 0 --end 50]

questions.json is either a JSON list of strings or a list of objects with a
"question" key (extra keys are copied through). Questions run one at a time
with a fresh conversation, except that consecutive objects sharing the same
"thread" value are asked as one conversation: each sees the previous
question/answer pairs of its thread (last 6, like the web app) so follow-ups
("and for reading?", "why?") can be tested the way users ask them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()
    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    items = [{"question": q} if isinstance(q, str) else dict(q) for q in items][args.start:args.end]
    from explorer.agent import Agent
    agent = Agent()
    out_path = Path(args.out)
    results = []
    history: list = []
    prev_thread = None
    for i, item in enumerate(items):
        t0 = time.time()
        thread = item.get("thread")
        if thread is None or thread != prev_thread:
            history = []
        prev_thread = thread
        try:
            r = agent.ask(item["question"], history=list(history))
            if thread is not None:
                history.append({"question": item["question"], "answer": r.answer,
                                "explanation": (r.plan or {}).get("explanation")})
                del history[:-6]
            table = None
            if r.table is not None:
                head = r.table.head(12).astype(object).where(r.table.head(12).notna(), None)
                table = {"columns": list(r.table.columns), "rows": int(len(r.table)),
                         "head": head.to_dict(orient="records")}
            rec = {**item, "route": r.route, "template": (r.plan or {}).get("template"),
                   "answer": r.answer, "plan": {k: v for k, v in (r.plan or {}).items() if not k.startswith("_")},
                   "table": table, "notes": (r.provenance or {}).get("notes"),
                   "facts": (r.provenance or {}).get("facts"), "method": (r.provenance or {}).get("method"),
                   "guards": r.guards, "summary_mode": r.summary_mode, "prose_issues": r.prose_issues,
                   "error": r.error}
        except Exception as e:  # noqa: BLE001
            rec = {**item, "route": "exception", "error": f"{type(e).__name__}: {e}"}
            if thread is not None:
                history.append({"question": item["question"], "answer": ""})
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"[{args.start + i}] {rec['route']} {rec.get('template') or ''} {rec['seconds']}s | "
              f"{item['question'][:80]}", flush=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} answers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
