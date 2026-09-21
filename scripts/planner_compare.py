"""Compare planner models on the live golden set.

    python scripts/planner_compare.py gemini-2.5-flash gemini-3.5-flash gemini-3.1-pro-preview
    python scripts/planner_compare.py --report data/golden_live_A.json data/golden_live_B.json

Each model runs scripts/golden_live.py in a subprocess with
PISA_PLANNER_MODEL set (the router, summarizer and translator stay on the
default model), then the per-case pass/fail, the planner's latency and its
token counts are summarized side by side. Prices are not assumed: the
token counts are printed so a price sheet can be applied.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(model: str, repeats: int) -> Path:
    env = {**os.environ, "PISA_PLANNER_MODEL": model, "PYTHONIOENCODING": "utf-8"}
    before = set(glob.glob(str(ROOT / "data" / "golden_live_*.json")))
    log = ROOT / "data" / f"planner_compare_{model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log, "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "golden_live.py"), "--repeats", str(repeats)],
                       env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), check=False)
    after = set(glob.glob(str(ROOT / "data" / "golden_live_*.json"))) - before
    if not after:
        raise SystemExit(f"no report written for {model}; see {log}")
    return Path(max(after, key=os.path.getmtime))


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    total = len(report)
    passed = sum(1 for c in report if c["passes"] > 0)
    planner_ms, planner_in, planner_out, planner_calls, model = 0.0, 0, 0, 0, None
    seconds, failures, replans = 0.0, [], 0
    for c in report:
        for r in c["runs"]:
            seconds += float(r.get("seconds") or 0)
            roles = ((r.get("timing") or {}).get("llm_by_role") or {})
            p = roles.get("planner") or {}
            planner_ms += float(p.get("ms") or 0)
            planner_in += int(p.get("tokens_in") or 0)
            planner_out += int(p.get("tokens_out") or 0)
            planner_calls += int(p.get("calls") or 0)
            model = model or p.get("model")
            replans += sum(1 for g in (r.get("guards") or []) if g.startswith("hook:grammar_replan")
                           or g.startswith("hook:replan"))
        if c["passes"] == 0:
            failures.append(f"{c['name']}: {'; '.join(c['runs'][0]['problems'])[:120]}")
    return {"file": path.name, "model": model, "passed": passed, "total": total,
            "planner_calls": planner_calls, "planner_ms_per_call": round(planner_ms / planner_calls) if planner_calls else None,
            "planner_tokens_in": planner_in, "planner_tokens_out": planner_out,
            "seconds_per_question": round(seconds / max(1, sum(len(c["runs"]) for c in report)), 1),
            "replans": replans, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--report", nargs="*", help="summarize existing report files instead of running")
    args = ap.parse_args()
    paths = [Path(p) for p in (args.report or [])]
    for model in args.models:
        print(f"running {model} …", flush=True)
        paths.append(run(model, args.repeats))
    rows = [summarize(p) for p in paths]
    cols = ["model", "passed", "total", "planner_calls", "planner_ms_per_call", "planner_tokens_in",
            "planner_tokens_out", "seconds_per_question", "replans"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    for r in rows:
        if r["failures"]:
            print(f"\n{r['model']} failures ({len(r['failures'])}):")
            for f in r["failures"]:
                print("  - " + f)


if __name__ == "__main__":
    main()
