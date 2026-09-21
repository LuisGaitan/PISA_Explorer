"""Golden regression set, tier 2: question -> answer through the language
model (tests/golden/questions.json). Run before a deploy:

    python scripts/golden_live.py                 # every case once
    python scripts/golden_live.py --repeats 2     # pass rate per case
    python scripts/golden_live.py --only kosovo_rank
    python scripts/golden_live.py --url https://.../   # against a running
                                                       # server instead of
                                                       # the local engine

Needs GEMINI_API_KEY (or .env) and the local DuckDB for the local mode.
Exit code 1 when any case fails in every repeat. Results are also written
to data/golden_live_<timestamp>.json (answers, plans, guards) for review.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "golden" / "questions.json"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[tuple[float, int]]:
    out = []
    for m in NUMBER.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        out.append((abs(v), len(raw.split(".")[1]) if "." in raw else 0))
    return out


def number_present(want: str, text: str) -> bool:
    w = float(want.replace(",", ""))
    d = len(want.split(".")[1]) if "." in want else 0
    tol = 0.5 * 10 ** (-d) + 1e-9
    return any(abs(v - w) <= tol for v, _ in numbers_in(text))


def judge(case: dict, result: dict) -> list[str]:
    problems = []
    answer = result.get("answer") or ""
    route = result.get("route")
    plan = result.get("plan") or {}
    prov = result.get("provenance") or {}
    variables = {v["variable"] for v in prov.get("variables") or []}
    if "route" in case and route != case["route"]:
        problems.append(f"route {route!r} != {case['route']!r}")
    if "route_any" in case and route not in case["route_any"]:
        problems.append(f"route {route!r} not in {case['route_any']}")
    if "template" in case and plan.get("template") != case["template"]:
        problems.append(f"template {plan.get('template')!r} != {case['template']!r}")
    if "template_any" in case and plan.get("template") not in case["template_any"]:
        problems.append(f"template {plan.get('template')!r} not in {case['template_any']}")
    notes_early = " ".join(prov.get("notes") or [])
    if case.get("notes_contain_any") and not any(f.lower() in notes_early.lower() for f in case["notes_contain_any"]):
        problems.append(f"notes have none of {case['notes_contain_any']}")
    if case.get("variables_any") and not (variables & set(case["variables_any"])):
        problems.append(f"none of {case['variables_any']} used (used: {sorted(variables)[:8]})")
    for v in case.get("variables_all", []):
        if v not in variables:
            problems.append(f"variable {v} not used")
    for v in case.get("variables_forbid", []):
        if v in variables:
            problems.append(f"variable {v} must not be used")
    for n in case.get("answer_numbers", []):
        if not number_present(n, answer):
            problems.append(f"answer lacks {n}")
    for frag in case.get("answer_contains", []):
        if frag.lower() not in answer.lower():
            problems.append(f"answer lacks {frag!r}")
    if case.get("answer_contains_any") and not any(f.lower() in answer.lower() for f in case["answer_contains_any"]):
        problems.append(f"answer has none of {case['answer_contains_any']}")
    for frag in case.get("answer_forbid", []):
        if frag.lower() in answer.lower():
            problems.append(f"answer must not say {frag!r}")
    notes = " ".join(prov.get("notes") or [])
    for frag in case.get("notes_contain", []):
        if frag.lower() not in notes.lower():
            problems.append(f"notes lack {frag!r}")
    if result.get("error"):
        problems.append(f"error: {result['error']}")
    return problems


def ask_local(agent, question: str) -> dict:
    r = agent.ask(question)
    return {"answer": r.answer, "route": r.route, "plan": r.plan, "provenance": r.provenance,
            "error": r.error, "guards": r.guards, "summary_mode": r.summary_mode,
            "prose_issues": r.prose_issues, "timing": r.timing}


def ask_http(url: str, question: str) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ask", data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json", "X-Institution": "golden-live-check"},
        method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["route"] = data.get("route") or ((data.get("plan") or {}).get("template") and "data") or None
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--url", help="ask a running server instead of the local engine")
    args = ap.parse_args()
    spec = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agent = None
    if not args.url:
        from explorer.agent import Agent
        agent = Agent()
    report, hard_failures = [], 0
    for case in spec["cases"]:
        if args.only and case["name"] != args.only:
            continue
        passes = 0
        runs = []
        for _ in range(args.repeats):
            t0 = time.time()
            try:
                result = ask_http(args.url, case["question"]) if args.url else ask_local(agent, case["question"])
            except Exception as e:  # noqa: BLE001
                result = {"answer": "", "error": f"{type(e).__name__}: {e}"}
            problems = judge(case, result)
            passes += not problems
            runs.append({"problems": problems, "answer": result.get("answer"), "route": result.get("route"),
                         "template": (result.get("plan") or {}).get("template"),
                         "plan": {k: v for k, v in (result.get("plan") or {}).items() if not k.startswith("_")},
                         "guards": result.get("guards"), "summary_mode": result.get("summary_mode"),
                         "prose_issues": result.get("prose_issues"), "seconds": round(time.time() - t0, 1)})
            flag = "ok  " if not problems else "FAIL"
            print(f"{flag} {case['name']} ({runs[-1]['seconds']}s, {result.get('route')}"
                  f"{', ' + str(result.get('summary_mode')) if result.get('summary_mode') else ''})"
                  + ("" if not problems else ": " + "; ".join(problems)))
        if passes == 0:
            hard_failures += 1
        report.append({"name": case["name"], "question": case["question"], "passes": passes,
                       "repeats": args.repeats, "runs": runs})
    out = ROOT / "data" / f"golden_live_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    total = len(report)
    print(f"\n{total - hard_failures}/{total} cases passed at least once; {hard_failures} failed every time."
          f" Details: {out}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
