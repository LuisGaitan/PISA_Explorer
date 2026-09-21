"""Check or re-record the golden plan set (tests/golden/plans.json).

    python scripts/golden_record.py            # run every case, report drift
    python scripts/golden_record.py --update   # rewrite expected values from
                                               # the current engine (do this
                                               # ONLY after an intentional
                                               # method change, and bump
                                               # METHOD_VERSION)
    python scripts/golden_record.py --only kosovo_science_rank_2025

Numbers are recorded to 2 decimals; notes_contain / notes_forbid /
error_contains / columns_absent are kept as written by hand.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

GOLDEN = ROOT / "tests" / "golden" / "plans.json"


def record(agent, case: dict) -> dict:
    from explorer.agent import CoverageError
    expect = dict(case.get("expect") or {})
    try:
        plan = copy.deepcopy(case["plan"])                          # never mutate the case
        if plan.get("_apply_strata"):
            agent._apply_strata(plan, str(plan.get("_question") or ""))
        table, _prov = agent.execute(plan)
    except CoverageError as e:
        expect.pop("cells", None); expect.pop("rows", None)
        expect.setdefault("error_contains", [str(e)[:80]])
        return expect
    expect.pop("error_contains", None)
    expect["rows"] = int(len(table))
    cells = []
    for cell in expect.get("cells") or []:
        match = cell["match"]
        mask = None
        for col, val in match.items():
            m = table[col].astype(str) == str(val)
            mask = m if mask is None else (mask & m)
        rows = table[mask]
        if len(rows) != 1:
            print(f"  {case['name']}: {match} matched {len(rows)} rows — kept as is")
            cells.append(cell)
            continue
        row = rows.iloc[0]
        new = {"match": match}
        for col in cell:
            if col == "match":
                continue
            v = row.get(col)
            new[col] = None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 2)
        cells.append(new)
    expect["cells"] = cells
    return expect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--only")
    args = ap.parse_args()
    spec = json.loads(GOLDEN.read_text(encoding="utf-8"))
    from explorer.agent import Agent
    from test_golden_plans import run_case
    agent = Agent()
    failed = 0
    for case in spec["cases"]:
        if args.only and case["name"] != args.only:
            continue
        if args.update:
            case["expect"] = record(agent, case)
            print(f"recorded {case['name']}")
            continue
        problems = run_case(agent, case, spec.get("tolerance", 0.02))
        status = "ok " if not problems else "FAIL"
        print(f"{status} {case['name']}" + ("" if not problems else ": " + "; ".join(problems)))
        failed += bool(problems)
    if args.update:
        GOLDEN.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {GOLDEN}")
    else:
        print(f"{len(spec['cases']) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
