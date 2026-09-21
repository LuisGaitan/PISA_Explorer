"""Golden regression set, tier 1: plan -> table, no language model.

tests/golden/plans.json holds analysis plans exactly as the planner emits
them and the numbers the engine must reproduce (several verified against
OECD publications). Any drift — an estimate, an SE, a rank, a note's wording,
a blank cell that should be blank — fails here before a deploy. Needs the
local DuckDB; skipped in CI. Re-record after an intentional method change
with scripts/golden_record.py --update.
"""

import copy
import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

GOLDEN = Path(__file__).resolve().parent / "golden" / "plans.json"
SPEC = json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent():
    from explorer.db import DB_PATH
    if not DB_PATH.exists():
        pytest.skip("no local DuckDB (CI)")
    from explorer.agent import Agent
    return Agent()


def _row(table, match: dict):
    mask = None
    for col, val in match.items():
        assert col in table.columns, f"column {col!r} missing from result"
        m = table[col].astype(str) == str(val)
        mask = m if mask is None else (mask & m)
    rows = table[mask]
    assert len(rows) == 1, f"{match} matched {len(rows)} rows"
    return rows.iloc[0]


def run_case(agent, case: dict, tolerance: float) -> list[str]:
    """Problems found for one case (empty = pass). Shared with the recorder."""
    from explorer.agent import CoverageError
    expect = case["expect"]
    problems = []
    try:
        # deep copy: execute() edits nested plan fields (overrides), and the
        # case must stay as written
        table, prov = agent.execute(copy.deepcopy(case["plan"]))
    except CoverageError as e:
        if "error_contains" not in expect:
            return [f"unexpected CoverageError: {e}"]
        for frag in expect["error_contains"]:
            if frag not in str(e):
                problems.append(f"error text lacks {frag!r}")
        return problems
    except Exception as e:  # noqa: BLE001
        return [f"{type(e).__name__}: {e}"]
    if "error_contains" in expect:
        return ["expected an error, got a table"]
    if "rows" in expect and len(table) != expect["rows"]:
        problems.append(f"rows {len(table)} != {expect['rows']}")
    for col in expect.get("columns_absent", []):
        if col in table.columns:
            problems.append(f"column {col} should be absent")
    for cell in expect.get("cells", []):
        try:
            row = _row(table, cell["match"])
        except AssertionError as e:
            problems.append(str(e))
            continue
        for col, want in cell.items():
            if col == "match":
                continue
            have = row.get(col)
            if want is None:
                if have is not None and not (isinstance(have, float) and math.isnan(have)):
                    problems.append(f"{cell['match']} {col}: expected blank, got {have}")
                continue
            if have is None or (isinstance(have, float) and math.isnan(have)):
                problems.append(f"{cell['match']} {col}: expected {want}, got blank")
            elif abs(float(have) - float(want)) > tolerance:
                problems.append(f"{cell['match']} {col}: expected {want}, got {float(have):.4f}")
    for col, frag in (expect.get("text_contains") or {}).items():
        if not table[col].astype(str).str.contains(frag, regex=False).any():
            problems.append(f"no {col} cell contains {frag!r}")
    notes = " ".join(prov.get("notes") or [])
    for frag in expect.get("notes_contain", []):
        if frag not in notes:
            problems.append(f"notes lack {frag!r}")
    for frag in expect.get("notes_forbid", []):
        if frag in notes:
            problems.append(f"notes must not say {frag!r}")
    return problems


@pytest.mark.parametrize("case", SPEC["cases"], ids=[c["name"] for c in SPEC["cases"]])
def test_golden_plan(agent, case):
    problems = run_case(agent, case, SPEC.get("tolerance", 0.02))
    assert not problems, f"{case['name']}: " + "; ".join(problems)
