"""Offline tests for the persona red-team round (2026-09-20): plan-shape
repairs, stratum rows, opposite-group and SE checks on prose, pairwise and
tie statements, level phrases, North Korea, rewrite follow-ups, test-mode
words. No language model, no database."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import regions, summary as summ   # noqa: E402
from explorer.agent import Agent               # noqa: E402

NAMES = {"FIN": "Finland", "EST": "Estonia", "JPN": "Japan", "KOR": "Korea", "HKG": "Hong Kong (China)",
         "TAP": "Chinese Taipei", "POL": "Poland", "DEU": "Germany"}


def _agent() -> Agent:
    a = Agent.__new__(Agent)
    a._fired = []
    a.present = {"2025": set(NAMES), "2022": set(NAMES), "2018": set(NAMES)}
    a.economy_names = dict(NAMES)
    return a


def _mentioned(text):
    import re
    caps = " " + re.sub(r"[^A-Za-z0-9 ]", " ", text) + " "
    low = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    return [c for c, n in NAMES.items() if f" {c} " in caps or f" {n.split(' (')[0].lower()} " in low]


# ---------- plan shape ----------

def test_plan_shape_placeholder_by_economy_gap_and_pooled_mean(monkeypatch):
    a = _agent()
    monkeypatch.setattr(a, "_auto_instrument", lambda plan: None)
    plan = {"template": "gap", "by": ["CNT", "group_col"], "group_col": "ST004D01T", "minuend": 2, "subtrahend": 1}
    a._normalize_plan_shape(plan, "gap")
    assert plan["by"] == ["CNT"] and plan["_by_placeholder"] == ["group_col"]
    plan = {"template": "gap", "by": ["CNT"], "group_col": "CNT", "minuend": "FIN", "subtrahend": "EST", "where": None}
    a._normalize_plan_shape(plan, "gap")
    assert plan["template"] == "weighted_mean" and plan["_pair"] == ["FIN", "EST"]
    assert plan["where"] == "CNT IN ('FIN', 'EST')" and plan["by"] == ["CNT"]
    plan = {"template": "weighted_mean", "by": [], "where": "CNT IN ('FIN', 'EST', 'JPN')"}
    a._normalize_plan_shape(plan, "weighted_mean")
    assert plan["by"] == ["CNT"] and plan["_pooled_to_rows"]
    assert sorted(plan["include_average_of"][0]) == ["EST", "FIN", "JPN"]
    plan = {"template": "weighted_mean", "by": None, "where": None, "include_average_of": ["FIN", "EST", "JPN"]}
    a._normalize_plan_shape(plan, "weighted_mean")
    assert plan["by"] == ["CNT"] and plan["where"] == "CNT IN ('EST', 'FIN', 'JPN')" and plan["_members_as_rows"]
    # a ranking with the OECD average beside it keeps every economy
    plan = {"template": "weighted_mean", "by": ["CNT"], "where": None, "include_oecd_average": True, "top_n": 10}
    a._normalize_plan_shape(plan, "weighted_mean")
    assert not plan.get("where") and not plan.get("_members_as_rows")


def test_case_expression_in_by_becomes_its_variable(monkeypatch):
    from explorer import catalog
    a = _agent()
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"], "label": [var],
         "var_type": ["double"], "value_labels": [None]}) if var in ("MALE", "ST004D01T") else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    plan = {"cycle_overrides": {"2025": {"by": ["CNT", "CASE WHEN MALE = 0 THEN 'Female' WHEN MALE = 1 THEN 'Male' END"]}}}
    assert a._plain_by(["CNT", "CASE WHEN ST004D01T = 1 THEN 'Female' ELSE 'Male' END"], plan) == ["CNT", "ST004D01T"]
    assert "hook:case_by_replaced" in a._fired


def test_named_ad_hoc_groups_get_their_own_average_rows():
    a = _agent()
    plan = {"include_average_of": [{"label": "Pacific Alliance", "members": ["CHL", "COL", "MEX", "PER"]},
                                   {"label": "Mercosur", "members": ["ARG", "BRA", "PRY", "URY"]}]}
    assert a._benchmarks(plan) == [("CHL", "COL", "MEX", "PER"), ("ARG", "BRA", "PRY", "URY")]
    a.present = {"2025": {"CHL", "COL", "MEX", "PER", "ARG", "BRA", "PRY", "URY"}}
    label, members = a._average_members(("ARG", "BRA", "PRY", "URY"), "2025", plan=plan)
    assert label == "Mercosur avg" and members == {"ARG", "BRA", "PRY", "URY"}


def test_null_safe_dummy_and_level_phrases(monkeypatch):
    from explorer import catalog
    a = _agent()
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2022"], "cycle": ["2022"], "label": [var],
         "var_type": ["double"], "value_labels": [None]}) if var == "IMMIG" else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    out = a._null_safe_dummy("CASE WHEN IMMIG = 2 THEN 1 ELSE 0 END")
    assert out == "CASE WHEN (IMMIG) IS NULL THEN NULL ELSE (CASE WHEN IMMIG = 2 THEN 1 ELSE 0 END) END"
    assert a._null_safe_dummy("ESCS") == "ESCS"
    plan = {"template": "weighted_mean", "measure": "CASE WHEN PV{pv}READ < 334.75 THEN 100.0 ELSE 0.0 END"}
    a._fix_level_phrases(plan, "how many students are at level 1 or below in reading?")
    assert "407.47" in plan["measure"] and plan["_level1_or_below"]
    plan = {"template": "weighted_mean", "measure": "CASE WHEN PV{pv}READ < 334.75 THEN 100.0 ELSE 0.0 END"}
    a._fix_level_phrases(plan, "share below level 1a in reading")
    assert "334.75" in plan["measure"]


# ---------- strata words, North Korea, regions ----------

def test_stratum_words_north_korea_and_arab_region():
    assert Agent.STRATA_EXCLUDE_WORDS.search("Kazakhstan math excluding the Nazarbayev Intellectual Schools stratum")
    assert Agent.STRATA_REST_WORDS.search("Tashkent city versus the rest of the country")
    assert not Agent.STRATA_REST_WORDS.search("what is the position of Nazarbayev Intellectual schools")
    assert regions.non_pisa_named("does north korea take part in pisa") == ["North Korea (DPRK)"]
    assert "north korea" in regions.NEGATIVE_ALIASES["KOR"]
    assert regions.canonical("arab countries") == "Arab countries"
    assert "MAR" in regions.REGIONS["Arab countries"] and "ISR" not in regions.REGIONS["Arab countries"]
    assert Agent.REGION_LIST_WORDS.search("give the results per region of Philippines")
    assert Agent.MODE_WORDS.search("did El Salvador take the test on paper or computer in 2022?")
    assert Agent.LINK_DATA_REQUEST.search("share at Level 6 in Singapore 2022 vs 2025, with the link error")
    assert Agent.REWRITE_WORDS.search("give me five bullets for my staff meeting")
    assert Agent.REWRITE_WORDS.search("explícamelo más sencillo, sin siglas")
    assert not Agent.REWRITE_WORDS.search("mean science score in Chile 2025")
    assert Agent.INDEX_QUESTION_WORDS.search("How is ESCS constructed in 2025 and which components changed?")
    assert not Agent.WHY_WORDS.search("with number of students and number of schools behind it")
    assert Agent.WHY_WORDS.search("what is behind the decline in Chile")


# ---------- statements and the prose check ----------

def test_facts_label_groups_pairs_ties_and_change_of_difference():
    t = pd.DataFrame({"CNT": ["POL", "POL"], "ST004D01T": [1.0, 2.0], "estimate": [450.95, float("nan")],
                      "se": [15.27, float("nan")], "cycle": "2022"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}MATH", "by": ["CNT", "ST004D01T"]}
    facts = summ.fact_sentences(t, plan, {}, NAMES, lambda m: "Mathematics score",
                                category_label=lambda c, v: f"{c} = {int(v)} ({'Female' if v == 1 else 'Male'})")
    assert any("ST004D01T = 1 (Female)" in f and "450.9" in f for f in facts)
    assert any("ST004D01T = 2 (Male)" in f and "no estimate" in f for f in facts)
    # ties and the rank range of a named economy
    t = pd.DataFrame({"rank": [1, 2, 3, 4, 5, 6], "CNT": ["TAP", "KOR", "JPN", "HKG", "EST", "FIN"],
                      "estimate": [546.2, 522.3, 525.4, 521.7, 507.9, 468.7],
                      "se": [3.1, 4.0, 4.2, 2.9, 2.3, 2.2], "cycle": "2025"})
    facts = summ.fact_sentences(t, {"template": "weighted_mean", "measure": "PV{pv}MATH", "sort_by": "estimate"},
                                {}, NAMES, lambda m: "Mathematics score", focus_codes=["JPN"])
    tie = next(f for f in facts if "not statistically different from Japan" in f)
    assert "Korea (KOR)" in tie and "Hong Kong (China) (HKG)" in tie and "rank 2 to rank 4" in tie
    # change of a difference across cycles = difference of the two changes
    t = pd.DataFrame({"CNT": ["FIN", "EST"], "estimate_2018": [520.08, 523.02], "se_2018": [2.31, 1.84],
                      "estimate_2025": [474.30, 499.24], "se_2025": [2.12, 2.53],
                      "change": [-45.78, -23.77], "se_change": [3.63, 3.63]})
    facts = summ.fact_sentences(t, {"template": "weighted_mean", "measure": "PV{pv}READ", "_pair": ["FIN", "EST"]},
                                {}, NAMES, lambda m: "Reading score")
    line = next(f for f in facts if f.startswith("Change 2025 minus 2018 in (Finland"))
    assert "-22.0" in line and "SE 4.43" in line and "statistically significant" in line
    # gap-in-gap: two economies' gender gaps compared
    t = pd.DataFrame({"CNT": ["FIN", "EST"], "contrast": ["male minus female"] * 2,
                      "estimate": [-44.69, -19.13], "se": [2.98, 3.79], "cycle": "2022"})
    facts = summ.fact_sentences(t, {"template": "gap", "measure": "PV{pv}READ"}, {}, NAMES, lambda m: "Reading score")
    assert any("Difference for Finland (FIN) minus for Estonia (EST)" in f and "-25.6" in f and "SE 4.82" in f
               for f in facts)


def test_prose_check_rejects_opposite_group_and_zero_se_and_partial_wording():
    t = pd.DataFrame({"CNT": ["POL", "POL"], "ST004D01T": [1.0, 2.0], "estimate": [450.95, float("nan")],
                      "se": [15.27, float("nan")], "cycle": "2022"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}MATH"}
    prov = {"notes": [], "method": "", "sample": [],
            "facts": ["Mean of Mathematics score — Poland (POL), ST004D01T = 1 (Female), PISA 2022: 450.9 (SE 15.3).",
                      "Mean of Mathematics score — Poland (POL), ST004D01T = 2 (Male), PISA 2022: no estimate."]}
    bad = "First-generation immigrant male students in Poland scored 450.9 points (SE 15.3)."
    issues = summ.check_prose(bad, t, prov, plan, "split by gender", _mentioned, {"POL"})
    assert any("attributes 450.9 to the male group" in i for i in issues)
    good = "Girls in Poland scored 450.9 points (SE 15.3); the boys' estimate is suppressed."
    assert summ.check_prose(good, t, prov, plan, "split by gender", _mentioned, {"POL"}) == []
    # a share who DISAGREE reported as agreeing
    prov = {"notes": [], "method": "", "sample": [],
            "facts": ["Mean of % with intelligence item (ST263Q02JA) in (1, 2) [1 = Strongly disagree; 2 = Disagree] "
                      "— Estonia (EST), PISA 2022: 73.9 (SE 0.69)."]}
    t = pd.DataFrame({"CNT": ["EST"], "estimate": [73.86], "se": [0.69], "cycle": "2022"})
    issues = summ.check_prose("73.9% of students in Estonia agreed or strongly agreed.", t, prov, plan,
                              "share who disagree", _mentioned, {"EST"})
    assert any("attributes 73.9 to the agree group" in i for i in issues)
    # SE rounded away
    t = pd.DataFrame({"CNT": ["FIN"], "estimate": [-1.3], "se": [0.04], "cycle": "2025"})
    prov = {"notes": [], "method": "", "sample": [], "facts": ["Mean — Finland (FIN), PISA 2025: -1.30 (SE 0.04)."]}
    issues = summ.check_prose("Finland's ESCS was -1.3 (SE 0.0).", t, prov, plan, "", _mentioned, {"FIN"})
    assert any("rounds a standard error to 0.0" in i for i in issues)
    # partial-table wording and "cannot compute the difference"
    issues = summ.check_prose("Based on the available data, Finland has the highest share; a complete list "
                              "would require the full table.", t, prov, plan, "", _mentioned, {"FIN"})
    assert any("partial or incomplete" in i for i in issues)
    issues = summ.check_prose("The system cannot directly compute the difference between these two gaps.",
                              t, prov, plan, "", _mentioned, {"FIN"})
    assert any("cannot compute a difference" in i for i in issues)


def test_link_error_answer_states_the_v5_rules_and_last_mode():
    a = _agent()
    text = a._link_error_answer("share")
    assert "share-specific link error" in text and "proficiency-level share" in text
    assert "cancels" in a._link_error_answer("cancels")
    assert "sampling-only" not in a._link_error_answer(None)


def test_rewrite_of_previous_answer_keeps_its_numbers(monkeypatch):
    from explorer import agent as agent_mod
    a = _agent()
    history = [{"question": "Ireland math 2025", "answer": "Ireland scored 492.3 points (SE 2.1) in 2025."}]
    monkeypatch.setattr(agent_mod, "generate", lambda *x, **k: "- Ireland: 492.3 points\n- margin of error 2.1")
    out = a._rewrite_previous("give me bullets", history, None)
    assert out.startswith("- Ireland: 492.3")
    monkeypatch.setattr(agent_mod, "generate", lambda *x, **k: "- Ireland: 499.0 points")
    assert a._rewrite_previous("give me bullets", history, None) == history[0]["answer"]
    assert "rewrite:numbers_changed" in a._fired
