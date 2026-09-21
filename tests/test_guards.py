"""Offline tests for the exactness layer added 2026-09-16: standard variables
per construct, app-authored fact sentences, the prose check that grounds a
model's draft in the result, the build/method stamp, and guard telemetry."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import standards, summary as summ, version   # noqa: E402

NAMES = {"FIN": "Finland", "USA": "United States", "KSV": "Kosovo", "BRA": "Brazil",
         "QCI": "B-S-J-Z (China)", "SGP": "Singapore", "RWA": "Rwanda"}
CODES = sorted(NAMES)


def _mentioned(text: str) -> list[str]:
    import re
    low = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    caps = " " + re.sub(r"[^A-Za-z0-9 ]", " ", text) + " "
    out = []
    for code, name in NAMES.items():
        base = re.sub(r"[^a-z0-9 ]", " ", name.split(" (")[0].lower())
        if f" {code} " in caps or f" {base} " in low:
            out.append(code)
    return out


def _label(expr):
    return {"PV{pv}MATH": "Mathematics score", "PV{pv}SCIE": "Science score",
            "ESCS": "Index of economic, social and cultural status (ESCS)"}.get(expr, str(expr))


# ---------- standards ----------

def test_standards_match_constructs_and_swap_look_alikes():
    from explorer.agent import Agent
    m = {s.construct for s in standards.matching("public vs private school gap in Mexico")}
    assert "public vs private school" in m
    assert not [s for s in standards.matching("government-dependent private schools in Chile")
                if s.construct == "public vs private school"]
    assert {s.construct for s in standards.matching("socioeconomic status and math")} >= {"socio-economic status"}
    assert {s.construct for s in standards.matching("gender gap")} >= {"gender"}
    assert standards.matching("mean science in Finland") == []
    # the planner prompt block names the variable per cycle and the reason
    block = standards.prompt_block(standards.matching("bullying in 2022"))
    assert "2018: BEINGBULLIED" in block and "2022: BULLIED" in block
    # a look-alike is swapped for the standard, and the switch is recorded
    agent = Agent.__new__(Agent)
    plan = {"template": "quartile_gap", "measure": "PV{pv}MATH", "quart_variable": "HOMEPOS",
            "cycles": ["2025"], "where": "CNT = 'BRA'"}
    agent._apply_standards(plan, "socio-economic gap in math in Brazil")
    assert plan["quart_variable"] == "ESCS" and plan["_standardized"]
    assert "standard:swap:socio-economic" in agent._fired
    # but not when the user named the look-alike themselves
    plan = {"template": "quartile_gap", "measure": "PV{pv}MATH", "quart_variable": "HOMEPOS",
            "cycles": ["2025"]}
    agent._apply_standards(plan, "socio-economic gap by HOMEPOS in Brazil")
    assert plan["quart_variable"] == "HOMEPOS"
    # school-type swap also moves the plan onto the joined view
    plan = {"template": "gap", "measure": "PV{pv}SCIE", "group_col": "PRIVATESCH",
            "minuend": 2, "subtrahend": 1, "cycles": ["2025"], "instrument": "stu_qqq"}
    agent._apply_standards(plan, "public vs private schools in Mexico")
    assert plan["group_col"] == "SC013Q01TA" and plan["instrument"] == "stu_sch"
    assert plan["_school_type_switched"]


def test_every_standard_variable_exists_in_the_catalog_for_its_cycles():
    from explorer.db import CATALOG_DIR
    if not (CATALOG_DIR / "variables.parquet").exists():
        pytest.skip("no local catalog (CI)")
    cat = pd.read_parquet(CATALOG_DIR / "variables.parquet")
    for s in standards.STANDARDS:
        for cycle, var in s.variables.items():
            hit = cat[(cat.variable == var) & (cat.cycle == cycle) & (cat.instrument == s.instrument)]
            assert not hit.empty, f"{s.construct}: {var} not in {s.instrument}_{cycle}"
        for var in s.companions:
            assert (cat.variable == var).any(), f"{s.construct}: companion {var} unknown"


# ---------- app-authored facts ----------

def test_fact_sentences_state_levels_changes_ranks_and_verdicts():
    single = pd.DataFrame({"CNT": ["FIN", "USA"], "estimate": [504.0, 480.5],
                           "se": [2.1, 3.0], "n_pv": 10, "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}SCIE", "cycles": ["2025"]}
    facts = summ.fact_sentences(single, plan, {}, NAMES, _label, focus_codes=["FIN", "USA"])
    text = " ".join(facts)
    assert "Finland (FIN), PISA 2025: 504.0 (SE 2.10)" in text
    assert "Finland (FIN) minus United States (USA): 23.5 (SE 3.66, independent samples), statistically significant" in text

    trend = pd.DataFrame({"CNT": ["BRA"], "estimate_2022": [403.5], "se_2022": [2.0],
                          "estimate_2025": [409.0], "se_2025": [1.9],
                          "change": [5.54], "se_change": [4.15]})
    facts = summ.fact_sentences(trend, plan, {}, NAMES, _label)
    assert facts == ["Mean of Science score — Brazil (BRA): 2022: 403.5 (SE 2.00) → 2025: 409.0 "
                     "(SE 1.90); change 2025 minus 2022 = 5.54 (SE 4.15), not statistically significant."]

    ranked = pd.DataFrame({"rank": list(range(1, 11)), "CNT": [f"C{i}" for i in range(9)] + ["KSV"],
                           "estimate": np.linspace(560, 357, 10), "se": 1.5, "n_pv": 10, "cycle": "2025"})
    facts = summ.fact_sentences(ranked, plan, {}, NAMES, _label, focus_codes=["KSV"])
    assert facts[0].startswith("The table has 10 rows (10 ranked economies")
    assert any("Kosovo (KSV), PISA 2025: rank 10 of 10, 357.0 (SE 1.50)" in f for f in facts)

    blank = trend.copy(); blank[["change", "se_change"]] = np.nan
    facts = summ.fact_sentences(blank, {**plan, "measure": "ESCS"}, {}, NAMES, _label)
    assert "not computed (see notes)" in facts[0]

    gap = pd.DataFrame({"contrast": ["MALE: 1 - 0"], "CNT": ["USA"], "estimate": [23.82],
                        "se": [2.9], "n_pv": 10, "cycle": "2025"})
    facts = summ.fact_sentences(gap, {"template": "gap", "measure": "PV{pv}MATH"}, {}, NAMES, _label)
    assert facts == ["Difference in Mathematics score — United States (USA), MALE: 1 - 0, "
                     "PISA 2025: 23.8 (SE 2.90), statistically significant."]


def test_fact_sentences_compare_a_member_with_its_group_average():
    t = pd.DataFrame({"CNT": ["FRA", "EU avg"], "estimate": [456.34, 454.11], "se": [3.24, 0.52],
                      "n_pv": 10, "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}READ", "cycles": ["2025"],
            "_benchmark_members": {"2025": {"EU avg": ["FRA", "DEU"] + [f"X{i}" for i in range(24)]}}}
    facts = summ.fact_sentences(t, plan, {}, {"FRA": "France"}, _label, focus_codes=["FRA"])
    line = [f for f in facts if "minus EU avg" in f][0]
    se = np.sqrt(3.24 ** 2 * (1 - 2 / 26) + 0.52 ** 2)
    assert f"France (FRA) minus EU avg: 2.23 (SE {se:.2f}, the economy's share of the 26-member average accounted for), not statistically significant." == line


def test_number_extraction_handles_commas_and_thousands():
    got = [(v, d) for _, v, d in summ._numbers_in("49.5 (SE 2.99, independent samples); 6,622 students; 1.96; 15-year-olds")]
    assert (2.99, 2) in got and (6622.0, 0) in got and (49.5, 1) in got and (1.96, 2) in got
    # the app's own pairwise statement makes its numbers and verdict legitimate
    t = pd.DataFrame({"CNT": ["QCI", "SGP"], "estimate": [612.43, 562.9], "se": [2.55, 1.57], "cycle": "2025"})
    prov = {"notes": [], "method": "", "sample": [],
            "facts": ["B-S-J-Z (China) (QCI) minus Singapore (SGP): 49.5 (SE 2.99, independent samples), statistically significant."]}
    issues = summ.check_prose("B-S-J-Z (China) (QCI) scored 49.5 points more than Singapore (SGP) (SE 2.99), "
                              "a statistically significant difference.", t, prov,
                              {"template": "weighted_mean"}, "", _mentioned, {"QCI", "SGP"})
    assert issues == []


def test_share_measures_keep_non_respondents_out_of_the_denominator(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent._fired = []
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
         "label": ["Parent expects child to work in engineering"], "var_type": ["double"],
         "value_labels": [json.dumps({"1.0": "Yes", "2.0": "No"})]}))
    raw = "CASE WHEN PA032Q03TA = 1.0 THEN 100.0 ELSE 0.0 END"
    assert agent._null_safe_share(raw) == f"CASE WHEN (PA032Q03TA) IS NULL THEN NULL ELSE ({raw}) END"
    assert agent._null_safe_share("PV{pv}MATH") == "PV{pv}MATH"
    assert Agent._measure_label(raw) == "% with Parent expects child to work in engineering (PA032Q03TA) = 1 (Yes)"
    assert Agent._measure_label("CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END") == "% below Level 2 in mathematics"


def test_comparability_and_item_questions_are_intercepted():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent._fired = []
    assert "link error" in agent._intercept("Are El Salvador's 2025 results comparable with Sweden 2022?")
    assert agent._intercept("compare Chile and Peru in science") is None
    assert Agent.ITEM_ASK_WORDS.search("I believe 2025 had some questions about how students evaluate information")
    assert not Agent.ITEM_ASK_WORDS.search("mean science score in Chile")


def test_fraction_shares_are_rescaled_and_multi_variable_cases_null_guarded(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    known = {"ST127Q01TA", "ST127Q02TA"}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2018"], "cycle": ["2018"], "label": [var],
         "var_type": ["double"], "value_labels": [None]}) if var in known else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    raw = "CASE WHEN ST127Q01TA = 2 OR ST127Q02TA = 2 THEN 1 WHEN ST127Q01TA IN (1, 3) THEN 0 ELSE 0 END"
    out = agent._null_safe_share(raw)
    assert out.startswith("CASE WHEN (COALESCE(ST127Q01TA, ST127Q02TA)) IS NULL THEN NULL ELSE (")
    assert "THEN 100.0" in out and "THEN 0.0" in out and "THEN 1 " not in out
    assert agent._null_safe_share("CASE WHEN ST127Q01TA = 2 THEN 100.0 ELSE NULL END") == "CASE WHEN ST127Q01TA = 2 THEN 100.0 ELSE NULL END"


def test_raw_sql_returns_aggregates_only():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    with pytest.raises(ValueError, match="aggregate"):
        agent._run_raw_sql("SELECT CNT, PV1MATH FROM stu_qqq_2025 LIMIT 5")
    with pytest.raises(ValueError, match="identifiers"):
        agent._run_raw_sql("SELECT CNTSCHID, avg(PV1MATH) FROM stu_qqq_2025 GROUP BY CNTSCHID")
    with pytest.raises(ValueError, match="aggregate"):
        agent._run_raw_sql("SELECT * FROM stu_qqq_2025")


def test_small_cells_are_suppressed_and_regression_names_collinear_terms():
    from explorer.agent import Agent
    from explorer.analysis import regression
    from explorer.estimator import ALL_WEIGHTS
    agent = Agent.__new__(Agent)
    agent._fired = []
    res = pd.DataFrame({"CNT": ["A", "B", "C"], "estimate": [400.0, 410.0, 420.0], "se": [2.0, 9.0, 3.0],
                        "n_pv": 10, "n": [500, 12, 65], "n_schools": [40, 3, 2], "wcov": [1.0, 1.0, 0.6]})
    plan = {}
    out = agent._suppress_small_cells(res, "2025", plan)
    assert "n" not in out.columns and np.isnan(out.loc[1, "estimate"]) and np.isnan(out.loc[1, "se"])
    assert np.isnan(out.loc[2, "estimate"])                      # 65 students but only 2 schools
    assert plan["_suppressed"] == {"2025": {"students": 1, "schools": 1}} and out.loc[0, "estimate"] == 400.0
    assert plan["_suppressed_rows"]["2025"] == [{"CNT": "B"}, {"CNT": "C"}]
    low = pd.DataFrame({"CNT": ["D"], "estimate": [390.0], "se": [2.0], "n_pv": 10, "n": [900], "n_schools": [50], "wcov": [0.57]})
    plan = {}
    agent._suppress_small_cells(low, "2025", plan)
    assert plan["_low_coverage"] == {"2025": [("D", 43)]}

    class FakeCon:
        def __init__(self, df): self.df = df
        def sql(self, q): return self
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame({w: rng.uniform(0.5, 1.5, n) for w in ALL_WEIGHTS})
    frame["y_1"] = rng.normal(500, 50, n); frame["x_1"] = rng.normal(0, 1, n); frame["x_2"] = 1.0
    import explorer.analysis as an
    orig = an.fetch_frame
    an.fetch_frame = lambda *a, **k: frame
    try:
        with pytest.raises(ValueError, match="single value"):
            regression(None, "t", "y", ["ESCS", "CASE WHEN MALE = 1 THEN 1 ELSE 0 END"], names=["ESCS", "male"])
    finally:
        an.fetch_frame = orig


def test_strata_hits_find_rare_labels_only():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent._strata_cache = [("2025", "KAZ21", "Intellectual schools", "intellectual schools"),
                           ("2025", "KAZ01", "General/Astana city", "general/astana city"),
                           ("2018", "KAZ0101", "KAZ - stratum 01: non-intellectual / Astana city",
                            "kaz - stratum 01: non-intellectual / astana city"),
                           ("2018", "MEX9797", "Undisclosed STRATUM - Mexico", "undisclosed stratum - mexico"),
                           ("2018", "ALB0101", "ALB - stratum 01: Urban / North / Public", "alb - stratum 01: urban / north / public")]
    agent.economy_names = {"MEX": "Mexico"}
    hits = agent._strata_hits("what is the position of Nazarbayev Intellectual schools in math")
    assert hits == [("2025", "KAZ21", "Intellectual schools")]          # not the "non-intellectual" strata
    assert agent._strata_hits("public schools in urban areas") == []
    assert agent._strata_hits("students in Mexico who repeated a grade") == []
    from explorer import regions
    assert regions.non_pisa_named("How many students in India repeated a grade?") == ["India"]
    assert regions.non_pisa_named("mean science in Indonesia") == []
    block = agent._strata_block(hits)
    assert "STRATUM = 'KAZ21'" in block and "economy KAZ" in block


def test_count_intercept_does_not_hijack_behaviour_questions():
    from explorer.agent import Agent
    C, O = Agent.COUNT_WORDS, Agent.OTHER_STAT_WORDS
    q = "How many students in Japan use AI chatbots for schoolwork?"
    assert C.search(q) and O.search(q)                      # goes to the planner
    assert C.search("How many students were tested in Japan?") and not O.search("How many students were tested in Japan?")


# ---------- the prose check ----------

def _check(text, table, plan, question="", notes=(), lang="English"):
    prov = {"notes": list(notes), "method": "Weighted mean.", "sample": [], "facts": []}
    allowed = set(table["CNT"]) if "CNT" in table.columns else set()
    allowed |= set(_mentioned(question)) | set(_mentioned(" ".join(notes)))   # as Agent._allowed_codes does
    return summ.check_prose(text, table, prov, plan, question, _mentioned, allowed, language=lang)


def test_prose_check_accepts_grounded_text_and_rejects_invented_numbers():
    t = pd.DataFrame({"CNT": ["KSV"], "estimate": [357.04], "se": [1.31], "rank": [83], "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}SCIE"}
    ok = "Kosovo (KSV) scored 357.0 points in science in 2025 (SE 1.3), ranking 83rd of 90 economies."
    assert _check(ok, t, plan, "where is kosovo ranked in science in 2025? 90 economies") == []
    assert _check("Kosovo scored 357 points (SE 1.3).", t, plan) == []          # rounding to 0 dp
    bad = _check("Kosovo scored 372.0 points (SE 1.3).", t, plan)
    assert bad and bad[0].startswith("numbers not in the result: 372.0")
    # a difference the model computed itself is not in the result
    t2 = pd.DataFrame({"CNT": ["FIN", "USA"], "estimate": [504.0, 480.5], "se": [2.1, 3.0]})
    assert "numbers not in the result: 23.5" in _check("Finland leads by 23.5 points.", t2, plan)[0]


def test_prose_check_rejects_foreign_economies_and_bad_verdicts():
    t = pd.DataFrame({"CNT": ["FIN"], "estimate": [504.0], "se": [2.1], "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}SCIE"}
    issues = _check("Finland scored 504.0 (SE 2.1), ahead of Singapore.", t, plan)
    assert any("names economies not in the result: SGP" in i for i in issues)
    # an economy the question names is allowed even if it has no row
    assert _check("Finland scored 504.0 (SE 2.1); Singapore has no row here.", t, plan,
                  question="compare Finland and Singapore") == []
    # a significance claim needs a verdict in the table
    issues = _check("Finland's 504.0 (SE 2.1) is significantly higher.", t, plan)
    assert any("claims 'statistically significant'" in i for i in issues)
    gap = pd.DataFrame({"CNT": ["BRA"], "estimate_2022": [403.5], "se_2022": [2.0], "estimate_2025": [409.0],
                        "se_2025": [1.9], "change": [5.54], "se_change": [4.15]})
    plan = {"template": "weighted_mean", "measure": "PV{pv}SCIE", "cycles": ["2022", "2025"]}
    assert _check("Brazil rose from 403.5 to 409.0, a change of 5.5 (SE 4.2) that is not statistically significant.",
                  gap, plan) == []
    issues = _check("Brazil's improvement of 5.5 points (SE 4.2) is statistically significant.", gap, plan)
    assert any("claims 'statistically significant'" in i for i in issues)


def test_prose_check_rejects_forbidden_claims_unless_the_notes_make_them():
    t = pd.DataFrame({"CNT": ["RWA"], "estimate": [317.3], "se": [2.0], "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}SCIE"}
    assert "claims an economy did not participate" in _check("Rwanda did not participate.", t, plan)
    assert "claims a variable was not collected / administered" in _check(
        "Well-being was not collected for Rwanda.", t, plan)
    assert "describes results as projections" in _check("These 2025 figures are projected.", t, plan)
    q = pd.DataFrame({"CNT": ["QCI"], "estimate": [612.4], "se": [2.5], "cycle": "2025"})
    assert "calls B-S-J-Z (China) Shanghai" in _check("Shanghai scored 612.4 (SE 2.5).", q, plan)
    assert _check("B-S-J-Z (China), which includes Shanghai, scored 612.4 (SE 2.5).", q, plan) == []
    assert "presents the app as an OECD product" in _check("This tool was developed by the OECD.", t, plan)
    # the same claim is fine when the app's notes say it
    assert _check("Rwanda did not take part in PISA 2018.", t, plan,
                  notes=["PISA 2018: Rwanda (RWA) did not take part in that cycle."]) == []
    # a cause attributed to a change is rejected in a "why" question
    trend = pd.DataFrame({"CNT": ["BRA"], "estimate_2022": [403.5], "se_2022": [2.0],
                          "estimate_2025": [409.0], "se_2025": [1.9], "change": [5.54], "se_change": [4.15]})
    issues = _check("Brazil rose 5.5 points (SE 4.2), due to curriculum reform.", trend, plan,
                    question="why did Brazil improve?")
    assert "attributes a cause (PISA cannot establish causes)" in issues


# ---------- stamp and telemetry ----------

def test_version_stamp_and_guard_telemetry():
    assert version.stamp().startswith("build ") and f"method v{version.METHOD_VERSION}" in version.stamp()
    from explorer.agent import Agent, AgentResult
    agent = Agent.__new__(Agent)
    agent._fire("intercept:link_error"); agent._fire("intercept:link_error"); agent._fire("hook:x")
    assert agent._fired == ["intercept:link_error", "hook:x"]
    r = AgentResult("q", "a", guards=list(agent._fired), summary_mode="llm-retry",
                    prose_issues=["numbers not in the result: 1.5"])
    a = r.analytics()
    assert a["guards"] == ["intercept:link_error", "hook:x"] and a["summary_mode"] == "llm-retry"
    assert a["prose_issues"] == ["numbers not in the result: 1.5"] and a["build"] == version.BUILD


def test_translation_that_changes_a_number_is_discarded(monkeypatch):
    from explorer import agent as agent_mod
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent._fired = []
    monkeypatch.setattr(agent_mod, "generate", lambda *a, **k: "Kosovo obtuvo 375.0 puntos (EE 1.3).")
    assert agent._localize("Kosovo scored 357.0 points (SE 1.3).", "Spanish") == "Kosovo scored 357.0 points (SE 1.3)."
    assert "localize:numbers_changed" in agent._fired
    monkeypatch.setattr(agent_mod, "generate", lambda *a, **k: "Kosovo obtuvo 357.0 puntos (EE 1.3).")
    assert agent._localize("Kosovo scored 357.0 points (SE 1.3).", "Spanish").startswith("Kosovo obtuvo")
