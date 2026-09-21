"""Checks that need neither the OECD data nor an API key: the cross-cycle
trend arithmetic, the Fay-BRR/Rubin combination on a synthetic replicate
frame, the institution-name normalizer, and the SQL guards."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer.analysis import trend                     # noqa: E402
from explorer.estimator import ALL_WEIGHTS, combine     # noqa: E402


def _result(cnt, est, se):
    return pd.DataFrame({"CNT": cnt, "estimate": est, "se": se, "n_pv": 10})


def test_trend_three_cycles_change_is_last_minus_first():
    r18 = _result(["FIN", "USA"], [520.0, 505.0], [2.3, 3.6])
    r22 = _result(["FIN", "USA", "KHM"], [490.0, 504.0, 380.0], [2.3, 4.3, 3.0])
    r25 = _result(["FIN", "KHM"], [480.0, 390.0], [2.0, 3.1])
    out = trend({"2018": r18, "2022": r22, "2025": r25}, by=("CNT",)).set_index("CNT")
    assert list(out.columns) == ["estimate_2018", "se_2018", "estimate_2022", "se_2022",
                                 "estimate_2025", "se_2025", "change", "se_change"]
    assert out.loc["FIN", "change"] == pytest.approx(-40.0)
    assert out.loc["FIN", "se_change"] == pytest.approx(np.sqrt(2.3**2 + 2.0**2))
    # an economy missing from a cycle keeps its row (outer join), with NaN change
    assert np.isnan(out.loc["KHM", "estimate_2018"])
    assert np.isnan(out.loc["KHM", "change"])
    assert np.isnan(out.loc["USA", "estimate_2025"])


def test_trend_legacy_two_frame_call_still_works():
    r18 = _result(["FIN"], [520.0], [2.0])
    r22 = _result(["FIN"], [490.0], [2.0])
    out = trend(r18, r22, by=("CNT",))
    assert out.change.iloc[0] == pytest.approx(-30.0)
    assert "estimate_2025" not in out.columns


def test_combine_rubin_and_fay_brr_variance():
    """One group, 2 PVs, 80 replicates: sampling variance = sum(sq)/20,
    imputation variance = var(PV means), total = mean(U) + (1 + 1/M) * B."""
    rows = []
    rng = np.random.default_rng(0)
    for pv, main in ((1, 500.0), (2, 502.0)):
        rows.append({"pv": pv, "rep": 0, "value": main})
        for rep in range(1, len(ALL_WEIGHTS)):
            rows.append({"pv": pv, "rep": rep, "value": main + rng.normal(0, 4)})
    reps = pd.DataFrame(rows)
    out = combine(reps)
    assert out.estimate.iloc[0] == pytest.approx(501.0)
    u = []
    for pv, main in ((1, 500.0), (2, 502.0)):
        r = reps[(reps.pv == pv) & (reps.rep > 0)].value
        u.append(((r - main) ** 2).sum() / 20)
    b = np.var([500.0, 502.0], ddof=1)
    assert out.se.iloc[0] == pytest.approx(np.sqrt(np.mean(u) + 1.5 * b))
    assert int(out.n_pv.iloc[0]) == 2


def test_clean_institution_normalizes_and_rejects():
    from explorer.app import clean_institution
    assert clean_institution("  Universit%C3%A4t   Wien ") == "Universität Wien"
    assert clean_institution("Penn GSE") == "Penn GSE"
    assert clean_institution("X") is None
    assert clean_institution("") is None
    assert len(clean_institution("A" * 200)) == 80


def test_sql_guards_reject_writes_and_multiple_statements():
    from explorer.agent import Agent
    with pytest.raises(ValueError):
        Agent._check_fragment("CNT = 'USA'; DROP TABLE x")
    with pytest.raises(ValueError):
        Agent._check_identifier("CNT; DROP")
    Agent._check_fragment("CNT IN ('USA','FIN')")   # fine
    Agent._check_identifier("ST004D01T")             # fine


def test_validate_plan_names_missing_fields():
    from explorer.agent import Agent
    with pytest.raises(ValueError, match="missing measure"):
        Agent._validate_plan({"template": "weighted_mean", "cycles": ["2025"]})
    with pytest.raises(ValueError, match="missing measure"):      # "None" placeholder
        Agent._validate_plan({"template": "weighted_mean", "measure": "None"})
    with pytest.raises(ValueError, match="group_col"):
        Agent._validate_plan({"template": "gap", "measure": "PV{pv}MATH",
                              "minuend": 2, "subtrahend": 1})
    with pytest.raises(ValueError, match="unknown template"):
        Agent._validate_plan({"template": "explore"})
    # a threshold share filed as a proportion with only a measure is complete
    Agent._validate_plan({"template": "weighted_proportion",
                          "measure": "CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END"})
    Agent._validate_plan({"template": "weighted_proportion", "variable": "REPEAT", "value": 1})


def test_catalog_search_prefers_phrase_over_scattered_tokens(monkeypatch):
    from explorer import catalog
    frame = pd.DataFrame({
        "variable": ["ST300Q01JA", "PROWBST", "BELONG", "ST016Q01NA", "CM033Q01S"],
        "table_name": ["stu_qqq_2025"] * 4 + ["stu_cog_2025"],
        "cycle": ["2025"] * 5,
        "instrument": ["stu_qqq"] * 4 + ["stu_cog"],
        "label": ["Discuss how well you are doing at school",
                  "Proportion of staff focused on well-being",
                  "Sense of belonging (WLE)",
                  "Overall, how satisfied are you with your life as a whole these days?",
                  "Chocolate and Health - Q01 (Scored Response)"],
        "var_type": ["double"] * 5, "value_labels": [None] * 5, "n_value_labels": [0] * 5,
    })
    monkeypatch.setattr(catalog, "_load", lambda: catalog._prepare(frame))
    top = catalog.search("well-being", limit=3).variable.tolist()
    assert top[0] == "PROWBST"                      # phrase match wins
    assert "ST300Q01JA" not in top[:1]              # "well" alone is a stopword
    top = catalog.search("life satisfaction", limit=3).variable.tolist()
    assert top[0] == "ST016Q01NA"                   # satisf~ stem + "life"
    assert catalog.search("belonging", limit=2).variable.tolist()[0] == "BELONG"


def test_regions_expand_per_cycle_and_aliases():
    from explorer import regions
    present = {"2018": {"ARG", "BRA", "CHL", "PAN", "JAM", "USA"},
               "2025": {"ARG", "BRA", "CHL", "GTM", "PRY", "SLV", "USA"}}
    out = regions.expand(["Latin America"], present)
    assert out["2018"]["codes"] == ["ARG", "BRA", "CHL", "JAM", "PAN"]
    assert out["2025"]["codes"] == ["ARG", "BRA", "CHL", "GTM", "PRY", "SLV"]
    assert "GTM" in out["2018"]["absent"] and "JAM" in out["2025"]["absent"]
    assert regions.canonical("LatAm") == "Latin America and the Caribbean"
    assert regions.canonical("the EU") is None or regions.canonical("EU") == "European Union"
    assert regions.canonical("Nordics") == "Nordic countries"
    assert "JPN" in regions.REGIONS["Asia"] and "SAU" in regions.REGIONS["Asia"]
    with pytest.raises(ValueError, match="unknown region"):
        regions.expand(["Atlantis"], present)


def test_summary_view_passes_full_ranking_and_focus_rows(monkeypatch):
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)          # no DB needed for these helpers
    from explorer import catalog
    labels = json.dumps({"KSV": "Kosovo", "SGP": "Singapore", "MAR": "Morocco"})
    fake = pd.DataFrame({"variable": ["CNT"], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
                         "label": ["Country"], "var_type": ["string"], "value_labels": [labels]})
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: fake if cycle == "2025" else fake.head(0))
    codes = [f"C{i:02d}" for i in range(86)] + ["ARE", "MAR", "KSV", "SGP"]
    est = list(range(90, 0, -1))
    table = pd.DataFrame({"CNT": codes, "estimate": est, "se": 1.0, "n_pv": 10, "cycle": "2025"})
    table.insert(0, "rank", range(1, 91))
    shown, truncation, focus = agent._summary_view(table, "where is kosovo ranked? countries are ranked")
    assert len(shown) == 90 and truncation == ""          # full compact ranking
    assert "n_pv" not in shown.columns
    assert "KSV: rank 89 of 90" in focus
    assert "ARE:" not in focus                             # lowercase "are" is not the UAE
    # a large unranked table keeps the window and the warning
    table2 = table.drop(columns="rank")
    shown2, truncation2, focus2 = agent._summary_view(table2, "compare Singapore and Morocco")
    assert len(shown2) == 30 and "WARNING" in truncation2
    assert "SGP:" in focus2 and "MAR:" in focus2 and "KSV:" not in focus2


def test_economies_named_in_question_are_found_by_name_or_capital_code(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent.present = {"2025": {"RWA", "KEN", "ARE", "QCI"}, "2022": {"ARE"}}
    labels = json.dumps({"RWA": "Rwanda", "KEN": "Kenya", "ARE": "United Arab Emirates",
                         "QCI": "B-S-J-Z (China)"})
    fake = pd.DataFrame({"variable": ["CNT"], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
                         "label": ["Country"], "var_type": ["string"], "value_labels": [labels]})
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: fake if cycle == "2025" else fake.head(0))
    assert agent._economies_in_data("How did rwanda do?") == ["RWA"]
    assert agent._economies_in_data("compare KEN and RWA") == ["KEN", "RWA"]
    assert agent._economies_in_data("how are things") == []          # "are" is not ARE
    assert agent._economies_in_data("United Arab Emirates reading") == ["ARE"]
    assert agent._economies_in_data("How did India do?") == []


def test_estimator_handles_empty_input_without_crashing():
    from explorer.estimator import ALL_WEIGHTS, combine, replicates_from_frame
    empty = pd.DataFrame(columns=["CNT", "m_1"] + ALL_WEIGHTS)
    reps = replicates_from_frame(empty, ["m_1"], by=("CNT",))
    assert reps.empty and list(reps.columns)[:4] == ["CNT", "pv", "rep", "value"]
    out = combine(reps, by=("CNT",))
    assert out.empty and list(out.columns)[:4] == ["CNT", "estimate", "se", "n_pv"]


# ---------- coverage of optional questionnaires ----------

def _coverage_agent(monkeypatch, coverage: dict):
    """An Agent with no DB, a fixed participant list and a fake coverage table
    keyed by (variable, table)."""
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent.present = {"2022": {"USA", "ESP", "FRA", "DEU"}, "2018": {"USA", "ESP", "FRA", "DEU"}}
    agent.economy_names = {"USA": "United States", "ESP": "Spain", "FRA": "France", "DEU": "Germany"}
    labels = {"EXPWB": "Experienced Well-being (Previous Day) (WLE)", "PV1MATH": "Math PV 1",
              "SWBP": "Subjective well-being: Positive affect (WLE)"}

    def fake_describe(var, cycle=None):
        return pd.DataFrame({"variable": [var], "table_name": [f"stu_qqq_{cycle or '2022'}"],
                             "cycle": [cycle or "2022"], "label": [labels.get(var, var)],
                             "var_type": ["double"], "value_labels": [None]})

    def fake_coverage(var, table):
        row = coverage.get((var, table))
        if row is None:
            return None
        with_data = set(row)
        every = {"USA", "ESP", "FRA", "DEU"} | with_data
        return {"n_economies": len(every), "n_with_data": len(with_data),
                "partial": with_data != every, "with_data": with_data if with_data != every else None,
                "missing": (every - with_data) if with_data != every else None}

    monkeypatch.setattr(catalog, "describe", fake_describe)
    monkeypatch.setattr(catalog, "coverage", fake_coverage)
    return agent


def test_coverage_guard_blocks_a_cycle_the_named_economy_never_collected(monkeypatch):
    cov = {("EXPWB", "stu_qqq_2022"): ["ESP", "FRA"], ("PV1MATH", "stu_qqq_2022"): ["USA", "ESP", "FRA", "DEU"]}
    agent = _coverage_agent(monkeypatch, cov)
    plan = {"template": "correlation", "x": "EXPWB", "y": "PV{pv}MATH", "where": "CNT IN ('USA')"}
    findings = agent._coverage_check(plan, ["2022"], "stu_qqq", {}, {})
    assert list(findings) == ["2022"]
    f = findings["2022"][0]
    assert f["variable"] == "EXPWB" and f["blocked"] and f["named_missing"] == ["USA"]
    msg = agent._coverage_message(findings)
    assert "United States (USA) did not administer" in msg and "2 of 4 economies" in msg
    assert "Spain (ESP)" in msg                       # economies with data are listed
    note = agent._coverage_note(f)
    assert "omitted" in note and "PISA 2022" in note
    # one of two named economies lacking it => not blocked, but noted
    plan2 = {**plan, "where": "CNT IN ('USA', 'ESP')"}
    f2 = agent._coverage_check(plan2, ["2022"], "stu_qqq", {}, {})["2022"][0]
    assert not f2["blocked"] and f2["named_missing"] == ["USA"]
    assert "NOT collected for United States (USA)" in agent._coverage_note(f2)
    # a complete variable never produces a finding
    plan3 = {"template": "weighted_mean", "measure": "PV{pv}MATH", "where": "CNT IN ('USA')"}
    assert agent._coverage_check(plan3, ["2022"], "stu_qqq", {}, {}) == {}


def test_coverage_cards_flag_named_economies_and_offer_alternatives(monkeypatch):
    cov = {("EXPWB", "stu_qqq_2022"): ["ESP", "FRA"],
           ("SWBP", "stu_qqq_2018"): ["USA", "ESP", "FRA", "DEU"]}
    agent = _coverage_agent(monkeypatch, cov)
    hits = pd.DataFrame({"variable": ["EXPWB", "SWBP"], "table_name": ["stu_qqq_2022", "stu_qqq_2018"],
                         "cycle": ["2022", "2018"],
                         "label": ["Experienced Well-being (Previous Day) (WLE)",
                                   "Subjective well-being: Positive affect (WLE)"],
                         "n_value_labels": [0, 0], "score": [50.0, 40.0]})
    cards = agent._cards(hits, named=["USA"])
    assert "stu_qqq_2022: NOT collected for USA" in cards
    assert "SWBP" in cards and "NOT collected for USA (collected in 4" not in cards
    alts = agent._coverage_alternatives(hits, {"USA"}, {"EXPWB"})
    assert alts == ["Subjective well-being: Positive affect (WLE) (SWBP; PISA 2018)"]


def test_user_facing_text_drops_planner_notation():
    from explorer.agent import Agent
    text = ("Please choose: 1) COOPAGR and PV{pv}MATH, or 2) IC174Q10JA and "
            "PV1MATH; `PV{pv}SCIE` too")
    out = Agent._plain(text)
    assert "{pv}" not in out and "PV1MATH" not in out and "`" not in out
    assert out.count("mathematics score") == 2 and "science score" in out
    assert Agent._plain(None) == ""


def test_gender_override_direction_is_aligned_to_the_main_plan():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    # main plan: female minus male (1 - 2); override says male minus female (1 - 0)
    plan = {"template": "gap", "group_col": "ST004D01T", "minuend": 1, "subtrahend": 2,
            "cycle_overrides": {"2025": {"group_col": "MALE", "minuend": 1, "subtrahend": 0}}}
    overrides = {k: v for k, v in plan["cycle_overrides"].items()}
    agent._align_gender_direction(plan, overrides)
    assert overrides["2025"] == {"group_col": "MALE", "minuend": 0, "subtrahend": 1}
    assert plan["cycle_overrides"]["2025"]["minuend"] == 0          # same dict object
    assert plan["_direction_fixed"] and "re-aligned" in plan["_direction_fixed"][0]
    # a consistent override is left alone
    plan2 = {"template": "gap", "group_col": "ST004D01T", "minuend": 2, "subtrahend": 1,
             "cycle_overrides": {"2025": {"group_col": "MALE", "minuend": 1, "subtrahend": 0}}}
    ov2 = dict(plan2["cycle_overrides"])
    agent._align_gender_direction(plan2, ov2)
    assert ov2["2025"] == {"group_col": "MALE", "minuend": 1, "subtrahend": 0}
    assert "_direction_fixed" not in plan2


def test_blank_estimates_distinguish_non_participation_from_missing_variables():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent.present = {"2018": {"MEX", "BRA"}, "2022": {"MEX", "BRA", "SLV"}}
    agent.economy_names = {"SLV": "El Salvador", "MEX": "Mexico", "BRA": "Brazil"}
    table = pd.DataFrame({"CNT": ["MEX", "BRA", "SLV"],
                          "estimate_2018": [408.0, 383.0, np.nan], "se_2018": [2.0, 2.0, np.nan],
                          "estimate_2022": [395.0, np.nan, 343.0], "se_2022": [2.0, np.nan, 2.0]})
    notes = agent._missing_estimate_notes(table)
    assert notes[0].startswith("PISA 2018: El Salvador (SLV) did not take part")
    assert "1 row(s) have no estimate" in notes[1]          # Brazil's blank 2022 cell is a real gap
    assert agent._missing_estimate_notes(table.dropna()) == []


# ---------- link errors, derived-group gaps, deterministic method answers ----------

def test_trend_adds_the_link_error_variance_when_given():
    a, b = _result(["FIN"], [520.0], [2.0]), _result(["FIN"], [510.0], [2.0])
    out = trend({"2022": a, "2025": b}, by=["CNT"])
    assert abs(out.se_change.iloc[0] - np.sqrt(8.0)) < 1e-9
    out2 = trend({"2022": a, "2025": b}, by=["CNT"], link_error=3.0)
    assert abs(out2.se_change.iloc[0] - np.sqrt(8.0 + 9.0)) < 1e-9
    assert out2.change.iloc[0] == -10.0                     # estimates untouched
    from explorer import link_errors
    assert link_errors.domain_of("PV{pv}SCIE") == "SCIE" and link_errors.domain_of("ESCS") is None
    assert link_errors.loaded() and len(link_errors.LINK_ERRORS) == 9
    assert link_errors.link_error("2022", "2025", "PV{pv}SCIE") == 3.116
    assert link_errors.link_error("2018", "2022", "PV1MATH") == 2.24
    assert link_errors.link_error("2022", "2025", "ESCS") is None
    # proficiency-level shares have their own (unloaded) link errors
    assert link_errors.link_error("2022", "2025", "CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END") is None
    assert link_errors.domain_of("CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END") == "MATH"


def test_gap_accepts_a_derived_two_group_expression():
    import duckdb
    from explorer.analysis import gap
    rng = np.random.default_rng(0)
    n = 400
    immig = rng.choice([1, 2, 3], size=n, p=[0.7, 0.15, 0.15])
    base = np.where(immig == 1, 500.0, 460.0) + rng.normal(0, 30, n)
    df = pd.DataFrame({"CNT": "BRA", "IMMIG": immig})
    for i in range(1, 11):
        df[f"PV{i}SCIE"] = base + rng.normal(0, 5, n)
    for w in ALL_WEIGHTS:
        df[w] = rng.uniform(0.5, 1.5, n)
    con = duckdb.connect()
    con.register("t", df)
    out = gap(con, "t", "PV{pv}SCIE", "IMMIG", 1, 0, by=("CNT",),
              group_expr="CASE WHEN IMMIG = 1 THEN 1 WHEN IMMIG IN (2, 3) THEN 0 END",
              group_label="non-immigrant minus immigrant")
    assert len(out) == 1 and out.contrast.iloc[0] == "non-immigrant minus immigrant: 1 - 0"
    assert 25 < out.estimate.iloc[0] < 55 and out.se.iloc[0] > 0


def test_participation_rule_ignores_analysis_questions():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    assert agent._is_participation_question("Did India participate in 2025 PISA?")
    assert agent._is_participation_question("do you have data on Rwanda?")
    assert not agent._is_participation_question("Compare between Egypt and Israel in PISA 2025")
    assert not agent._is_participation_question("Compare between mean science score in Israel and Egypt in PISA 2025")
    assert not agent._is_participation_question("How did rwanda do?")


def test_method_and_coverage_rate_questions_are_intercepted():
    from explorer.agent import Agent
    L, C = Agent.LINK_WORDS, Agent.COVERAGE_RATE_WORDS
    assert L.search("Why is OECD link error excluded?")
    assert L.search("Your analysis puts Brazil's science score change as significant whereas the OECD analysis marks it as non-significant. Why is that?")
    assert not L.search("How did countries in Latin America perform in PISA 2025 vs OECD?")
    assert C.search("how did coverage of 15-year-olds change in Latin American countries between 2022 and 2025?")
    assert C.search("do you have data on coverage rates?")
    assert not C.search("do you have the 2025 data?")
    # audit: neither interceptor may swallow ordinary data questions
    assert not L.search("what is the difference between Brazil and the OECD results?")
    assert not L.search("compare Brazil with the OECD average in science")
    assert not C.search("what percentage of 15-year-olds are below Level 2 in math?")
    assert not C.search("what share of students in Brazil are in the top quartile?")
    agent = Agent.__new__(Agent)
    assert "Annex A5" in agent._link_error_answer() and "Annex A2" in agent._coverage_rate_answer()


def test_blank_plausible_values_are_described_as_not_released():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent.present = {"2018": {"VNM", "FIN"}, "2022": {"VNM", "FIN"}}
    agent.economy_names = {"VNM": "Viet Nam", "FIN": "Finland"}
    table = pd.DataFrame({"CNT": ["VNM", "FIN"], "estimate_2018": [np.nan, 520.0], "se_2018": [np.nan, 2.0],
                          "estimate_2022": [470.0, 510.0], "se_2022": [3.0, 2.0]})
    notes = agent._missing_estimate_notes(table, {"template": "weighted_mean", "measure": "PV{pv}SCIE"})
    gap_notes = agent._missing_estimate_notes(table, {"template": "gap", "measure": "PV{pv}SCIE"})
    assert "did not release" not in gap_notes[0]                 # a blank gap cell is an empty group
    assert len(notes) == 1 and "did not release" in notes[0] and "Viet Nam (VNM)" in notes[0]
    assert "not administered" not in notes[0]
    f = {"variable": "PV1SCIE", "label": "Science PV", "cycle": "2018", "n_with_data": 79, "n_economies": 80,
         "with_data": [], "missing": ["VNM"], "named": ["VNM"], "named_missing": ["VNM"], "blocked": True}
    assert "did not release its science results" in agent._coverage_note(f)
    assert "did not release its science results" in agent._coverage_message({"2018": [f]})


def test_explore_drops_weak_matches(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    weak = pd.DataFrame({"variable": ["FLSCHOOL"], "table_name": ["stu_qqq_2022"], "cycle": ["2022"],
                         "label": ["Financial education in school lessons (WLE)"], "n_value_labels": [0], "score": [2.4]})
    monkeypatch.setattr(agent, "_retrieve", lambda terms, per_term=12: weak)
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: weak.head(0).assign(var_type=[], value_labels=[]))
    res = agent._explore("do you have data on coverage rates?", ["coverage rates"])
    assert res.route == "explore" and "No catalog variables matched" in res.answer


# ---------- student + school joined view ----------

def test_joined_view_maps_to_its_physical_tables_and_columns_are_checked(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    assert Agent.underlying_tables("stu_sch_2022") == ["stu_qqq_2022", "sch_qqq_2022"]
    assert Agent.underlying_tables("stu_qqq_2025") == ["stu_qqq_2025"]
    agent = Agent.__new__(Agent)
    monkeypatch.setattr(agent, "_table_columns",
                        lambda t: {"CNT", "PV1MATH", "W_FSTUWT", "ST004D01T"} if t == "stu_qqq_2022"
                        else {"CNT", "PV1MATH", "W_FSTUWT", "SC013Q01TA"})
    rows = {"SC013Q01TA": ["sch_qqq_2022"], "PV1MATH": ["stu_qqq_2022"], "ST004D01T": ["stu_qqq_2022"]}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var] * len(rows.get(var, [])), "table_name": rows.get(var, []),
         "cycle": ["2022"] * len(rows.get(var, [])), "label": [var] * len(rows.get(var, [])),
         "var_type": ["double"] * len(rows.get(var, [])), "value_labels": [None] * len(rows.get(var, []))}))
    ok = {"template": "gap", "measure": "PV{pv}MATH", "group_col": "ST004D01T", "minuend": 2, "subtrahend": 1,
          "where": "CNT = 'MEX'"}
    agent._check_columns(ok, "stu_qqq_2022")                       # nothing raised
    bad = {**ok, "group_col": "SC013Q01TA"}
    with pytest.raises(ValueError, match="SCHOOL questionnaire variable.*stu_sch"):
        agent._check_columns(bad, "stu_qqq_2022")
    agent._check_columns(bad, "stu_sch_2022")                      # the joined view has it


def test_joined_view_exists_in_the_local_database():
    from explorer.db import DB_PATH
    if not DB_PATH.exists():
        pytest.skip("no local DuckDB (CI)")
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cols = {r[0] for r in con.execute("SELECT column_name FROM information_schema.columns "
                                      "WHERE table_name = 'stu_sch_2022'").fetchall()}
    assert {"CNT", "CNTSCHID", "W_FSTUWT", "W_FSTURWT80", "PV10SCIE", "SC013Q01TA"} <= cols
    n_stu = con.execute("SELECT count(*) FROM stu_qqq_2022").fetchone()[0]
    n_view = con.execute("SELECT count(*) FROM stu_sch_2022").fetchone()[0]
    assert n_stu == n_view                                           # LEFT JOIN: no row lost or duplicated



# ---------- several measures, dropped-measure backstop, overview ----------

def test_plan_with_measures_list_validates_and_labels(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    Agent._validate_plan({"template": "weighted_mean", "measures": ["PV{pv}MATH", "ESCS"]})
    with pytest.raises(ValueError):
        Agent._validate_plan({"template": "weighted_mean"})
    agent = Agent.__new__(Agent)
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
         "label": ["Index of economic, social and cultural status"], "var_type": ["double"], "value_labels": [None]}))
    plan = {"template": "weighted_mean", "measures": ["PV{pv}MATH", "PV{pv}READ", "ESCS", "PV{pv}MATH"]}
    lst = agent._measure_list(plan)
    assert [l for l, _ in lst] == ["Mathematics score", "Reading score",
                                   "Index of economic, social and cultural status (ESCS)"]
    single = {"template": "weighted_mean", "measures": ["PV{pv}SCIE"]}
    assert agent._measure_list(single) == [] and single["measure"] == "PV{pv}SCIE"
    assert agent._measure_list({"template": "gap", "measures": ["a", "b"]}) == []


def test_requested_measures_left_out_of_a_plan_are_named():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    q = "give me mathematics, reading and socioeconomic status for Argentina"
    plan = {"template": "weighted_mean", "measure": "PV{pv}MATH", "where": "CNT = 'ARG'"}
    assert agent._dropped_measures(q, plan) == ["reading", "socio-economic status (ESCS)"]
    full = {"template": "weighted_mean", "measures": ["PV{pv}MATH", "PV{pv}READ", "ESCS"]}
    assert agent._dropped_measures(q, full) == []
    assert agent._dropped_measures("math gender gap in Germany", plan) == []     # one measure named
    assert agent._dropped_measures("reading by ESCS quartile in Chile",
                                   {"template": "quartile_means", "measure": "PV{pv}READ",
                                    "quart_variable": "ESCS"}) == []


def test_overview_question_is_intercepted_but_topic_searches_are_not():
    from explorer.agent import Agent
    O = Agent.OVERVIEW_WORDS
    assert O.search("what variables is pisa measuring in latam?")
    assert O.search("What does PISA measure?")
    assert not O.search("find me data related to well-being from 2025 data")
    assert not O.search("what data do you have about bullying?")
    agent = Agent.__new__(Agent)
    agent.coverage = {"2018": {}, "2022": {}, "2025": {}}
    ans = agent._overview_answer()
    assert "mathematics, reading and science" in ans and "Learning in the Digital World" in ans



def test_unknown_columns_use_the_whole_catalog_and_school_type_is_standardized(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    monkeypatch.setattr(agent, "_table_columns", lambda t: {"CNT", "PV1MATH", "ESCS", "W_FSTUWT"})
    known = {"GLOBMIND": ["stu_qqq_2018"], "PV1MATH": ["stu_qqq_2025"], "ESCS": ["stu_qqq_2025"]}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var] * len(known.get(var, [])), "table_name": known.get(var, []),
         "cycle": ["2018"] * len(known.get(var, [])), "label": [var] * len(known.get(var, [])),
         "var_type": ["double"] * len(known.get(var, [])), "value_labels": [None] * len(known.get(var, []))}))
    # a 2018-only index is recognised as a variable even when checking the 2025 table
    assert agent._unknown_columns("GLOBMIND", "stu_qqq_2025") == ["GLOBMIND"]
    assert agent._unknown_columns("CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END", "stu_qqq_2025") == []
    with pytest.raises(ValueError, match="exists only in 2018"):
        agent._check_columns({"template": "weighted_mean", "measure": "GLOBMIND"}, "stu_qqq_2025")
    plan = {"template": "gap", "group_col": "PRIVATESCH", "minuend": 2, "subtrahend": 1, "instrument": "stu_sch"}
    agent._prefer_reported_school_type(plan)
    assert plan["group_col"] == "SC013Q01TA" and plan["_school_type_switched"]
    from explorer import standards
    names = lambda q: [s.construct for s in standards.matching(q)]   # noqa: E731
    assert "public vs private school" in names("difference between mexico's public and private schools")
    assert "public vs private school" not in names("how did rwanda do?")



# ---------- benchmark averages, named-economy backstop, "why no results" ----------

def test_group_average_for_any_group_and_avg_rows_are_never_ranked(monkeypatch):
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    agent.present = {"2025": {"FRA", "DEU", "ESP", "USA", "BRA"}}
    agent.economy_names = {}
    monkeypatch.setattr(agent, "_oecd_codes", lambda c: {"FRA", "DEU", "ESP", "USA"})
    assert agent._benchmarks({"include_oecd_average": True, "include_average_of": ["European Union", "OECD"]}) \
        == ["OECD", "European Union"]
    assert agent._benchmarks({"include_average_of": ["FRA", "DEU", "BRA"]}) == [("BRA", "DEU", "FRA")]
    label, members = agent._average_members("EU", "2025")
    assert label == "EU avg" and members == {"FRA", "DEU", "ESP"}
    assert agent._average_members("OECD", "2025") == ("OECD avg", {"FRA", "DEU", "ESP", "USA"})
    assert agent._average_members("Narnia", "2025") is None
    res = pd.DataFrame({"CNT": ["FRA", "DEU", "ESP", "BRA"], "estimate": [450.0, 470.0, 460.0, 400.0],
                        "se": [3.0, 4.0, 3.0, 2.0], "n_pv": 10})
    avg = agent._group_average_rows(res, members, "EU avg")
    assert avg.CNT.iloc[0] == "EU avg" and abs(avg.estimate.iloc[0] - 460.0) < 1e-9
    assert abs(avg.se.iloc[0] - np.sqrt(9 + 16 + 9) / 3) < 1e-9
    # blank-cell notes ignore benchmark rows
    agent.present = {"2018": {"FRA"}, "2025": {"FRA"}}
    table = pd.DataFrame({"CNT": ["FRA", "EU avg"], "estimate_2018": [1.0, np.nan], "se_2018": [1.0, np.nan],
                          "estimate_2025": [2.0, 3.0], "se_2025": [1.0, 1.0]})
    notes = agent._missing_estimate_notes(table, {"template": "weighted_mean", "measure": "ESCS"})
    assert all("did not take part" not in n for n in notes)


def test_named_economies_left_out_of_a_plan_are_stated(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent.present = {"2025": {"FRA", "DEU", "ESP", "USA"}}
    labels = json.dumps({"FRA": "France", "DEU": "Germany", "ESP": "Spain", "USA": "United States"})
    fake = pd.DataFrame({"variable": ["CNT"], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
                         "label": ["Country"], "var_type": ["string"], "value_labels": [labels]})
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: fake if cycle == "2025" else fake.head(0))
    monkeypatch.setattr(agent, "_oecd_codes", lambda c: {"FRA", "DEU", "ESP", "USA"})
    q = "compare France with Germany and Spain in reading"
    assert agent._dropped_economies(q, {"cycles": ["2025"], "where": "CNT = 'FRA'"}) == ["DEU", "ESP"]
    assert agent._dropped_economies(q, {"cycles": ["2025"], "where": "CNT IN ('FRA','DEU','ESP')"}) == []
    assert agent._dropped_economies(q, {"cycles": ["2025"]}) == []                     # all economies
    assert agent._dropped_economies("France vs the EU average", {"cycles": ["2025"], "where": "CNT = 'FRA'",
                                                                 "include_average_of": ["European Union"]}) == []
    assert agent._dropped_economies("compare Germany and Spain with the EU average",
                                    {"cycles": ["2025"], "where": "CNT = 'DEU'",
                                     "include_average_of": ["European Union"]}) == []   # Spain is in the EU average


def test_why_no_results_is_answered_from_coverage_and_the_oecd_reason(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    W = Agent.WHY_MISSING_WORDS
    assert W.search("Why can’t I see mathematics and reading results for Uzbekistan?")
    assert W.search("why are there no math scores for Vietnam in 2018")
    assert not W.search("why is Uzbekistan's science score so low?")
    assert not W.search("what is the reading score of Uzbekistan?")
    agent = Agent.__new__(Agent)
    agent.present = {"2022": {"FIN"}, "2025": {"UZB", "FIN"}}
    agent.economy_names = {"UZB": "Uzbekistan", "FIN": "Finland"}
    def cov(var, table):
        if table == "stu_qqq_2025" and var in ("PV1MATH", "PV1READ"):
            return {"n_economies": 90, "n_with_data": 89, "partial": True, "with_data": set(), "missing": {"UZB"}}
        return {"n_economies": 90, "n_with_data": 90, "partial": False, "with_data": None, "missing": None}
    monkeypatch.setattr(catalog, "coverage", cov)
    ans = agent._missing_results_answer(["UZB"])
    assert "did not take part in PISA 2022" in ans
    assert "results for science only" in ans and "mathematics and reading plausible values were not released" in ans
    assert "Data Adjudication" in ans and "not collected" not in ans
    assert "all three domains" in agent._missing_results_answer(["FIN"])



def test_adversarial_guards_qci_ranking_and_null_groups():
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    assert agent._qci_note("How does Shanghai compare to Singapore?", {"where": "CNT IN ('QCI','SGP')"})
    assert agent._qci_note("Singapore vs Japan", {"where": "CNT IN ('SGP','JPN')"}) is None
    assert agent._qci_note("B-S-J-Z vs Singapore", {"where": "CNT IN ('QCI','SGP')"}) is None
    plan = {"top_n": 1, "sort_by": "estimate"}
    agent._keep_full_ranking("Rank Latin American countries by the gap and name the largest", plan)
    assert plan["top_n"] is None and plan["_ranking_kept"]
    plan2 = {"top_n": 1, "sort_by": "estimate"}
    agent._keep_full_ranking("which country has the largest gap?", plan2)
    assert plan2["top_n"] == 1
    res = pd.DataFrame({"CNT": ["CHL"] * 3, "SC013Q01TA": [1.0, 2.0, np.nan],
                        "estimate": [-17.0, -21.4, -30.2], "se": [5.8, 3.1, 12.0], "n_pv": 10})
    kept, dropped = Agent._drop_null_groups(res, ["CNT", "SC013Q01TA"])
    assert len(kept) == 2 and dropped == {"SC013Q01TA": 1}
    same, none = Agent._drop_null_groups(res.dropna(), ["CNT"])
    assert len(same) == 2 and none == {}



def test_pure_count_questions_are_intercepted_but_mixed_ones_are_planned():
    from explorer.agent import Agent
    C, O = Agent.COUNT_WORDS, Agent.OTHER_STAT_WORDS
    assert C.search("How many students were tested in Uzbekistan in 2025?") and not O.search("How many students were tested in Uzbekistan in 2025?")
    mixed = "How many students were tested in Uzbekistan in 2025 and what percentage were girls?"
    assert C.search(mixed) and O.search(mixed)          # goes to the planner; sample size rides along
    assert not C.search("what is the mean science score in Brazil?")



def test_proportion_rows_state_their_category(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    labels = json.dumps({"0": "Female/Other", "1": "Male"})
    fake = pd.DataFrame({"variable": ["MALE"], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
                         "label": ["Derived gender"], "var_type": ["double"], "value_labels": [labels]})
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: fake)
    assert Agent._category_label("MALE", 0, "stu_qqq_2025") == "MALE = 0 (Female/Other)"
    assert Agent._category_label("MALE", 1.0, "stu_sch_2025") == "MALE = 1 (Male)"
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: fake.head(0))
    assert Agent._category_label("IMMIG", 2, "stu_qqq_2022") == "IMMIG = 2"



def test_false_not_collected_claims_are_corrected_from_coverage(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent.present = {"2018": {"USA", "CAN"}, "2025": {"USA", "CAN"}}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"], "label": [var],
         "var_type": ["double"], "value_labels": [None]}) if var in ("ST004D01T", "MALE") else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    def cov(var, table):
        if var == "ST004D01T" and table == "stu_qqq_2025":
            return {"n_economies": 90, "n_with_data": 76, "partial": True, "with_data": set(), "missing": {"CAN"}}
        return {"n_economies": 90, "n_with_data": 90, "partial": False, "with_data": None, "missing": None}
    monkeypatch.setattr(catalog, "coverage", cov)
    plan = {"template": "gap", "where": "CNT = 'USA'", "cycles": ["2018", "2025"],
            "substitution_note": "For 2025 the MALE flag is used as ST004D01T was not collected for the US in 2025."}
    agent._verify_substitution_claims(plan)
    assert "ST004D01T" in plan["_claim_corrected"] and "standard convention" in plan["substitution_note"]
    # Canada really lacks ST004D01T in the 2025 public file — but it was
    # withheld, not "not collected": the wording is replaced either way
    plan2 = {"template": "gap", "where": "CNT = 'CAN'", "cycles": ["2025"],
             "substitution_note": "MALE is used as ST004D01T was not collected for Canada in 2025."}
    agent._verify_substitution_claims(plan2)
    assert plan2["_claim_corrected"] == ["ST004D01T"] and "not released" in plan2["substitution_note"]
    assert "not collected" not in plan2["substitution_note"]



def test_index_trends_are_flagged_non_comparable_but_scores_are_not(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    labels = {"CURIO": "Students' curiousity (WLE)", "ESCS": "Index of economic, social and cultural status",
              "ST301Q01JA": "Agree/disagree: I am curious about many different things."}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"], "label": [labels.get(var, var)],
         "var_type": ["double"], "value_labels": [None]}) if var in labels else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    assert agent._trend_comparability("PV{pv}MATH", {}, ["2018", "2025"], {}) is None
    assert agent._trend_comparability("CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END", {}, ["2018", "2025"], {}) is None
    assert "standardized within each PISA cycle" in agent._trend_comparability("CURIO", {}, ["2022", "2025"], {})
    assert "standardized" in agent._trend_comparability("ESCS", {}, ["2018", "2025"], {})
    assert "different variables" in agent._trend_comparability(
        "ST301Q01JA", {}, ["2022", "2025"], {"2025": {"measure": "ST301Q06JA"}})
    assert agent._trend_comparability("ST301Q01JA", {}, ["2022", "2025"], {}) is None      # same item: comparable
    table = pd.DataFrame({"CNT": ["ARG", "ARG"], "measure": ["Mathematics score", "ESCS (ESCS)"],
                          "estimate_2018": [379.0, -0.95], "estimate_2025": [366.0, -0.63],
                          "change": [-13.0, 0.32], "se_change": [4.6, 0.04]})
    plan = {}
    agent._blank_non_comparable_changes(table, plan, [("Mathematics score", "PV{pv}MATH"), ("ESCS (ESCS)", "ESCS")],
                                        ["2018", "2025"], {})
    assert table.change.iloc[0] == -13.0 and np.isnan(table.change.iloc[1]) and np.isnan(table.se_change.iloc[1])
    assert plan["_non_comparable"] and "ESCS" in plan["_non_comparable"][0]


def test_ai_use_questions_are_data_questions_and_english_ones_skip_translation(monkeypatch):
    from explorer.agent import Agent
    agent = Agent.__new__(Agent)
    A, D = Agent.AI_WORDS, Agent.AI_DATA_WORDS
    q = "What is the AI usage rate among students in Japan?"
    assert A.search(q) and D.search(q)
    assert A.search("Is there data on the rate of AI use in education?")
    assert not A.search("what is the average science score in Finland?")
    assert agent._localize("Hello", "English") == "Hello" and agent._localize("Hello", None) == "Hello"



def test_false_claims_nested_in_cycle_overrides_are_corrected_too(monkeypatch):
    from explorer.agent import Agent
    from explorer import catalog
    agent = Agent.__new__(Agent)
    agent.present = {"2018": {"USA"}, "2025": {"USA"}}
    monkeypatch.setattr(catalog, "describe", lambda var, cycle=None: pd.DataFrame(
        {"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"], "label": [var],
         "var_type": ["double"], "value_labels": [None]}) if var in ("ST004D01T", "MALE") else pd.DataFrame(
        columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"]))
    monkeypatch.setattr(catalog, "coverage", lambda var, table: {"n_economies": 90, "n_with_data": 90, "partial": False,
                                                                  "with_data": None, "missing": None})
    plan = {"template": "gap", "where": "CNT = 'USA'", "cycles": ["2018", "2025"], "group_col": "ST004D01T",
            "cycle_overrides": {"2025": {"group_col": "MALE", "minuend": 1, "subtrahend": 0,
                                         "substitution_note": "MALE is used as ST004D01T was not collected for the US."}}}
    agent._verify_substitution_claims(plan)
    assert "substitution_note" not in plan["cycle_overrides"]["2025"]
    assert "standard convention" in plan["substitution_note"] and "ST004D01T" in plan["_claim_corrected"]
