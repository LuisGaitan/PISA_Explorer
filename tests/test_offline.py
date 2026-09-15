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
    assert reps.empty and list(reps.columns) == ["CNT", "pv", "rep", "value"]
    out = combine(reps, by=("CNT",))
    assert out.empty and list(out.columns) == ["CNT", "estimate", "se", "n_pv"]


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
