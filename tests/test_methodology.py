"""Offline tests for the conventions fixed after the 2026-09-20 methodology
and adversarial red team: where the link error applies, constant benchmark
membership, minimum reporting sizes, code succession, locale-aware prose
checks, strata phrase matching, level labels."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import link_errors, regions, summary as summ   # noqa: E402
from explorer.analysis import trend                            # noqa: E402


def _agent():
    from explorer.agent import Agent
    a = Agent.__new__(Agent)
    a._fired = []
    a.present = {"2018": {"USA", "FRA", "QAZ", "LUX"}, "2022": {"USA", "FRA", "AZE", "CRI"},
                 "2025": {"USA", "FRA", "AZE", "CRI", "LUX"}}
    a.economy_names = {"USA": "United States", "FRA": "France", "QAZ": "Baku (Azerbaijan)",
                       "AZE": "Azerbaijan", "LUX": "Luxembourg", "CRI": "Costa Rica"}
    return a


def test_link_error_applies_to_levels_only():
    a = _agent()
    le, mode = a._link_error_for("gap", "PV{pv}MATH", ["2018", "2025"], ("CNT",), {})
    assert le is None and mode == "cancels"
    le, mode = a._link_error_for("regression", "PV{pv}READ", ["2022", "2025"], ("CNT",), {})
    assert le is None and mode == "cancels"
    le, mode = a._link_error_for("weighted_mean", "PV{pv}MATH", ["2018", "2025"], ("CNT",), {})
    assert le == 2.551 and mode == "mean"
    le, mode = a._link_error_for("percentiles", "PV{pv}SCIE", ["2022", "2025"], ("CNT",), {})
    assert le == 3.116 and mode == "percentile-approx"
    le, mode = a._link_error_for("weighted_mean", "ESCS", ["2018", "2025"], ("CNT",), {})
    assert le is None and mode == "none"


def test_trend_accepts_per_row_link_errors():
    r22 = pd.DataFrame({"CNT": ["BRA", "CHL"], "estimate": [55.4, 44.0], "se": [0.9, 1.0], "n_pv": 10})
    r25 = pd.DataFrame({"CNT": ["BRA", "CHL"], "estimate": [52.5, 43.0], "se": [0.9, 1.0], "n_pv": 10})
    per_row = pd.DataFrame({"CNT": ["BRA", "CHL"], "link_error": [1.25, 0.8]})
    out = trend({"2022": r22, "2025": r25}, by=("CNT",), link_error=per_row).set_index("CNT")
    assert out.loc["BRA", "se_change"] == pytest.approx(np.sqrt(0.9 ** 2 + 0.9 ** 2 + 1.25 ** 2))
    assert out.loc["CHL", "se_change"] == pytest.approx(np.sqrt(1.0 ** 2 + 1.0 ** 2 + 0.8 ** 2))


def test_benchmark_rows_use_constant_membership_and_nan_aware_n():
    a = _agent()
    basis18 = pd.DataFrame({"CNT": ["USA", "FRA", "LUX"], "estimate": [480.0, 490.0, 470.0],
                            "se": [3.0, 2.0, 1.0], "n_pv": 10})
    basis22 = pd.DataFrame({"CNT": ["USA", "FRA", "CRI"], "estimate": [470.0, 480.0, np.nan],
                            "se": [3.0, 2.0, np.nan], "n_pv": 10})
    per_cycle = {"2018": basis18.copy(), "2022": basis22.copy()}
    bench = {"2018": {None: {"OECD avg": (basis18, {"USA", "FRA", "LUX"})}},
             "2022": {None: {"OECD avg": (basis22, {"USA", "FRA", "CRI"})}}}
    plan = {}
    a._append_benchmark_rows(per_cycle, bench, ["2018", "2022"], False, plan)
    avg18 = per_cycle["2018"][per_cycle["2018"]["CNT"] == "OECD avg"].iloc[0]
    avg22 = per_cycle["2022"][per_cycle["2022"]["CNT"] == "OECD avg"].iloc[0]
    assert avg18["estimate"] == pytest.approx(485.0) and avg22["estimate"] == pytest.approx(475.0)  # USA+FRA only
    assert avg18["se"] == pytest.approx(np.sqrt(9 + 4) / 2)
    assert plan["_benchmark_members"]["2018"]["OECD avg"] == ["FRA", "USA"]
    assert plan["_benchmark_dropped"]["OECD avg"] == ["LUX"]           # CRI had no estimate, LUX no 2022 row
    # NaN-aware N in a single cycle
    rows = a._group_average_rows(basis22, {"USA", "FRA", "CRI"}, "OECD avg")
    assert rows["se"].iloc[0] == pytest.approx(np.sqrt(9 + 4) / 2)


def test_spain_2018_reading_and_world_average_memberships():
    a = _agent()
    a.present["2018"] = {"USA", "FRA", "ESP"}
    a._oecd_codes = lambda cycle: {"USA", "FRA", "ESP"}
    plan = {}
    label, members = a._average_members("OECD", "2018", "PV{pv}READ", plan)
    assert members == {"USA", "FRA"} and plan["_avg_exclusions"]["2018"]["OECD avg"] == ["ESP"]
    label, members = a._average_members("OECD", "2018", "PV{pv}MATH", {})
    assert "ESP" in members
    label, members = a._average_members("world average", "2025", "PV{pv}MATH", {})
    assert label == "All-participant avg" and members == a.present["2025"]


def test_code_succession_note_instead_of_did_not_take_part():
    a = _agent()
    table = pd.DataFrame({"CNT": ["QAZ", "AZE"], "estimate_2018": [397.6, np.nan], "se_2018": [2.4, np.nan],
                          "estimate_2025": [np.nan, 408.6], "se_2025": [np.nan, 2.0],
                          "change": [np.nan, np.nan], "se_change": [np.nan, np.nan]})
    notes = a._missing_estimate_notes(table, {"template": "weighted_mean", "measure": "PV{pv}SCIE", "by": ["CNT"]})
    text = " ".join(notes)
    assert "did not take part" not in text
    assert "Azerbaijan (AZE) appears in that cycle as Baku (Azerbaijan) (QAZ)" in text
    assert regions.CODE_SUCCESSION["UKR"] == ("QUR", "QUA")


def test_suppressed_rows_are_not_called_unreleased():
    a = _agent()
    table = pd.DataFrame({"CNT": ["TUR", "KAZ"], "STRATUM": ["TUR01", "KAZ21"], "estimate": [np.nan, 581.0],
                          "se": [np.nan, 1.3], "n_pv": 10, "cycle": "2025"})
    plan = {"template": "weighted_mean", "measure": "PV{pv}MATH", "by": ["CNT", "STRATUM"],
            "_suppressed_rows": {"2025": [{"CNT": "TUR", "STRATUM": "TUR01"}]}}
    notes = a._missing_estimate_notes(table, plan)
    assert not any("did not release" in n for n in notes)


def test_prose_check_reads_locale_thousands():
    t = pd.DataFrame({"CNT": ["FRA"], "estimate": [-20.2], "se": [4.26], "cycle": "2022"})
    prov = {"notes": [], "method": "", "sample": [{"table": "stu_qqq_2022", "students": 6770}], "facts": []}
    issues = summ.check_prose("L'écart est de -20.2 points (SE 4.26) sur 6 770 élèves.", t, prov,
                              {"template": "gap"}, "", lambda s: [], {"FRA"}, language="French")
    assert issues == []
    issues = summ.check_prose("La brecha es -20.2 (SE 4.26) con 6.770 estudiantes.", t, prov,
                              {"template": "gap"}, "", lambda s: [], {"FRA"}, language="Spanish")
    assert issues == []


def test_strata_need_a_phrase_or_an_alias():
    a = _agent()
    a._strata_cache = [("2025", "KAZ21", "Intellectual schools", "intellectual schools"),
                       ("2025", "QSC01", "Publicly Funded Non-FE / lowest 20%", "publicly funded non-fe / lowest 20%"),
                       ("2025", "QUK01", "England/Academy/London", "england/academy/london"),
                       ("2025", "KAZ03", "General/Shymkent city", "general/shymkent city")]
    assert a._strata_hits("position of Nazarbayev Intellectual schools") == [("2025", "KAZ21", "Intellectual schools")]
    assert a._strata_hits("make a short summary of the tool and its author") == []
    assert a._strata_hits("Shymkent results") == []                       # one word, no phrase
    assert [h[1] for h in a._strata_hits("Scotland maths 2025")] == ["QSC01"]
    assert [h[1] for h in a._strata_hits("how did England do")] == ["QUK01"]


def test_level_labels_and_null_safe_unwrap():
    from explorer.agent import Agent
    assert link_errors.level_name("MATH", "669.30") == "Level 6"
    assert link_errors.level_name("SCIE", "409.54") == "Level 2"
    assert link_errors.level_name("READ", "1.5") is None
    assert Agent._measure_label("CASE WHEN PV{pv}MATH >= 669.3 THEN 100.0 ELSE 0.0 END") == "% at or above Level 6 in mathematics"
    wrapped = "CASE WHEN (PV1MATH) IS NULL THEN NULL ELSE (CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END) END"
    assert Agent._measure_label(wrapped) == "% below Level 2 in mathematics"
    assert "Level 6 >= 669.3" in link_errors.levels_prompt()


def test_intercept_gets_no_verdict_and_multi_measure_pairs():
    reg = pd.DataFrame({"CNT": ["CHL", "CHL"], "term": ["(intercept)", "ESCS"], "estimate": [455.2, 29.1],
                        "se": [2.4, 1.8], "n_pv": 10, "cycle": "2025"})
    facts = summ.fact_sentences(reg, {"template": "regression", "measure": "PV{pv}SCIE"}, {}, {"CHL": "Chile"}, str)
    assert "statistically significant" not in facts[0] and "statistically significant" in facts[1]
    multi = pd.DataFrame({"CNT": ["PHL", "SGP"] * 2, "measure": ["Mathematics score"] * 2 + ["Reading score"] * 2,
                          "estimate": [370.6, 562.9, 366.9, 534.9], "se": [1.7, 1.6, 2.4, 1.9], "n_pv": 10, "cycle": "2025"})
    facts = summ.fact_sentences(multi, {"template": "weighted_mean", "measures": ["PV{pv}MATH", "PV{pv}READ"]}, {},
                                {"PHL": "Philippines", "SGP": "Singapore"}, str, focus_codes=["PHL", "SGP"])
    pairs = [f for f in facts if " minus " in f]
    assert len(pairs) == 2 and all("statistically significant" in p for p in pairs)
    assert any("(Mathematics score)" in p for p in pairs) and any("(Reading score)" in p for p in pairs)
