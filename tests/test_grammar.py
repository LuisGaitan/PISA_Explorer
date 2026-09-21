"""Offline tests of the closed plan grammar (explorer/grammar.py): the
planner's form compiles to the legacy plan the engine runs, and anything
outside the form is refused with a message naming the field. No language
model, no database (the catalog is stubbed)."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import catalog, grammar   # noqa: E402

KNOWN = {"ESCS", "ST004D01T", "MALE", "IMMIG", "SC001Q01TA", "SC013Q01TA", "ST263Q02JA", "ST184Q01HA",
         "ICTWKDY", "REPEAT", "BELONG", "STRATUM", "CNTSCHID"}
PRESENT = {"EST", "POL", "ITA", "FIN", "KAZ", "JPN", "CHL", "COL", "MEX", "PER", "ARG", "BRA", "PRY",
           "URY", "DEU", "DNK", "QTJ"}
INVENTED = {"TJK": "QTJ", "TWN": "TAP"}


@pytest.fixture(autouse=True)
def stub_catalog(monkeypatch):
    def describe(var, cycle=None):
        if var in KNOWN:
            return pd.DataFrame({"variable": [var], "table_name": ["stu_qqq_2025"], "cycle": ["2025"],
                                 "label": [var], "var_type": ["double"], "value_labels": [None]})
        return pd.DataFrame(columns=["variable", "table_name", "cycle", "label", "var_type", "value_labels"])
    monkeypatch.setattr(catalog, "describe", describe)


def compile_(g):
    return grammar.compile_plan(g, INVENTED, PRESENT)


def test_measures_compile_to_the_engine_forms():
    assert grammar.compile_measure({"score": "MATH"}) == "PV{pv}MATH"
    assert grammar.compile_measure("reading") == "PV{pv}READ"
    assert grammar.compile_measure({"variable": "ESCS"}) == "ESCS"
    assert grammar.compile_measure({"level_share": {"domain": "MATH", "side": "below", "level": "2"}}) == \
        "CASE WHEN PV{pv}MATH < 420.07 THEN 100.0 ELSE 0.0 END"
    assert grammar.compile_measure({"level_share": {"domain": "SCIE", "side": "at_or_above", "level": "5"}}) == \
        "CASE WHEN PV{pv}SCIE >= 633.33 THEN 100.0 ELSE 0.0 END"
    joint = grammar.compile_measure({"level_share": {"domains": ["MATH", "READ", "SCIE"], "side": "below",
                                                     "level": "2", "combine": "all"}})
    assert "PV{pv}MATH < 420.07 AND PV{pv}READ < 407.47 AND PV{pv}SCIE < 409.54" in joint
    assert grammar.compile_measure({"code_share": {"variable": "ST263Q02JA", "codes": [1, 2]}}) == \
        "CASE WHEN ST263Q02JA IN (1, 2) THEN 100.0 WHEN ST263Q02JA IS NOT NULL THEN 0.0 END"
    assert grammar.compile_measure({"threshold_share": {"variable": "REPEAT", "op": ">=", "value": 1}}) == \
        "CASE WHEN REPEAT >= 1 THEN 100.0 WHEN REPEAT IS NOT NULL THEN 0.0 END"
    assert grammar.compile_measure({"score": "MATH", "divide_by": 40}) == "PV{pv}MATH / 40"
    assert grammar.compile_measure({"school_mean_of": "ESCS"}) == "AVG(ESCS) OVER (PARTITION BY CNTSCHID)"


def test_invented_variables_and_columns_are_refused():
    with pytest.raises(grammar.GrammarError, match="not a variable"):
        grammar.compile_measure({"variable": "ESCS_Q"})
    with pytest.raises(grammar.GrammarError, match="not in the catalog"):
        compile_({"statistic": "mean", "measures": ["MATH"], "by": ["CNT", "group_col"]})
    with pytest.raises(grammar.GrammarError, match="not in the catalog"):
        compile_({"statistic": "mean", "measures": ["MATH"], "filters": [{"variable": "ESCS_QUARTER", "op": "=", "values": [4]}]})
    with pytest.raises(grammar.GrammarError, match="unknown statistic"):
        compile_({"statistic": "pooled_mean", "measures": ["MATH"]})
    with pytest.raises(grammar.GrammarError, match="unknown proficiency level"):
        grammar.compile_measure({"level_share": {"domain": "MATH", "side": "below", "level": "7"}})
    with pytest.raises(grammar.GrammarError, match="number"):
        compile_({"statistic": "mean", "measures": ["MATH"], "filters": [{"variable": "IMMIG", "op": "in", "values": ["2; DROP TABLE x"]}]})


def test_economies_filters_and_invented_codes():
    p = compile_({"statistic": "mean", "measures": [{"score": "MATH"}], "economies": ["TJK", "KAZ"],
                  "filters": [{"variable": "IMMIG", "op": "in", "values": [2, 3]},
                              {"variable": "STRATUM", "op": "in", "values": ["KAZ21"]}]})
    assert p["template"] == "weighted_mean" and p["measure"] == "PV{pv}MATH"
    assert p["where"] == "CNT IN ('QTJ', 'KAZ') AND IMMIG IN (2, 3)"      # TJK -> QTJ, STRATUM left to the app
    assert p["_grammar_notes"] == ["stratum filter left to the app"]
    p = compile_({"statistic": "mean", "measures": ["SCIE"], "filters": [{"variable": "CNT", "op": "in", "values": ["JPN"]}]})
    assert p["where"] == "CNT = 'JPN'"


def test_contrasts_predictors_and_benchmarks():
    p = compile_({"statistic": "gap", "measures": ["READ"], "economies": ["FIN"], "cycles": ["2022"],
                  "contrast": {"variable": "ST004D01T", "minuend": [2], "subtrahend": [1], "label": "male minus female"}})
    assert (p["group_col"], p["minuend"], p["subtrahend"]) == ("ST004D01T", 2.0, 1.0)
    p = compile_({"statistic": "gap", "measures": ["SCIE"], "instrument": "stu_sch",
                  "contrast": {"variable": "SC001Q01TA", "minuend": [1, 2], "subtrahend": [4, 5, 6], "label": "rural minus city"}})
    assert p["group_col"] == "CASE WHEN SC001Q01TA IN (1, 2) THEN 1 WHEN SC001Q01TA IN (4, 5, 6) THEN 0 END"
    assert p["minuend"] == 1 and p["subtrahend"] == 0 and p["group_label"] == "rural minus city"
    p = compile_({"statistic": "gap", "measures": ["MATH"], "contrast": {"variable": "CNT", "minuend": ["FIN"], "subtrahend": ["EST"]}})
    assert p["group_col"] == "CNT" and p["minuend"] == "FIN"
    p = compile_({"statistic": "regression", "measures": ["MATH"], "economies": ["DEU"],
                  "predictors": [{"variable": "IMMIG", "codes": [2], "reference": [1], "label": "second-gen vs native"},
                                 {"variable": "ESCS", "label": "ESCS"}, {"school_mean_of": "ESCS"}]})
    assert p["predictors"][0] == "CASE WHEN IMMIG IN (2) THEN 1 WHEN IMMIG IN (1) THEN 0 END"   # NULL stays NULL
    assert p["predictor_names"] == ["second-gen vs native", "ESCS", "school average of ESCS"]
    # two dummies of one variable: a sibling category is 0, not missing
    ex, _ = grammar.compile_predictors([{"variable": "IMMIG", "codes": [2], "reference": [1]},
                                        {"variable": "IMMIG", "codes": [3], "reference": [1]}])
    assert ex == ["CASE WHEN IMMIG IN (2) THEN 1 WHEN IMMIG IN (1, 3) THEN 0 END",
                  "CASE WHEN IMMIG IN (3) THEN 1 WHEN IMMIG IN (1, 2) THEN 0 END"]
    m = grammar.compile_measure({"count_share": {"variables": ["ST263Q02JA", "ST184Q01HA"], "codes": [2], "op": "=", "count": 1}})
    assert m.startswith("CASE WHEN (CASE WHEN ST263Q02JA IN (2) THEN 1 ELSE 0 END + CASE WHEN ST184Q01HA IN (2)") and "= 1 THEN 100.0" in m
    p = compile_({"statistic": "mean", "measures": ["MATH"],
                  "benchmarks": ["OECD", "European Union", "world",
                                 {"label": "Mercosur", "members": ["ARG", "BRA", "PRY", "URY"]}]})
    assert p["include_oecd_average"] is True
    assert p["include_average_of"] == ["European Union", "All participants",
                                       {"label": "Mercosur", "members": ["ARG", "BRA", "PRY", "URY"]}]
    with pytest.raises(grammar.GrammarError, match="not a benchmark group"):
        compile_({"statistic": "mean", "measures": ["MATH"], "benchmarks": ["Pacific Alliance"]})


def test_overrides_sort_and_new_statistics():
    p = compile_({"statistic": "share", "cycles": ["2018", "2022"], "economies": ["EST"],
                  "measures": [{"code_share": {"variable": "ST184Q01HA", "codes": [1, 2]}}],
                  "cycle_overrides": {"2022": {"measure": {"code_share": {"variable": "ST263Q02JA", "codes": [1, 2]}},
                                               "by": ["CNT", "MALE"]}},
                  "sort": {"by": "estimate", "desc": False, "top_n": 5}})
    assert p["cycle_overrides"]["2022"]["measure"].startswith("CASE WHEN ST263Q02JA IN (1, 2)")
    assert p["cycle_overrides"]["2022"]["by"] == ["CNT", "MALE"]
    assert (p["sort_by"], p["sort_desc"], p["top_n"]) == ("estimate", False, 5)
    assert compile_({"statistic": "sd", "measures": ["MATH"]})["template"] == "weighted_sd"
    p = compile_({"statistic": "resilient_share", "measures": ["MATH"], "level": "3"})
    assert p["template"] == "resilient_share" and p["quart_variable"] == "ESCS" and p["level"] == 482.38
    p = compile_({"statistic": "correlation", "x": {"variable": "ICTWKDY"}, "y": "MATH", "quart_variable": "ESCS"})
    assert p["x"] == "ICTWKDY" and p["y"] == "PV{pv}MATH" and p["quart_variable"] == "ESCS"
    assert compile_({"statistic": "between_school_share", "measures": ["SCIE"]})["template"] == "between_school_share"
    with pytest.raises(grammar.GrammarError, match="share needs"):
        compile_({"statistic": "share", "measures": [{"score": "MATH"}]})
    assert compile_({"action": "clarify", "clarify": "which cycle?"})["action"] == "clarify"
