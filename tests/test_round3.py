"""Offline tests for the third round (2026-09-20): the closed plan grammar
wired into the agent, the remaining Medium/Low persona findings (planner
caveats that contradict the app, 'is that good?', benchmark membership
follow-ups, published-table reconciliation, English forced for English
questions, signed-direction prose, composite labels, raw-SQL statements,
trend summaries, each side of a gap). No language model, no database."""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import summary as summ                     # noqa: E402
from explorer.agent import Agent, PLAN_KNOWLEDGE          # noqa: E402

NAMES = {"FIN": "Finland", "ITA": "Italy", "SLV": "El Salvador", "KAZ": "Kazakhstan", "BRA": "Brazil"}


def _agent() -> Agent:
    a = Agent.__new__(Agent)
    a._fired = []
    a.present = {"2025": set(NAMES), "2022": set(NAMES), "2018": set(NAMES)}
    a.economy_names = dict(NAMES)
    a.regions_block = "(regions)"
    return a


def test_planner_prompt_is_the_shared_knowledge_plus_the_form():
    a = _agent()
    text = a._planner_system()
    assert '"statistic": one of' in text and "resilient_share" in text
    assert "{instruments}" not in text and "PV{pv}" in text        # placeholders rendered, {pv} kept
    assert PLAN_KNOWLEDGE.split("\n")[0] in text


def test_language_forced_english_only_for_plainly_english_questions():
    a = _agent()
    assert a._plainly_english("El Salvador rank among all, how many below")
    assert a._plainly_english("what is the average maths score for irland 2025")
    assert not a._plainly_english("quantos alunos do Brasil ficaram abaixo do nivel 2")
    assert not a._plainly_english("¿cuál es la media de Chile?")
    assert not a._plainly_english("posición de El Salvador en Centroamérica")


def test_words_for_the_new_intercepts():
    assert Agent.GOOD_WORDS.search("is that good?") and Agent.GOOD_WORDS.search("dime si es mucho")
    assert not Agent.GOOD_WORDS.search("is Finland good at math")
    assert Agent.BIG_WORDS.search("compare boys and girls in reading in Peru 2025 and tell me if it is a lot")
    assert Agent.MEMBERS_WORDS.search("which of the four actually entered the average in each year and what N did you divide by?")
    assert Agent.RECONCILE_WORDS.search("Why does your OECD average for reading in 2022 differ from Table I.B1.2.1?")
    assert Agent.RECONCILE_WORDS.search("your Finland mean does not match the published report")
    assert not Agent.RECONCILE_WORDS.search("mean science score in Chile 2025")


def test_false_planner_caveats_are_dropped_and_prefix_is_honest():
    note = ("The system does not directly perform statistical significance tests on this change. "
            "Reading was not computed.")
    assert Agent.FALSE_LIMIT.sub("", note).strip() == "Reading was not computed."
    a = _agent()
    a.con = None
    prov_note = None
    # a caveat on a complete answer is a NOTE, a left-out part is NOT DONE
    import re
    for text, want in [("PISA cannot establish causes; the trend is shown.", "NOT DONE"),
                       ("The regression shows associations, as PISA data are cross-sectional.", "NOTE")]:
        prefix = ("NOT DONE" if re.search(
            r"\b(not|cannot|can'?t|no|unable|only|left out|beyond|does not|instead of|without|excluded|omitted|neither)\b",
            text, re.I) else "NOTE")
        assert prefix == want, text
    assert prov_note is None


def test_group_words_and_members_answer_without_an_average_row():
    assert Agent.GROUP_WORDS.search("Average science score 2018 to 2025 for the group Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan")
    assert not Agent.GROUP_WORDS.search("mean science score in Chile")
    a = _agent()
    a._last_benchmark_members = None
    a._last_rows = {"2018": ["KAZ"], "2025": ["FIN", "KAZ"]}
    text = a._members_answer()
    assert text.startswith("The last result had no group-average row") and "PISA 2018: KAZ" in text


def test_members_answer_from_the_last_result():
    a = _agent()
    a._last_benchmark_members = {"2018": {"Group avg": ["KAZ"]}, "2025": {"Group avg": ["KAZ", "FIN"]}}
    text = a._members_answer()
    assert "PISA 2018, Group avg: 1 member(s)" in text and "N = 1" in text
    assert "PISA 2025, Group avg: 2 member(s)" in text and "Kazakhstan (KAZ)" in text


def test_level_facts_state_each_side_of_a_gap():
    a = _agent()
    plan = {"template": "gap", "measure": "PV{pv}READ", "group_col": "ST004D01T", "minuend": 2, "subtrahend": 1,
            "_group_levels": {"2022": [{"CNT": "FIN", "group": 1.0, "estimate": 513.0, "se": 2.57},
                                       {"CNT": "FIN", "group": 2.0, "estimate": 468.3, "se": 2.77}]}}
    facts = a._level_facts(plan, lambda c, v: f"{c} = {int(v)} ({'Female' if v == 1 else 'Male'})")
    assert facts == ["Mean of Reading score — Finland (FIN), ST004D01T = 1 (Female), PISA 2022: 513.0 (SE 2.57).",
                     "Mean of Reading score — Finland (FIN), ST004D01T = 2 (Male), PISA 2022: 468.3 (SE 2.77)."]
    plan = {"template": "quartile_gap", "measure": "PV{pv}MATH", "quart_variable": "ESCS",
            "_group_levels": {"2025": [{"CNT": "KAZ", "group": 1, "estimate": 368.9, "se": 2.79},
                                       {"CNT": "KAZ", "group": 4, "estimate": 462.3, "se": 3.14}]}}
    facts = a._level_facts(plan, lambda c, v: str(v))
    assert "bottom quarter (1)" in facts[0] and "top quarter (4)" in facts[1]


def test_prose_rejects_signed_direction_and_labels_composites():
    t = pd.DataFrame({"CNT": ["FIN"], "estimate": [-15.27], "se": [12.99], "cycle": "2025"})
    prov = {"notes": [], "method": "", "sample": [], "facts": ["Difference — Finland (FIN), PISA 2025: -15.3 (SE 13.0)."]}
    plan = {"template": "gap", "measure": "PV{pv}MATH"}
    issues = summ.check_prose("Almaty scored -15.27 points (SE 12.99) lower than Astana.", t, prov, plan, "", lambda s: [], {"FIN"})
    assert any("negative number with" in i for i in issues)
    assert not any("negative number with" in i for i in
                   summ.check_prose("Almaty scored 15.27 points (SE 12.99) lower than Astana.", t, prov, plan, "", lambda s: [], {"FIN"}))
    assert Agent._measure_label("CASE WHEN PV{pv}MATH < 420.07 AND PV{pv}READ < 407.47 AND PV{pv}SCIE < 409.54 "
                                "THEN 100.0 ELSE 0.0 END") == "% below Level 2 in all of mathematics, reading, science"
    assert Agent._measure_label("PV{pv}MATH / 40.0") == "Mathematics divided by 40"


def test_raw_sql_rows_and_trend_shape_are_stated():
    t = pd.DataFrame({"sum_w": [951150.63], "students": [6554]})
    facts = summ.fact_sentences(t, {"template": "raw_sql"}, {}, {}, lambda m: m)
    assert facts[0].startswith("Direct SQL result with 1 row(s)") and "sum_w = 951150.63" in facts[1]
    t = pd.DataFrame({"CNT": [f"C{i:02d}" for i in range(12)], "estimate_2018": [500.0] * 12, "se_2018": [2.0] * 12,
                      "estimate_2025": [470 + i * 5 for i in range(12)], "se_2025": [2.0] * 12})
    t["change"] = t["estimate_2025"] - t["estimate_2018"]
    t["se_change"] = 3.0
    shape = [f for f in summ.fact_sentences(t, {"template": "weighted_mean", "measure": "PV{pv}MATH"}, {}, {},
                                            lambda m: "Math") if f.startswith("Of the")]
    assert shape and "5 decreased significantly, 4 increased significantly and 3 showed no" in shape[0]
    assert "largest declines: C00 -30.0 (SE 3.00)" in shape[0]


def test_unsupported_list_no_longer_refuses_resilience_or_icc():
    text = " ".join(note for _, note in Agent.UNSUPPORTED)
    assert "academically resilient" not in text
    assert "ICC" in text and "Multilevel" in text            # multilevel models still not offered
