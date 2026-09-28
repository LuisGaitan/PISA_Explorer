"""Offline tests for the fourth round (2026-09-26, from the v23-v26 traffic):
the verified topic map behind "what variables do I use for X", deterministic
answers about data handling, the model and attribution, the cycle overview
for bare "pisa 2018", the nearest-answer rewrites (forecasts, years with no
cycle, ages PISA does not test) and the default chart. No language model."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer import topics                                  # noqa: E402
from explorer import grammar                                 # noqa: E402
from explorer.agent import Agent, OECD_ACKNOWLEDGEMENT, PLAN_SYSTEM   # noqa: E402
from explorer.db import CATALOG_DIR                          # noqa: E402

NAMES = {"FIN": "Finland", "MEX": "Mexico", "FRA": "France", "COL": "Colombia"}


def _agent() -> Agent:
    a = Agent.__new__(Agent)
    a._fired = []
    a.present = {"2025": set(NAMES), "2022": set(NAMES), "2018": set(NAMES)}
    a.economy_names = dict(NAMES)
    a.regions_block = "(regions)"
    a.coverage = {"2018": {"economies": 81, "students": 617772},
                  "2022": {"economies": 80, "students": 613744},
                  "2025": {"economies": 90, "students": 755721}}
    a.con = None
    return a


# ---------- topic map ----------

def test_topic_families_expand_and_display():
    assert topics.expand("PV1-10MATH") == [f"PV{i}MATH" for i in range(1, 11)]
    assert topics.expand("W_FSTURWT1-80")[-1] == "W_FSTURWT80"
    assert topics.expand("ESCS") == ["ESCS"]
    assert topics.display("PV1-10READ") == "PV1READ … PV10READ"


def test_topics_match_the_questions_students_ask():
    got = {t.construct for t in topics.matching(
        "What variables do I use for math scores and social emotional learning in 2022")}
    assert got == {"mathematics score", "social and emotional skills"}
    assert {t.construct for t in topics.matching("What variable holds the reading score in 2025?")} == {"reading score"}
    assert {t.construct for t in topics.matching("which variables should I use for student weights and replicate weights")} == {"weights"}
    assert "socio-economic status" in {t.construct for t in topics.matching("which variables measure socioeconomic status in 2018?")}
    # a domain word without a score word is not a score question
    assert not [t for t in topics.matching("reading enjoyment") if t.construct == "reading score"]
    # school tracks and programmes (a Tartu user asked for Mittelschule vs
    # Gymnasium and was told no variable existed: PROGN and ISCEDP do)
    for q in ("Is there also classification into Mittelschule and Gymnasium in Austria?",
              "compare general and vocational programmes in Italy",
              "what variable gives the ISCED level of the student's programme",
              "what types of schools are there in Germany's sample"):
        assert "study programme and school track" in {t.construct for t in topics.matching(q)}, q
    assert not [t for t in topics.matching("How do you track my data?")
                if t.construct == "study programme and school track"]
    track = [t for t in topics.TOPICS if t.construct == "study programme and school track"][0]
    assert track.variables == {"2018": ("PROGN", "ISCEDL", "ISCEDD", "ISCEDO"),
                               "2022": ("PROGN", "ISCEDP"), "2025": ("PROGN", "ISCEDP")}


def test_topic_answer_lines_name_the_codes_per_cycle():
    lines = topics.answer_lines(topics.matching("math scores and social emotional learning"), ("2022",))
    text = " ".join(lines)
    assert "PV1MATH … PV10MATH" in text and "PERSEVAGR" in text and "GROSAGR" in text
    assert "Rubin" in text                      # how to use the ten PVs
    # a topic absent from the cycle asked says where it exists
    lines = topics.answer_lines(topics.matching("growth mindset"), ("2025",))
    assert "not in PISA 2025" in lines[0] and "GROSAGR" in lines[0]


def test_topic_rows_carry_family_code_and_role():
    rows = topics.rows(topics.matching("reading score"), ("2018", "2022", "2025"),
                       label_of=lambda code, cycle: "Plausible Value 1 in Reading")
    assert rows[0]["variable"] == "PV1READ … PV10READ" and rows[0]["_code"] == "PV1-10READ"
    assert rows[0]["label"] == "Plausible values 1-10 in Reading"
    assert rows[0]["cycles"] == "2018, 2022, 2025" and rows[0]["role"] == "standard"


def test_every_topic_variable_exists_in_the_catalog_for_its_cycle():
    if not (CATALOG_DIR / "variables.parquet").exists():
        pytest.skip("no local catalog (CI)")
    cat = pd.read_parquet(CATALOG_DIR / "variables.parquet")
    have = set(zip(cat.variable, cat.cycle, cat.table_name))
    missing = [(t.construct, cycle, member)
               for t in topics.TOPICS for cycle, codes in t.variables.items()
               for code in codes for member in topics.expand(code)
               if (member, cycle, f"{t.instrument}_{cycle}") not in have]
    assert missing == []


def test_family_collapse():
    assert Agent._family_of("PV6MATH") == "PV1-10MATH"
    assert Agent._family_of("PV10CRTH_NC") == "PV1-10CRTH_NC"
    assert Agent._family_of("W_FSTURWT23") == "W_FSTURWT1-80"
    assert Agent._family_of("ESCS") is None and Agent._family_of("PVMATH") is None


def test_variable_questions_are_forced_to_explore():
    assert Agent.VARIABLE_ASK_WORDS.search("What variables do I use for math scores and social emotional learning in 2022")
    assert Agent.VARIABLE_ASK_WORDS.search("What variable holds the reading score in 2025?")
    assert Agent.VARIABLE_ASK_WORDS.search("which variables should I use for student weights")
    assert Agent.VARIABLE_ASK_WORDS.search("what is the variable name for ESCS")
    assert not Agent.VARIABLE_ASK_WORDS.search("mean science score in Finland in 2025")
    assert not Agent.VARIABLE_ASK_WORDS.search("gender gap in reading in Italy")


# ---------- data handling, model, attribution ----------

def test_privacy_words_and_answer_are_truthful():
    a = _agent()
    for q in ["How do you track my data?", "do you store my questions?", "what is your privacy policy",
              "is my conversation private?", "are my questions sent to google?", "do you use cookies"]:
        assert a._intercept(q) is not None, q
        assert "intercept:privacy" in a._fired
        a._fired.clear()
    for q in ["share of students in private schools in Chile", "how many students were tracked in 2022",
              "record of grade repetition in Mexico"]:
        assert not Agent.PRIVACY_WORDS.search(q), q
    text = a._privacy_answer()
    for must in ("records: every question", "institution", "Gemini", "aggregate results",
                 "never leave the server", "no student- or school-level record", "IP address"):
        assert must in text, must
    for never in ("does not track", "runs entirely in your browser", "no information about your questions"):
        assert never not in text, never


def test_model_and_attribution_answers():
    a = _agent()
    assert a._intercept("which AI model do you use?") is not None and "intercept:model" in a._fired
    assert a._intercept("are you chatgpt?") is not None
    assert "Gemini" in a._model_answer() and "never sees a student record" in a._model_answer()
    a._fired.clear()
    assert a._intercept("Does this tool give credits to OECD?") is not None
    assert "intercept:attribution" in a._fired
    assert a._intercept("how should I cite these results?") is not None
    assert a._intercept("is this an official OECD tool?") is not None
    text = a._attribution_answer()
    assert OECD_ACKNOWLEDGEMENT in text and "CY09MS" in text and "MIT" in text
    assert "not an OECD product" in text
    # "why does your figure differ from the OECD table" is a reconciliation, not attribution
    assert Agent.ATTRIBUTION_WORDS.search("why does your figure differ from the official OECD table") is None or \
        Agent.RECONCILE_WORDS.search("why does your figure differ from the official OECD table")


# ---------- bare cycle ----------

def test_bare_cycle_detection():
    a = _agent()
    assert a._bare_cycle("pisa 2018") == ["2018"]
    assert a._bare_cycle("PISA 2025 data") == ["2025"]
    assert a._bare_cycle("do you have the pisa 2018 dataset?") == ["2018"]
    assert a._bare_cycle("2022 results") == ["2022"]
    assert a._bare_cycle("pisa data") == []
    assert a._bare_cycle("what is pisa") is None            # an explanation, not an overview
    assert a._bare_cycle("pisa 2019") is None               # not a cycle: the nearest-answer rule
    assert a._bare_cycle("mean science score in Finland 2025") is None
    assert a._bare_cycle("pisa 2018 gender gap") is None


def test_cycle_answer_states_counts_domains_and_files():
    a = _agent()
    text = a._cycle_answer(["2018"])
    assert "81 economies" in text and "617,772" in text and "major domain: reading" in text
    assert "global competence" in text and "stu_qqq_2018" in text and "sch_qqq_2018" in text
    assert "PISADIFF" not in text
    text = a._cycle_answer(["2025"])
    assert "8 September 2026" in text and "Learning in the Digital World" in text
    assert "Yes." in a._cycle_answer([])                    # no year: the coverage answer


# ---------- nearest answer ----------

def test_nearest_answer_rules():
    a = _agent()
    notes, hints = a._nearest_answer(
        "given the available data, show me the expected reading performance score for Mexico for the next 10 years")
    assert notes and "cannot forecast" in hints[0] and "2018, 2022 and 2025" in hints[0]
    assert "no forecasting model" in notes[0]
    notes, hints = a._nearest_answer("give me reading scores for 16 year olds in France in 2019")
    assert any("no PISA 2019" in n for n in notes) and any("2018 and 2022" in n for n in notes)
    assert any("15-year-olds" in n for n in notes)
    assert len(hints) == 2
    notes, _ = a._nearest_answer("reading in France in 2015")
    assert "exists but is not loaded" in notes[0] and "2018" in notes[0]
    notes, _ = a._nearest_answer("math in Mexico in 2035")
    assert "no forecasting model" in notes[0]
    assert a._nearest_answer("regress math on ESCS with predictors gender and immigrant status in 2022") == ([], [])
    assert a._nearest_answer("mean science score in Finland in 2025") == ([], [])


def test_planner_prompts_carry_the_new_rules():
    # the grammar planner's rules (production) and the legacy planner's
    for text in (grammar.RULES, PLAN_SYSTEM):
        assert "FORECAST" in text and "AMBIGUOUS breakdown" in text
        assert "15-year-old sample" in text
    a = _agent()
    assert "never a silent choice" in a._planner_system()


def test_viz_default_note_and_words():
    assert Agent.VIZ_WORDS.search("give me a graph of Pisa data from 2025")
    assert "ranks every economy" in Agent.VIZ_DEFAULT_NOTE
    assert Agent.MAJOR_DOMAIN == {"2018": "reading", "2022": "mathematics", "2025": "science"}


# ---------- round 4b: fixes from the variable-question probe ----------

def test_variable_asks_tolerate_modifiers_and_missing_word():
    assert Agent.VARIABLE_ASK_WORDS.search("what weight and identifier variables do I need to merge student and school files?")
    assert Agent.VARIABLE_ASK_WORDS.search("which 2022 student questionnaire variables measure curiosity")
    assert Agent.DATA_TERM_ASK_WORDS.search("what's the senate weight?")
    assert Agent.DATA_TERM_ASK_WORDS.search("how do I merge student and school files")
    assert Agent.DATA_TERM_ASK_WORDS.search("where are the plausible values")
    assert not Agent.DATA_TERM_ASK_WORDS.search("mean science score in Finland in 2025")
    assert Agent.VARIABLE_ASK_WORDS.search("what's the exact score cutoff variable for level 2 math?")
    assert Agent.VARIABLE_ASK_WORDS.search("is there a global competence variable in 2022 or 2025?")
    assert Agent.VARIABLE_ASK_WORDS.search("what science competency subscales exist in 2025?")
    assert Agent.VARIABLE_ASK_WORDS.search("does the school questionnaire have a class size variable?")
    assert Agent.VARIABLE_ASK_WORDS.search("qué variables mido para el bienestar")
    assert Agent.VARIABLE_ASK_WORDS.search("quais variáveis medem bullying")


def test_invented_codes_are_caught_and_real_codes_are_recognized():
    if not (CATALOG_DIR / "variables.parquet").exists():
        pytest.skip("no local catalog (CI)")
    a = _agent()
    bad = a._invented_codes("use the school identifier (`SCHOOLID`) and the school weight (`W_FSCHWT`), "
                            "with W_FSTUWT for students; PISA and the OECD use ISCED levels.")
    assert bad == ["SCHOOLID", "W_FSCHWT"]
    assert a._invented_codes("The OECD average in PISA 2025 was 481.9 (SE 0.44).") == []
    assert a._invented_codes("Finland (FIN) and the USA are in the data.") == []
    assert a._catalog_codes_in("what is SENWT and how does it differ from W_FSTUWT?") == ["SENWT", "W_FSTUWT"]
    assert a._catalog_codes_in("how did FIN do in PISA") == []


def test_topic_guards_and_ordering():
    got = {t.construct for t in topics.matching("variables on height or weight (BMI)")}
    assert "weights" not in got
    assert "weights" in {t.construct for t in topics.matching("which variables hold the student weights")}
    assert "mathematics anxiety and self-efficacy" not in {t.construct for t in topics.matching("anxiety disorders")}
    assert "mathematics anxiety and self-efficacy" in {t.construct for t in topics.matching("math anxiety in 2022")}
    assert "classroom climate and teacher support" not in {t.construct for t in topics.matching("disciplinary action against students")}
    assert "enjoyment and motivation" not in {t.construct for t in topics.matching("interest in politics")}
    # the specific topic comes before the broad one
    order = [t.construct for t in topics.matching("is there a variable on AI use at school, from the ICT questionnaire?", ("2025",))]
    assert order.index("AI use") < order.index("ICT and digital devices")
    order = [t.construct for t in topics.matching("financial literacy and social emotional learning in 2022", ("2022",))]
    assert order[0] == "financial literacy"


def test_new_topics_exist():
    names = {t.construct for t in topics.TOPICS}
    for c in ("school staff composition", "remote instruction and the pandemic (2022)",
              "environmental education (2025)", "feedback from teachers", "parental involvement and support"):
        assert c in names
    assert "parental involvement and support" in {t.construct for t in topics.matching("what variables exist on parental involvement in 2025")}
    assert "environmental education (2025)" in {t.construct for t in topics.matching("environmental awareness variables 2025")}


def test_own_words_strip_ask_filler():
    a = _agent()
    assert a._own_words("variables on students' sleep") == ["sleep"]
    assert a._own_words("I need variables for a project on financial literacy and growth mindset in 2022") == \
        ["financial", "literacy", "growth", "mindset"]
    assert a._own_words("what variables do I use for math scores") == ["math", "scores"]


def test_latest_cycle_means_2025():
    import re
    assert re.search(r"\b(latest|most recent|newest|current|last) (cycle|round|pisa|year|assessment|data|edition)\b",
                     "variables on bullying in the latest cycle", re.I)


# ---------- round 4c: teacher / parent topics, typos, instrument overview ----------

def test_spelling_correction_touches_only_misspelled_common_words():
    a = _agent()
    assert a._spell("weigth varaibles for anaylsis") == "weight variables for analysis"
    assert a._spell("finantial literasy varibales pls") == "financial literacy variables pls"
    assert "hook:spell" in a._fired
    a._fired.clear()
    assert a._spell("mean science score in Finland in 2025") == "mean science score in Finland in 2025"
    assert a._spell("what is SENWT") == "what is SENWT"          # codes and capitals are kept
    assert a._fired == []


def test_instrument_overview_detection_and_answer():
    a = _agent()
    for q, kind in [("what's the teacher questionnaire table called?", "teacher"),
                    ("what variables come from the parent questionnaire?", "parent"),
                    ("is there a parent questionnaire in every cycle?", "parent"),
                    ("what does the school questionnaire contain?", "school")]:
        m = Agent.INSTRUMENT_ASK_WORDS.search(q)
        assert m and (m.group("kind") or m.group("kind2")).lower() == kind, q
    assert not Agent.INSTRUMENT_ASK_WORDS.search("what variables measure teacher self-efficacy?")
    assert not Agent.INSTRUMENT_ASK_WORDS.search("mean science score by school type in Chile")
    if not (CATALOG_DIR / "variables.parquet").exists():
        pytest.skip("no local catalog (CI)")
    # a questionnaire mention with a construct is a variable question, not an overview
    assert a._intercept("does the school questionnaire have a class size variable?") is None
    assert a._intercept("what variable tells me the administration mode (paper or computer)?") is None
    assert a._intercept("what's the teacher questionnaire table called?") is not None
    text = a._instrument_answer("teacher", [])
    assert "tch_qqq_<cycle>" in text and "19 economies in 2018" in text and "SEFFCM" in text
    text = a._instrument_answer("parent", ["2025"])
    assert "Option_PQ" in text and "PARINVOL" in text and "2018:" not in text


def test_teacher_parent_and_mode_topics():
    got = {t.construct for t in topics.matching("what variables measure teacher self-efficacy?")}
    assert got == {"teacher self-efficacy"}
    assert "teacher job satisfaction and well-being" in {t.construct for t in topics.matching("teacher job satisfaction variables")}
    assert "teacher professional development and training" in {t.construct for t in topics.matching("what variables cover teacher professional development?")}
    assert "teachers' use of digital resources" in {t.construct for t in topics.matching("is there a variable on how often teachers use digital tools?")}
    assert "administration mode (paper or computer)" in {t.construct for t in topics.matching("what variable tells me the administration mode (paper or computer)?")}
    assert "parental involvement and support" in {t.construct for t in topics.matching("is there a variable on parents attending school meetings?")}
    lines = topics.answer_lines(topics.matching("parents attending school meetings in 2018", ("2018",)), ("2018",))
    assert any("PA008Q05TA" in l for l in lines)
    lines = topics.answer_lines(topics.matching("teacher self-efficacy in 2025", ("2025",)), ("2025",))
    assert any("SETEACH" in l and "tch_qqq" in l for l in lines)


def test_spanish_and_portuguese_data_asks_route_to_explore():
    assert Agent.DATA_TERM_ASK_WORDS.search("cuáles son las variables de ponderación (pesos) para el análisis?")
    assert Agent.DATA_TERM_ASK_WORDS.search("quais são os valores plausiveis? where are the valores plausibles")
