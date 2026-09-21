"""The conversational layer: natural-language question -> catalog retrieval ->
Gemini fills an analysis plan -> survey-correct execution -> summary.

Design rules (each traces to a defect found in an earlier text-to-SQL prototype):
  - The LLM is a query planner, never a data container: it sees ~40 retrieved
    catalog cards, never a schema dump, never the data.
  - Statistics run through the validated templates in analysis.py; the LLM
    fills parameters, it does not derive weighting/PV/BRR logic.
  - Variable substitution is always explicit: plans carry a substitution_note
    that is surfaced to the user, never silent.
  - Every result carries PROVENANCE: source tables, variables with labels,
    filter, student counts, and the exact method that produced each number.
  - Raw SQL (escape hatch for questions the templates can't express) is
    read-only, SELECT-only, and shown to the user verbatim.
"""

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import catalog, llm, regions, standards
from . import summary as summ
from .version import BUILD, stamp
from .analysis import (
    correlation,
    crosstab,
    gap,
    percentile_spread,
    percentiles,
    quartile_gap,
    quartile_means,
    regression,
    trend,
    weighted_mean,
    weighted_proportion,
)
from .db import connect
from .llm import generate, generate_json
from . import link_errors

MAX_CARDS = 40
MAX_RESULT_ROWS = 500

INSTRUMENTS = ["stu_qqq", "sch_qqq", "tch_qqq", "stu_cog", "stu_tim", "stu_ttm",
               "flt_qqq", "flt_cog", "flt_tim", "crt_cog", "ldw_cog",
               "stu_sch",   # virtual: stu_qqq joined to sch_qqq
               "stu_crt"]   # virtual (2022): stu_qqq joined to crt_cog (creative thinking PVs + weights)
CYCLES = ["2018", "2022", "2025"]
DEFAULT_CYCLE = "2025"          # a question that names no cycle means the latest
# The acknowledgement the PISA public-use-file terms of use require, verbatim,
# followed by the file identifiers.
OECD_ACKNOWLEDGEMENT = ("Programme for International Student Assessment (PISA) "
                        "Organisation for Economic Co-operation and Development (OECD), Paris")
SOURCE_LINE = (OECD_ACKNOWLEDGEMENT + " — public-use databases 2018 (CY07MSU), 2022 "
               "(CY08MSP), 2025 (CY09MS); OECD disclaimers apply "
               "(oecd.org/en/about/terms-conditions/oecd-disclaimers)")
ESTIMATE_COL = re.compile(r"^estimate(_\d{4})?$")

FORBIDDEN_SQL = re.compile(
    r"\b(copy|attach|detach|install|load|pragma|export|import|create|insert|"
    r"update|delete|alter|drop|call|set|reset)\b", re.IGNORECASE)

TERMS_SYSTEM = """You are the router inside PISA Explorer, an independent web app
(not an OECD product) where a user chats with the OECD PISA 2018/2022/2025
public-use databases. How the app works (answer
questions about it truthfully): the user's question becomes an analysis plan;
validated survey statistics run locally; the app itself then AUTOMATICALLY
renders a chart when the result has a chartable shape (group comparisons ->
bar chart with 95% CI whiskers; cross-cycle 2018/2022/2025 -> dumbbell chart
with one dot per cycle; group gaps -> diverging bars), plus the data table, a
CSV export button, and a provenance
card. Single-number results and raw-SQL results get a table but no chart — to
get a chart, the user should ask for a comparison across groups, countries, or
cycles. Never claim "I cannot visualize data": charts appear automatically for
comparison-shaped results.

Reply with JSON: {"data_question": bool, "intent": "analyze"|"explore",
"search_terms": [str, ...], "direct_answer": str|null,
"language": str, "question_en": str}.
"language" = the English name of the language the question is written in
("English", "Japanese", "Spanish" …); "question_en" = a faithful English
rendering of the question (the question itself when it is English) — the
app's checks run on it, so keep economy names, years and variable codes
exact. search_terms are always English. direct_answer is written in the
user's language.
If the question needs data — including a follow-up that continues or answers a
clarification from the conversation context — set data_question=true and give
2-6 short catalog search terms (constructs, topics, variable ideas — e.g.
"gender", "mathematics", "socioeconomic status", "bullying", "immigrant").
intent="explore" when the user wants to know WHAT DATA EXISTS — "what data /
variables / questions do you have about X", "find me data on X", "is there
anything on X", "which indices cover X" — the app then lists the matching
catalog variables (no statistic is computed), so give 3-8 search terms that
cover the topic broadly (synonyms and related constructs). intent="analyze"
when the user wants a number, share, comparison, ranking, trend or
relationship. If it is conversational, about PISA in general, or about this
app's abilities (no data needed), set data_question=false and write
direct_answer.

direct_answer speaks AS the app ("PISA Explorer"), never as a language model:
never say "I am an AI / text-based model", never mention routing or search
terms. Example:
  User: "can you show me a visualization / chart of that?"
  -> {"data_question": false, "search_terms": [],
      "direct_answer": "Charts appear automatically when a result compares
      groups: country or group comparisons draw bars with confidence whiskers,
      cross-cycle questions draw dumbbells, and gaps draw diverging bars.
      The last result was a single number, which has no chart form — ask for a
      comparison instead (for example: 'compare reading scores for students
      with high vs low ICT autonomy, top vs bottom quartile, by country') and
      the chart will render with it."}
A follow-up like "yes, top vs bottom quartile" that answers a clarifying
question IS a data question — combine it with the context and route it to data."""

PLAN_SYSTEM = """You plan analyses of the OECD PISA 2018, 2022 and 2025 databases.

DATABASE: DuckDB. Tables are <instrument>_<cycle>, e.g. stu_qqq_2018, stu_qqq_2022,
stu_qqq_2025 (instruments: {instruments}; cycles: 2018, 2022, 2025).
stu_sch_<cycle> = every student row joined to its school's questionnaire
(sch_qqq, one row per school, by CNTSCHID): use instrument "stu_sch" whenever
a SCHOOL variable (a sch_qqq card: SC…, school type, location, size,
resources) is combined with a STUDENT outcome or grouping — e.g. public vs
private gap => instrument stu_sch, template gap, group_col SC013Q01TA. All
student columns, weights (W_FSTUWT) and PVs are there unchanged; a school
variable never exists in stu_qqq itself. STANDARD SCHOOL VARIABLES — these
exist in sch_qqq in EVERY cycle and you may use them whether or not a card
lists them (they count as "named above"): SC013Q01TA public (1) vs private
(2) school as reported by the principal — ALWAYS use it for "public vs
private" (minuend=2, subtrahend=1 = private minus public); do not substitute
PRIVATESCH (57 economies in 2025) or SCHLTYPE (1 private independent / 2
private government-dependent / 3 public) unless the user asks for the
government-dependent distinction; SC001Q01TA school location: 1 village/rural
(<3,000), 2 small town, 3 town, 4 city (100,000-1M), 5 large city, 6 megacity
(2022/2025) — "rural vs urban/city" => gap with a CASE group_col, e.g. rural
= SC001Q01TA IN (1, 2) vs city = SC001Q01TA >= 4, and group_label MUST name
what is left out ("village/small town (<15,000) minus city (>100,000);
towns of 15,000-100,000 excluded"); EDUSHORT / STAFFSHORT
shortage indices (all cycles); CLSIZE class size; SCHSIZE school size and
STRATIO student-teacher ratio (2018 and 2022 only).
AI USE (2025 only, student questionnaire, 84 of 90 economies): ST438Q01DA–
ST438Q04DA "How often do you use AI chatbots (e.g. ChatGPT) for your school
work to …" (1 never … 5 every day or almost; use weighted_proportion of a
code or weighted_mean of the code), AIUSESCH = students' AI use at school
(WLE index); IC170Q10DA / IC171Q10DA AI tools at / outside school (ICT
questionnaire, 44 economies). No cycle assessed AI literacy. Student
questionnaire (stu_qqq_*) holds achievement plausible values PV1..PV10 for
MATH/READ/SCIE, final weight W_FSTUWT, replicate weights, ESCS (socio-economic
index), and CNT (ISO-3 country code, e.g. 'USA', 'KOR', 'DEU').
There is also escs_trend (OECD comparable-ESCS across cycles).

ECONOMY CODES that are NOT ISO-3 (use exactly these): QCI = B-S-J-Z (China:
Beijing, Shanghai, Jiangsu, Zhejiang; 2018 and 2025 only), TAP = Chinese
Taipei, MAC = Macao (China), HKG = Hong Kong (China), KSV = Kosovo, QAT =
Qatar, QAZ = Baku (Azerbaijan; 2018/2022) vs AZE = Azerbaijan (2025), QUR =
Ukrainian regions (2022) / QUA = Ukrainian regions (2025), QKI = Kurdistan
Region (Iraq; 2025), QTJ = Dushanbe (Tajikistan; 2025), QMC = Moscow City (2018,
released separately by the OECD), QMR = Moscow region and QRT =
Tatarstan (2018), RUS = Russia (2018). Never invent codes: an economy is a
single CNT value. Not every economy is in every cycle (e.g. ARM, KEN, KGZ,
MUS, RWA, ZMB, ECU joined in 2025; JAM only 2022; BIH, BLR, UKR only 2018).

GENDER: ST004D01T (1=Female, 2=Male) in 2018 and 2022. In 2025 fourteen
economies (ARG, AUS, BEL, CAN, CHL, COL, DEU, DNK, ESP, IRL, ISL, NLD, NZL,
URY) release only the derived flag MALE (1=Male, 0=Female/Other), which is
complete for all 90 economies — so for 2025 use MALE (gap: group_col="MALE",
minuend=1, subtrahend=0; dummy: "CASE WHEN MALE = 1 THEN 1 ELSE 0 END"). In a
plan that spans 2018/2022 AND 2025, keep ST004D01T in the main fields and put
the 2025 variant in cycle_overrides (see below). Using MALE in 2025 is the
standard convention for EVERY economy (it is complete for all 90), not a
missing-data workaround. The contrast must point the SAME way in every
cycle: male minus female is minuend=2, subtrahend=1 with ST004D01T and
minuend=1, subtrahend=0 with MALE (the app re-aligns a mismatch, but plan it
right) — do not claim ST004D01T was not collected for an
economy unless a card says so; the substitution_note should just say that
2025 uses the derived MALE flag.

PISA 2025 SPECIFICS: 90 economies (80 in 2018/2022). Science was the major
domain; the same proficiency-level cutoffs apply in every cycle (the scales are
linked). Uzbekistan (UZB) took all three tests in 2025 but the OECD released
only its SCIENCE results (mathematics and reading plausible values are not in
the public database — its rows show no estimate for those domains; never say
they were "not collected"). Extra 2025 PVs:
science competency subscales PV{{pv}}SEPS / PV{{pv}}SEDE / PV{{pv}}SEID,
environmental science PV{{pv}}SENV, and the new "Learning in the Digital World"
domain PV{{pv}}CMPS (computational problem solving), PV{{pv}}CPPK
(computational practices / prior knowledge), PV{{pv}}CMOD, PV{{pv}}CPRO
(computer-based economies only). The 2025 student file also carries the
ICT-familiarity (IC*) and parent-questionnaire (PA*) items; tch_qqq_2025
covers 19 economies. stu_ttm = cognitive item process data (2018, 2025);
stu_tim = questionnaire timing (all cycles); ldw_cog_2025 = LDW item
responses and process data (scored items P1M…S, timing …TT, actions …A) —
LDW *scores* are the PV{{pv}}CMPS etc. columns in stu_qqq_2025.

You receive VARIABLE CARDS retrieved from the official codebooks — the only
variables you may use besides the ones named above. If the ideal variable is
not among them, use the closest available AND explain the substitution in
substitution_note. Never silently substitute. If nothing fits, action="clarify".

Reply with ONLY this JSON:
{{
 "action": "analyze" | "clarify",
 "clarify": str|null,
 "template": "weighted_mean"|"weighted_proportion"|"gap"|"quartile_means"|
             "quartile_gap"|"correlation"|"percentiles"|"percentile_spread"|
             "crosstab"|"regression"|"raw_sql",
 "cycles": chronological list drawn from ["2018","2022","2025"] — e.g.
            ["2025"] (latest; the default when no cycle is named),
            ["2018","2022","2025"] (a full trend), ["2022","2025"] (the latest
            change),
 "cycle_overrides": {{"<cycle>": {{field: value, ...}}}}|null   (per-cycle
            replacements for plan fields when a variable differs by cycle,
            e.g. {{"2025": {{"group_col": "MALE", "minuend": 1, "subtrahend": 0}}}};
            the system merges them into the plan for that cycle only and
            states them in the provenance),
 "instrument": "stu_qqq" etc.,
 "measure": SQL expression; write {{pv}} for the plausible-value slot, e.g. "PV{{pv}}MATH",
            or a plain expression like "ESCS" (weighted_mean and gap only),
 "measures": [str, ...]|null   (weighted_mean only: SEVERAL measures for the same
            groups and cycles — "math, reading and ESCS for Argentina", "a
            profile of X" — one SQL expression each, e.g. ["PV{{pv}}MATH",
            "PV{{pv}}READ", "ESCS"]; the system runs each and stacks the rows
            with a `measure` column, so nothing asked for is dropped),
 "group_label": str|null   (gap with a CASE group_col: readable name of the
            contrast, e.g. "non-immigrant minus immigrant"),
 "variable": str|null, "value": num|null, "valid_values": [nums]|null   (weighted_proportion:
            % of valid respondents with variable=value; valid_values = the non-missing codes),
 "group_col": str|null, "minuend": num|null, "subtrahend": num|null     (gap: mean of
            minuend group minus subtrahend group),
 "quart_variable": str|null   (quartile_means / quartile_gap: the CONTINUOUS
            variable whose weighted within-group quartiles define the groups,
            e.g. "ESCS"; quartile_means returns the mean of `measure` in each
            quarter 1..4, quartile_gap returns top minus bottom quarter),
 "x": str|null, "y": str|null   (correlation: the two variables/expressions;
            {{pv}} allowed, e.g. x="AUTICT", y="PV{{pv}}READ" — this is the
            WEIGHTED correlation with a proper BRR standard error; always
            prefer it over raw_sql corr()),
 "ps": [ints]|null   (percentiles: which weighted percentiles of `measure`,
            default [10,25,50,75,90]),
 "upper": int|null, "lower": int|null   (percentile_spread: P<upper> minus
            P<lower> of `measure`, default 90 and 10 — the standard
            within-country inequality/dispersion measure),
 "row_var": str|null, "col_var": str|null, "valid_rows": [nums]|null,
 "valid_cols": [nums]|null   (crosstab: weighted two-way table — within each
            row_var category, the % in each col_var category (row % sum to
            100), each cell with its own SE; list the valid non-missing codes;
            max 12 categories per side),
 "predictors": [str, ...]|null   (regression: weighted least squares of
            `measure` (the outcome, {{pv}} allowed) on up to 6 predictor
            expressions; encode binary contrasts as 0/1 dummies, e.g.
            "CASE WHEN ST004D01T = 2 THEN 1 ELSE 0 END" for a male dummy;
            use for "controlling for" questions),
 "predictor_names": [str, ...]|null   (regression, REQUIRED with predictors:
            short readable labels, parallel to predictors, stating what a
            positive coefficient means — e.g. ["ESCS (socioeconomic index)",
            "male (1) vs female (0)"]; these become the term names everyone
            reads, so make the direction unambiguous),
 "include_oecd_average": bool|null   (when the user wants per-country results
            PLUS the OECD average: the system appends a CNT = "OECD avg" row —
            the official convention, the unweighted mean of OECD member
            countries' estimates; requires "CNT" in by),
 "include_average_of": [str, ...]|null   (BENCHMARK rows: "OECD" and/or a
            region name from the REGIONS list — "European Union", "Latin
            America and the Caribbean", "Nordic countries" … — or an explicit
            list of CNT codes for an ad-hoc group; the system appends one
            "<group> avg" row per cycle (unweighted mean of the member
            economies present in that cycle, same convention as the OECD
            average). "France vs the EU average" => where = "CNT = 'FRA'",
            include_average_of = ["European Union"]. NEVER build an average
            by listing members in `where` — that only returns the members'
            rows. Two or more NAMED ad-hoc groups ("Pacific Alliance avg and
            Mercosur avg") are objects, one per group, never one merged list:
            include_average_of = [{{"label": "Pacific Alliance", "members":
            ["CHL","COL","MEX","PER"]}}, {{"label": "Mercosur", "members":
            ["ARG","BRA","PRY","URY"]}}]. "The average of A, B, C" as ONE
            number is also a benchmark row with by = ["CNT"] and where = the
            members — never by = [] (a pooled student mean across economies
            is no OECD statistic). A benchmark the user established earlier in the
            conversation stays in every later plan of that conversation
            until the user drops it; requires "CNT" in by),
 "by": [grouping columns, usually ["CNT"]],
 "regions": [str, ...]|null   (a geographic/institutional group the user
            names — "Latin America", "Asia", "the EU", "Nordic countries",
            "OECD" — chosen from the REGIONS list below; the system expands it
            to the exact economies present in each cycle and states them. Use
            this INSTEAD of enumerating a region's codes in `where`; combine
            with `where` only for extra conditions),
 "where": SQL boolean filter or null, e.g. "CNT IN ('USA','KOR')",
 "sql": str|null  (raw_sql only: one read-only SELECT; use this ONLY when no
        template fits — raw SQL gets no automatic weighting/PV/BRR treatment),
 "sort_by": "estimate"|"change"|null, "sort_desc": bool, "top_n": int|null
        (for ranking/top-N/bottom-N questions: the system sorts the result and
        keeps top_n rows, so the reported rows ARE the answer),
 "substitution_note": str|null,
 "limitation_note": str|null   (when the question also asks for something no
            template can deliver — causes, "factors behind" a change, reasons,
            predictions, policy effects — still run the computable part and
            state here, in one or two sentences, what was NOT done and why:
            PISA is a repeated cross-section, so it cannot establish what
            caused a change; within a cycle the app can estimate associations
            with the regression, correlation or quartile_gap templates),
 "explanation": one sentence of what will be computed
}}

REGIONS available for the `regions` field (members present per cycle):
{regions}
"OECD" is also accepted (the data's own OECD membership flag).

Reply with EXACTLY ONE JSON object — never a JSON array, never multiple plans.
SEVERAL MEASURES asked at once (averages of different variables side by side,
"math, reading and socio-economic status for X", "give me all of these for
X") => weighted_mean with "measures" listing every one of them (PV{{pv}}…
for scores, the index name for ESCS-type indices). NEVER drop a requested
measure silently: whatever cannot be computed in the same plan (a different
template, a variable no card covers) is named in limitation_note so the
answer says what was left out — never raw_sql for this.

Rules: a COMPOUND question (a computable analysis plus a "why" / "which
factors" / "what explains it" part) => action="analyze" the computable part
and fill limitation_note — never answer with action="clarify" just because
one part is out of reach; a partial answer with a stated limitation beats a
question back to the user.
COVERAGE: a card line "NOT collected for X" or "collected in only N of K
economies" means the variable belongs to an optional questionnaire (or a
national option) that only those economies administered — for the others
every value is missing and the estimate would be empty. Never plan such a
variable for an economy that did not collect it. Instead: (a) if SOME of the
named economies collected it, analyze those and list the others in
limitation_note ("not collected for X") — a partial answer beats a question
back; (b) if a DIRECT measure of the same construct exists for the economy in
another cycle (e.g. the 2018 well-being indices SWBP / BELONG when 2022 EXPWB
or 2025 has nothing direct), use that cycle and say so in the explanation —
this outranks the "no cycle named = 2025" default; (c) only when no card
measures the construct for the economy in any cycle, action="clarify" and
tell the user which economies do have it. A variable that measures a
DIFFERENT construct is not a substitute: never present, say, a conflict-
resolution item as a "proxy for well-being". State coverage facts only as the
cards give them: never claim a variable "was not collected" for an economy
unless its card says NOT collected for that economy.
Several explanatory variables against ONE outcome ("association between A, B
and math scores") => action="analyze" with template regression, predictors =
every variable named (with predictor_names) — never a question asking which
one to analyze first. A gap, quartile_gap, correlation or regression asked
for SEVERAL subjects ("the gap in each subject", "math and reading") => run
it for the first subject named (science when none is named) and list the
other subjects in limitation_note — never clarify to ask which one first.
"What is behind / what explains a decline" => the trend of the score itself
(weighted_mean over the cycles) with limitation_note; a regression across
cycles does not answer it (its intercept is not the mean score).
"X and the top N" / "X compared with the best countries" => ONE ranking plan
over all economies (by ["CNT"], sort_by "estimate", no where filter, top_n
null): the app locates X's row and rank in the full table — never a
clarification asking to do it in two steps.
A claim to check that needs two constructs ("high curiosity but low growth
mindset in 2025") when only one exists in that cycle => analyze the one
that exists (CURIO 2025) and state in limitation_note that the other is
available only in another cycle (GROSAGR 2022) — never clarify.
"How did X do" / "X's results" / "X's performance" with no domain named =>
weighted_mean with measures ["PV{{pv}}MATH", "PV{{pv}}READ", "PV{{pv}}SCIE"]
(all three domains, science listed last as the 2025 major domain), not
science alone.
A SHARE of students defined by a condition ("repeated a grade only once",
"below Level 2", "answered yes") => template weighted_mean with a measure
of the form CASE WHEN <condition> THEN 100.0 WHEN <valid but not the
condition> THEN 0.0 ELSE NULL END — always 100.0 / 0.0 (a percentage),
never 1 / 0, and never a CASE expression in weighted_proportion's
`variable` field (that field takes a plain variable code). If they cannot enter one regression, run correlation on
the first and put the rest in limitation_note.
The "clarify" text is shown verbatim to a non-technical reader: plain
language, variables named by their label with the code in parentheses, no
{{pv}}, no SQL, no JSON.
A two-group difference whose groups are SETS of codes (immigrant = IMMIG
2 or 3 vs non-immigrant = IMMIG 1; top vs bottom ESCS quarter is quartile_gap
instead) => template gap with group_col = a CASE expression yielding 1/0,
e.g. "CASE WHEN IMMIG = 1 THEN 1 WHEN IMMIG IN (2, 3) THEN 0 END",
minuend=1, subtrahend=0, and group_label = "non-immigrant minus immigrant".
Sample sizes ("how many students were tested / sampled in X") come with
EVERY answer automatically (the provenance lists sampled students and the
weighted 15-year-old population per table): never write raw_sql to count
students — plan the other statistic the question asks for (or weighted_mean
of PV{{pv}}SCIE when nothing else is asked) and say in the explanation that
the sample size accompanies the answer.
A period longer than the loaded cycles ("past decade", "since 2015/2012")
=> analyze 2018-2025 and say in the explanation that earlier cycles are not
loaded here.
Comparisons across cycles => list every cycle
asked about, in chronological order ("over time" / "trend" / "since 2018" =>
all three unless the user narrows it); the system runs the template per
cycle, reports every cycle side by side, and adds the change from the FIRST
to the LAST listed cycle. Questionnaire INDICES (WLE scales such as BELONG,
CURIO, AIUSESCH, and ESCS) are standardized within each cycle (OECD mean 0,
SD 1): a trend request on them still lists the cycles side by side, but the
app BLANKS the change column and says why — plan it, and say in the
explanation that cross-cycle changes in such indices are not comparable.
Test scores (PV…) and shares of a response code ARE comparable across cycles.
When the question names no achievement domain, use
science (the 2025 major domain) and say so in the explanation. A question that names no cycle means the latest one (2025) — say so in
the explanation. Achievement questions always use the PV{{pv}} form. Filter to
the countries the user names; if none named, ask yourself whether all
economies (80 in 2018/2022, 90 in 2025) is really wanted — for rankings it is.
Percentages of a category => weighted_proportion with valid_values listed.
"High vs low X" for a continuous X => quartile_gap (never invent thresholds).
Distribution / spread / inequality within groups => percentiles or
percentile_spread. Two categorical variables against each other => crosstab.
"Controlling for" / "after accounting for" => regression.
Share below/above a PISA proficiency level => weighted_mean with a threshold
measure, e.g. "% below Level 2 in math" => measure =
"CASE WHEN PV{{pv}}MATH < 420.07 THEN 100.0 ELSE 0.0 END"; "top performers"
= Level 5 or above (>= the Level 5 bound); "low achievers" = below Level 2.
Proficiency-level LOWER BOUNDS (use these exact values, never invent one):
{levels}
A share or mean WITHIN ESCS quarters ("% below Level 2 among the top ESCS
quarter", "repetition rate by ESCS quarter") => template quartile_means with
quart_variable = "ESCS" and the share as the measure (a 100/0 CASE): one row
per quarter and economy — never a clarify about filtering by quarter.
"P90-P10", "P75-P25", interdecile / interquartile range, "dispersion",
"spread", "inequality of scores" => template percentile_spread (upper,
lower) — never percentiles followed by a subtraction in the summary.
"World average" / "global average" / "average of all participants" =>
include_average_of = ["All participants"] (the app averages every economy
in the cycle); the OECD average is a DIFFERENT benchmark (member countries
only) — never substitute one for the other silently.
"Which policy works best" / "should countries do X" / "effect of a policy"
=> analyze the within-cycle association (regression or correlation, latest
cycle unless one is named) and put in limitation_note that PISA cannot
evaluate policies causally — never a cross-cycle regression.
DIFFERENCES BETWEEN ECONOMIES are never a reason to clarify: "gap in A minus
gap in B", "did the gap shrink more in A than in B", "difference between A
and B and did it change", "A's change minus B's change", "which countries
are statistically tied with A" => plan the statistic for BOTH (or all)
economies in one table (where = "CNT IN ('A', 'B')", by = ["CNT"]; a gap
stays template gap); the app itself then states every pairwise difference,
the change in a difference (= the difference of the two changes), and the
economies not statistically different from a named one, each with its SE.
Never write that the system "cannot compute" such a difference."""

SUMMARY_SYSTEM = """You summarize PISA analysis results for a general audience.
Write 2-5 sentences. Cite the key numbers with their standard errors like
"465 points (SE 4.0)". Copy every number exactly as it appears in the table
— never round further (an index of 0.28 with SE 0.02 stays 0.28 and 0.02). Treat |estimate| > 1.96*SE as statistically significant
and say so in plain language. In a multi-cycle table the columns are
estimate_<cycle> per cycle and `change` = last cycle minus first cycle;
describe the path over time (e.g. 2018 → 2022 → 2025), not just the endpoints.
All three cycles — including PISA 2025, released in 2026 — are real, published
survey results: never describe 2025 figures as projections, forecasts or
expectations, and do not remark on their being real (just report them as
you would any other cycle). Never attribute a change or a difference to causes, factors or
policies that are not in the table: if the notes say causes cannot be
established from PISA, say that explicitly in one sentence and point to what
the app can estimate instead. If a substitution_note or method note is present,
state it plainly. Do not invent numbers not in the table. Name variables by
their labels in plain words, never by codes like PV1MATH or ST004D01T. If the
notes say a variable was not collected for an economy, say exactly that (it
was not administered there), never that the economy "has no well-being" or
similar. If the notes say the OECD did not RELEASE an economy's results for
a cycle, say that — never that the economy "did not administer" or "did not
participate in" the assessment. QCI is "B-S-J-Z (China)", the four provinces
Beijing, Shanghai, Jiangsu and Zhejiang together: never call it Shanghai,
Beijing or China. Rows whose group value is missing were excluded (see
notes): never describe an excluded group as an "overall" figure. If the
prompt lists SAMPLE SIZES, use them when the question asks how many students
were tested or sampled; they count the students inside the question's filter
(the economies and groups in the table, per table), never the whole cycle —
"6,573 students sampled in Chile", not "in the 2025 cycle". A percentage
table's `category` column states exactly which group the percentage refers
to (e.g. "MALE = 0 (Female/Other)" = the share who are girls): report that
group's share as given and never subtract it from 100 to describe the other
group unless the table also holds that row. When the `change` column is blank
for a measure, the notes explain that the index is standardized within each
cycle: report each cycle's level and explicitly say the change is not
comparable — never write that it rose, fell or changed significantly. In a
regression table the "(intercept)" row is the predicted score when every
predictor is 0 — never call it the country's performance or mean score;
report the predictor coefficients as associations. Describe every variable
by the label the verified statements give it; if the user's own wording
describes something else (a different construct, the other direction of a
share), say so in one clause and use the app's label. A share "in (1, 2)"
whose codes are labelled Disagree is a share who DISAGREE — never flip the
direction. A group's value label in a statement ("ST004D01T = 1 (Female)")
is the group the number belongs to. If the verified statements list
economies "not statistically different from" a named one, or a rank range,
report them as given. Never say the app "cannot compute" a difference or
its significance: the verified statements contain every pairwise
difference the app computed, and if one is not there, say it was not
computed."""


class CoverageError(ValueError):
    """Every requested cycle lacks a plan variable for every named economy —
    the honest answer is a statement of coverage, not an empty table."""

    def __init__(self, message: str, provenance: dict | None = None):
        super().__init__(message)
        self.provenance = provenance


@dataclass
class AgentResult:
    question: str
    answer: str
    plan: dict | None = None
    table: pd.DataFrame | None = None
    provenance: dict | None = None
    retrieved: pd.DataFrame | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    route: str = "data"          # conversational | clarify | explore | data | error
    timing: dict = field(default_factory=dict)   # total_ms, llm_calls, llm_ms
    guards: list[str] = field(default_factory=list)   # intercepts/hooks that fired
    summary_mode: str | None = None    # llm | llm-retry | app-authored
    prose_issues: list[str] = field(default_factory=list)   # first draft's problems

    def analytics(self) -> dict:
        """The loggable, PII-free summary of this exchange (question text
        included — testers are told questions are recorded)."""
        plan = self.plan or {}
        prov = self.provenance or {}
        where = str(plan.get("where") or "")
        countries = sorted(set(re.findall(r"'([A-Z]{3})'", where)))
        return {
            "question": self.question,
            "route": self.route,
            "template": plan.get("template"),
            "instrument": plan.get("instrument"),
            "cycles": plan.get("cycles"),
            "by": plan.get("by"),
            "countries": countries,
            "all_countries": bool(plan.get("by")) and "CNT" in (plan.get("by") or [])
                             and not countries,
            "variables": sorted({v["variable"] for v in prov.get("variables", [])}),
            "rows": int(len(self.table)) if self.table is not None else 0,
            "raw_sql": plan.get("template") == "raw_sql",
            "substitution": bool(plan.get("substitution_note")),
            "regions": plan.get("regions"),
            "limitation": bool(plan.get("limitation_note")),
            "answer": (self.answer or "")[:4000],
            "oecd_average": bool(plan.get("include_oecd_average")),
            "benchmarks": plan.get("include_average_of"),
            "missing_rows": any("no estimate" in n for n in prov.get("notes", [])),
            "error": self.error,
            "answer_chars": len(self.answer or ""),
            "guards": list(self.guards),
            "summary_mode": self.summary_mode,
            "prose_issues": list(self.prose_issues)[:10],
            "build": BUILD,
            **self.timing,
        }


class Agent:
    # What is actually loaded — asserted to the router so the model's training
    # memory ("PISA 2025 is upcoming") never overrides reality. Innovative
    # domains per cycle are stated because the model tends to invent one.
    INNOVATIVE = {"2018": "global competence (questionnaire only in the public file)",
                  "2022": "creative thinking (crt_cog_2022)",
                  "2025": "Learning in the Digital World (LDW plausible values "
                          "PV1-10 CMPS/CPPK/CMOD/CPRO in stu_qqq_2025 and item "
                          "file ldw_cog_2025)"}

    def __init__(self):
        self.con = connect(read_only=True)
        self.coverage = self._coverage()
        self.present = self._present_codes()
        self.economy_names = self._economy_names()
        self.facts = self._facts_block()
        self.regions_block = regions.prompt_block(self.present)
        self.no_school_link = self._no_school_link()
        self._fired: list[str] = []

    def _no_school_link(self) -> dict[str, set[str]]:
        """Economies whose student file carries no school identifier (five in
        2025): their students cannot be joined to the school questionnaire,
        so school-variable analyses have nothing for them — a fact stated,
        never an empty result or a "did not take part"."""
        out: dict[str, set[str]] = {}
        for cycle in self.coverage:
            try:
                rows = self.con.sql(f"SELECT CNT FROM stu_qqq_{cycle} GROUP BY CNT "
                                    "HAVING count(CNTSCHID) = 0").fetchall()
            except Exception:  # noqa: BLE001
                continue
            if rows:
                out[cycle] = {r[0] for r in rows}
        return out

    # Every intercept and post-hoc hook that changes an answer records its
    # name here; the list rides on the event so the admin dashboard shows
    # which guards fire, how often, and with what feedback. A guard that
    # never fires can be retired; one that fires with thumbs-down is a hijack.
    def _fire(self, name: str) -> None:
        fired = self.__dict__.setdefault("_fired", [])
        if name not in fired:
            fired.append(name)

    def _economy_names(self) -> dict[str, str]:
        """CNT code -> economy name from the catalog's own value labels."""
        names: dict[str, str] = dict(regions.NAME_FALLBACK)
        for cycle in ("2025", "2022", "2018"):
            desc = catalog.describe("CNT", cycle=cycle)
            desc = desc[desc.table_name.str.startswith("stu_qqq")]
            if not desc.empty and desc.iloc[0].value_labels:
                for k, v in json.loads(desc.iloc[0].value_labels).items():
                    names.setdefault(k, v)
        return names

    def _economies_block(self) -> str:
        """The exact participant list per cycle, from the data — so the model
        can never claim an economy is absent (or present) from memory."""
        every = set().union(*self.present.values()) if self.present else set()
        legend = "; ".join(f"{c} = {self.economy_names.get(c, c)}" for c in sorted(every))
        lines = [f"ECONOMY CODES: {legend}"]
        for cycle, codes in self.present.items():
            lines.append(f"- In PISA {cycle} ({len(codes)}): {' '.join(sorted(codes))}")
        if "2025" in self.present:
            earlier = set().union(*(v for c, v in self.present.items() if c != "2025"))
            new = sorted(self.present["2025"] - earlier)
            gone = sorted(earlier - self.present["2025"])
            lines.append("- First participated in 2025: " + ", ".join(
                f"{c} ({self.economy_names.get(c, c)})" for c in new))
            lines.append("- In earlier cycles but not in 2025: " + ", ".join(
                f"{c} ({self.economy_names.get(c, c)})" for c in gone))
        return "\n".join(lines)

    def _present_codes(self) -> dict[str, set[str]]:
        out = {}
        for cycle in self.coverage:
            out[cycle] = {r[0] for r in self.con.sql(
                f"SELECT DISTINCT CNT FROM stu_qqq_{cycle}").fetchall()}
        return out

    def _oecd_codes(self, cycle: str) -> set[str]:
        return {r[0] for r in self.con.sql(
            f"SELECT DISTINCT CNT FROM stu_qqq_{cycle} WHERE OECD = 1").fetchall()}

    def _region_filter(self, plan: dict, cycles: list[str]) -> dict[str, dict]:
        """Deterministic expansion of plan['regions'] per cycle: the region's
        members present in that cycle's table (and those absent from it)."""
        names = [str(n) for n in (plan.get("regions") or []) if str(n).strip()]
        if not names:
            return {}
        oecd = [n for n in names if n.strip().lower() == "oecd"]
        other = [n for n in names if n not in oecd]
        present = {c: self.present.get(c, set()) for c in cycles}
        out = regions.expand(other, present) if other else \
            {c: {"codes": [], "absent": []} for c in cycles}
        for c in cycles:
            if oecd:
                out[c]["codes"] = sorted(set(out[c]["codes"]) | self._oecd_codes(c))
        return out

    def _coverage(self) -> dict[str, dict]:
        out = {}
        for cycle in CYCLES:
            try:
                n, k = self.con.sql(f"SELECT count(*), count(DISTINCT CNT) "
                                    f"FROM stu_qqq_{cycle}").fetchone()
                out[cycle] = {"students": int(n), "economies": int(k)}
            except Exception:  # noqa: BLE001 — a cycle not built locally
                continue
        return out

    def _facts_block(self) -> str:
        lines = [f"- PISA {c}: LOADED — {v['students']:,} students, "
                 f"{v['economies']} economies; innovative domain: {self.INNOVATIVE[c]}"
                 for c, v in self.coverage.items()]
        return (
            "\n\nFACTS ABOUT THIS APP'S DATA (authoritative — they override anything "
            f"you believe from training; today is {time.strftime('%Y-%m-%d')}):\n"
            + "\n".join(lines) +
            "\n- The OECD published the PISA 2025 results on 8 September 2026 and "
            "the full 2025 public-use database is in this app NOW. Never say 2025 "
            "data are upcoming, unavailable, or not yet released.\n"
            "- AI: no PISA cycle has ASSESSED AI literacy (the OECD announced Media "
            "and AI Literacy as the innovative domain for PISA 2029; not administered, "
            "no data). BUT the PISA 2025 STUDENT QUESTIONNAIRE does hold AI-USE data, "
            "in 84 of 90 economies (Japan included): ST438Q01DA–ST438Q04DA (how often "
            "students use AI chatbots such as ChatGPT for schoolwork, four purposes), "
            "the index AIUSESCH (students' AI use at school, WLE), IC170Q10DA / "
            "IC171Q10DA (AI tools at / outside school, ICT questionnaire, 44 economies), "
            "SC265Q13DA (teachers' professional development on AI) and teacher items "
            "TC045Q23*. A question about AI USE, AI adoption rates or AI in education "
            "is therefore a DATA question (data_question=true, intent=analyze, "
            "search_terms [\"artificial intelligence chatbot\", \"AI use school\"]) — "
            "never say the app has no AI data.\n"
            "- Not loaded: the 2025 Foreign Language Assessment (OECD release "
            "expected 2027).\n"
            "- Questionnaire indices across cycles: the app decides PER INDEX from the "
            "data. Indices the OECD re-standardizes in each cycle (ESCS and most WLE "
            "scales: OECD mean 0 in every cycle) get no cross-cycle change; indices the "
            "OECD kept on the earlier cycle's scale (trend scales such as BULLIED / "
            "BEINGBULLIED, BELONG, ANXMAT, MATHEFF, whose OECD-average is not 0 in a "
            "later cycle) DO get a change with a sampling SE, and every data answer's "
            "notes say which applies. Never state from memory that an index is or is "
            "not comparable — say the app checks it and point to a data question.\n"
            "- Benchmarks: any per-country result can carry an OECD-average row "
            "and/or a regional average row (EU, Latin America, Nordic countries, "
            "…) — ask e.g. \"compare France with the EU average in reading\"; a "
            "benchmark named once is kept for the rest of the conversation. There "
            "is no separate settings panel.\n"
            "- ABOUT THIS APP: PISA Explorer is an independent, open-source web "
            "app (MIT licence, github.com/LuisGaitan/PISA_Explorer) built by Luis "
            "Gaitan at the University of Pennsylvania Graduate School of "
            "Education, supported by the Penn GSE Learning Analytics and "
            "Artificial Intelligence program. It is NOT an OECD product and not "
            "affiliated with the OECD; it analyzes the OECD's public-use "
            "databases with the official methodology (final student weights, "
            "10 plausible values, 80 Fay-BRR replicate weights). Answer "
            "questions about the tool's author, purpose or affiliation only "
            "from this line.\n"
            "When a direct_answer concerns what data exist or when they were "
            "released, use these facts verbatim; if unsure, say the app covers "
            "PISA 2018, 2022 and 2025 and suggest asking a data question.\n\n"
            + self._economies_block() +
            "\nRULES ABOUT ECONOMIES: \"How did <economy> do?\" or any question "
            "naming an economy that appears in the lists above is a DATA question "
            "(data_question=true, intent=analyze) — never answer it from memory and "
            "never say the app lacks data for an economy that is listed. If an "
            "economy is NOT in any list, say exactly: \"<name> is not in the PISA "
            "2018, 2022 or 2025 public-use databases loaded here\" — do not "
            "speculate about other cycles or reasons. China as a whole is not in "
            "the data: only B-S-J-Z (QCI: Beijing, Shanghai, Jiangsu, Zhejiang) in "
            "2018 and 2025. Ukraine appears only as sets of regions (QUR 2022, QUA "
            "2025) and Iraq only as the Kurdistan Region (QKI, 2025). Everyday names "
            "map to these codes: Taiwan = TAP (Chinese Taipei), Turkey = TUR "
            "(Türkiye), South Korea = KOR, UAE = ARE, UK = GBR, Vietnam = VNM, "
            "Macau = MAC, Palestine = PSE, Czech Republic = CZE, Slovakia = SVK."
        )

    # Deterministic answer for "do you have the 2025 data?" — the one factual
    # question the model got wrong in production; the app answers it itself.
    COVERAGE_WORDS = re.compile(
        r"\b(available|access|have|has|include|includes|contain|loaded|released|"
        r"release|yet|cover|covers|coverage|which (years|cycles)|what (years|cycles))\b",
        re.IGNORECASE)

    WHY_WORDS = re.compile(
        r"\b(why|factor|factors|cause|causes|caused|reason|reasons|explain|explains|"
        r"behind (the|this|these|that|those|its|their|such)|driver|drivers|due to|because|attributable|impact of|effect of|"
        r"policy|policies|works best|should (countries|schools|governments)|intervention)\b",
        re.IGNORECASE)
    WHY_NOTE = ("PISA is a repeated cross-sectional survey of different students "
                "each cycle: it can show what changed, not what caused the change. "
                "Within a cycle the app can estimate associations between scores "
                "and student, family or school variables (correlation, regression, "
                "quartile gaps) — ask, for example, “regress science on ESCS and "
                "sense of belonging in Chile in 2025”.")

    PARTICIPATION_WORDS = re.compile(
        r"\b(participat\w*|take part|took part|taken part|"
        r"(have|has|any|got) data (on|for|about)|(is|are|was|were) \w+( \w+)? "
        r"(included|covered|in the (data|database|dataset)|part of pisa|in pisa))\b",
        re.IGNORECASE)
    # A question that asks for a statistic is never a participation question,
    # even when it says "in PISA 2025" ("compare Egypt and Israel in PISA 2025").
    ANALYSIS_WORDS = re.compile(
        r"\b(compare|comparison|mean|average|score|scores|rank|ranked|ranking|"
        r"gap|trend|difference|correlat\w*|percent\w*|share|proportion|top|"
        r"highest|lowest|best|worst|how (did|has|have|many|much)|regress\w*)\b",
        re.IGNORECASE)

    def _is_participation_question(self, question: str) -> bool:
        return bool(self.PARTICIPATION_WORDS.search(question)) and \
            not self.ANALYSIS_WORDS.search(question)

    # "Why is the link error excluded?" / "the OECD says this change is not
    # significant" — a methods question the model must not improvise on.
    LINK_WORDS = re.compile(
        r"link(ing)? error|equating error|scale linking|"
        r"oecd('s)? (analysis|report|reports|publication|table|tables|volume)"
        r".{0,100}\b(non-?significant|not significant|significan)|"
        r"\b(non-?significant|not significant|significan).{0,100}"
        r"oecd('s)? (analysis|report|reports|publication|table|tables|volume)",
        re.IGNORECASE | re.DOTALL)

    # "share of students at Level 6 ... 2022 vs 2025, with the link error":
    # a data request that mentions the link error, not a question about it
    LINK_DATA_REQUEST = re.compile(
        r"\b(with|including|include|add|adding|plus|using|apply|applying)\s+(the\s+)?"
        r"(published\s+)?link(ing)? error", re.IGNORECASE)

    def _link_error_answer(self, last_mode: str | None = None) -> str:
        applied = {
            "mean": "The last trend you asked for was a change in a mean score, so its change "
                    "SE includes the published link error.",
            "percentile-approx": "The last trend you asked for was a change in a percentile; "
                                 "the mean score's link error was added as an approximation "
                                 "(the OECD publishes percentile-specific link errors only in "
                                 "its technical annexes).",
            "share": "The last trend you asked for was a change in a proficiency-level share, so "
                     "a share-specific link error was derived per economy (every plausible "
                     "value shifted by ± the score link error; half the difference in the share) "
                     "and included in the change SE.",
            "cancels": "The last trend you asked for was a change in a within-cycle difference "
                       "(a gap, a quartile gap, a percentile spread, a slope or a correlation): "
                       "the linking shifts both groups alike, so no link error is added — the "
                       "OECD's own convention (PISA 2022 Results Volume I, Annex A7).",
            "none": "The last trend you asked for was a questionnaire index or a response-code "
                    "share, for which the OECD publishes no link error; its change SE is "
                    "sampling error only.",
        }.get(last_mode or "", "")
        status = ("This version applies them the way the OECD does: a change in a MEAN score "
                  "(mathematics, reading, science) gets the published link error; a change in a "
                  "percentile gets the mean's link error as an approximation; a change in the "
                  "share of students at or above (or below) a proficiency level gets a "
                  "share-specific link error derived per economy from the score link error; "
                  "a change in a within-cycle difference (gender gap, ESCS quartile gap, "
                  "P90-P10 spread, regression slope, correlation) gets none, because the linking "
                  "shifts both groups alike and cancels; and changes in questionnaire indices or "
                  "response-code shares get none, because the OECD publishes none for them. "
                  + applied
                  if link_errors.loaded() else
                  "This version does not yet apply them: the constants live in a "
                  "published table, not in the microdata, and have not been "
                  "loaded, so the change SEs shown here are slightly understated.")
        return (
            "Within a cycle, every estimate and standard error in this app "
            "reproduces the OECD's figures (final student weights, 10 plausible "
            "values, 80 Fay-BRR replicates). Across cycles the OECD adds one more "
            "term: the link error, the uncertainty from re-linking each cycle's "
            "scale to the previous one. It is a published constant per pair of "
            "cycles and domain (PISA 2025 Results Volume I, Annex A5), and the OECD "
            "computes SE(change) = sqrt(SE_first² + SE_last² + link_error²). "
            + status +
            " The practical effect: a change whose estimate is only just beyond "
            "1.96 SE here (for example Brazil's +5.5 points in science, SE 2.7, "
            "2022→2025) can be reported as not statistically significant by the "
            "OECD once the link error is added, while larger changes agree. "
            "Point estimates are unaffected. Every trend answer's provenance "
            "card states whether the link error is included.")

    # "Coverage rate" is Coverage Index 3 (the share of 15-year-olds the
    # sample represents) — a published table, not a variable in the files.
    COVERAGE_RATE_WORDS = re.compile(
        r"coverage (rate|rates|index|indices|of (the )?(15|fifteen)[- ]year[- ]olds|"
        r"of (the )?(target )?population)|coverage index|\bci ?3\b|population coverage|"
        r"exclusion rate|exclusion rates",
        re.IGNORECASE)

    # "Are El Salvador's 2025 results comparable with Germany's 2022?" — a
    # methods question with one correct answer (the scales are linked; the
    # link error belongs in the SE), which the planner answered from memory.
    COMPARABLE_WORDS = re.compile(r"\bcomparab(le|les|ility|ilidad)\b|\bcompatible\b", re.IGNORECASE)
    ITEM_ASK_WORDS = re.compile(r"\b(questions?|items?) (about|on|regarding)\b", re.IGNORECASE)

    # ---------- sampling strata: regions, school types and networks inside an economy ----------

    STRATUM_STOP = {"stratum", "general", "public", "private", "urban", "rural", "school",
                    "schools", "region", "large", "small", "north", "south", "east", "west",
                    "central", "state", "other", "lower", "upper", "secondary", "vocational",
                    "students", "student", "country", "countries", "which", "position",
                    "compare", "science", "reading", "mathematics"}
    # one-word stratum label segments that are descriptions, not place or
    # network names — they never identify an entity on their own
    STRATUM_GENERIC = {
        "private", "urban", "public", "rural", "mixed", "other", "south", "north", "general",
        "vocational", "government", "independent", "international", "basic", "suburban",
        "regular", "academic", "city", "secondary", "west", "female", "male", "large", "town",
        "small", "grammar", "catholic", "gymnasium", "east", "central", "middle", "maintained",
        "capital", "subsidized", "village", "center", "centre", "community", "primary", "medium",
        "national", "technical", "liceo", "indian", "russian", "romanian", "lithuanian", "german",
        "french", "arabic", "english", "spanish", "italian", "dutch", "flemish", "hungarian",
        "municipal", "federal", "state", "oficial", "official", "religious", "islamic", "boys",
        "girls", "coed", "metropolitan", "province", "district", "island", "coast", "interior",
        "lower", "upper", "combined", "comprehensive", "selective", "charter", "magnet",
        "modern", "special", "elite", "ordinary", "standard", "advanced", "science", "arts",
        "language", "bilingual", "immersion", "remote", "regional", "very", "more", "less",
        "total", "rest", "others", "year", "grade", "level", "type", "size", "area", "zone",
        "sector", "system", "cluster", "unit", "group", "mainland", "north-east", "south-west"}

    def _strata_index(self) -> list[tuple[str, str, str, str]]:
        """(cycle, code, label, lower-case label) for every stratum label in
        the student files — cached; ~4,000 entries."""
        cache = self.__dict__.get("_strata_cache")
        if cache is not None:
            return cache
        out = []
        for cycle in CYCLES:
            desc = catalog.describe("STRATUM", cycle=cycle)
            desc = desc[desc.table_name.str.startswith("stu_qqq")]
            if desc.empty or not desc.iloc[0].value_labels:
                continue
            for code, label in json.loads(desc.iloc[0].value_labels).items():
                out.append((cycle, str(code), str(label), str(label).lower()))
        self._strata_cache = out
        return out

    # Adjudicated sub-national entities that live inside another economy's
    # file: the alias selects their strata (codes differ per cycle).
    # alias -> (code prefixes, label regex): a stratum belongs to the entity
    # when its code starts with a prefix OR its label matches
    STRATA_ALIASES = {
        "scotland": (("QSC",), re.compile(r"^QSC\b|scotland", re.I)),
        "england": ((), re.compile(r"\bengland\b", re.I)),
        "wales": ((), re.compile(r"\bwales\b", re.I)),
        "northern ireland": ((), re.compile(r"northern ireland", re.I)),
        # Baku is also the 2018/2022 economy QAZ; in 2025 it is a stratum of AZE
        "baku": ((), re.compile(r"^Baku$", re.I)),
        # Nazarbayev Intellectual Schools, as officials abbreviate them
        "nis": ((), re.compile(r"^(?!.*non-intellectual).*\bintellectual\b", re.I)),
        "nis schools": ((), re.compile(r"^(?!.*non-intellectual).*\bintellectual\b", re.I)),
    }
    # aliases whose entity lives in another economy's file (economy code)
    STRATA_ALIAS_ECONOMY = {"scotland": "GBR", "england": "GBR", "wales": "GBR",
                            "northern ireland": "GBR", "baku": "AZE", "nis": "KAZ", "nis schools": "KAZ"}

    def _strata_entities(self, question: str) -> list[dict]:
        """The sub-economy entities a question names, each with its stratum
        codes per cycle: [{"name", "economy", "codes": {cycle: [code, ...]},
        "labels": {code: label}}]. Two routes: (a) a fixed alias for an
        adjudicated sub-national entity (Scotland = the QSC strata of GBR);
        (b) a label PHRASE of two or more consecutive words that appears in
        the question ("Intellectual schools" for "Nazarbayev Intellectual
        schools") - a single rare word is not enough, it hijacked ordinary
        questions ("summary", "position")."""
        index = self._strata_index()
        if not index:
            return []
        low_q = " " + re.sub(r"[^a-z0-9 ]", " ", (question or "").lower()) + " "
        low_q = re.sub(r"\s+", " ", low_q)
        entities = []
        # bare stratum codes in the question ("stratum KAZ21") name the entity directly
        codes_named = [c for c in re.findall(r"\b[A-Z]{3}\d{2,4}\b", question or "")
                       if any(code == c for _, code, _, _ in index)]
        if codes_named:
            ents = self._entities_from_codes(["STRATUM IN (" + ", ".join(f"'{c}'" for c in codes_named) + ")"])
            if ents:
                return ents
        for alias, (prefixes, rx) in self.STRATA_ALIASES.items():
            if f" {alias} " not in low_q:
                continue
            economy = self.STRATA_ALIAS_ECONOMY.get(alias, "GBR")
            ent = {"name": alias.title(), "economy": economy, "codes": {}, "labels": {}}
            for c, code, label, low in index:
                if (code.startswith(prefixes) or rx.search(label)) and \
                        (economy == "GBR" or code.startswith(economy)):
                    ent["codes"].setdefault(c, []).append(code)
                    ent["labels"][code] = label
            if ent["codes"]:
                if economy != "GBR":
                    # name the entity as the file does ("Intellectual schools")
                    last_label = ent["labels"][ent["codes"][max(ent["codes"])][0]]
                    ent["name"] = re.sub(r"^\w{3} - stratum \d+:\s*", "", last_label).split("/")[0].strip() or ent["name"]
                if not any(e["economy"] == economy and e["codes"] == ent["codes"] for e in entities):
                    entities.append(ent)
        if entities:
            return entities
        economy_words = {w for name in getattr(self, "economy_names", {}).values()
                         for w in re.findall(r"[a-z]{4,}", str(name).lower())}
        words = {w for w in re.findall(r"[a-z]{4,}", (question or "").lower())
                 if w not in self.STRATUM_STOP and w not in catalog.STOPWORDS
                 and w not in economy_words and w not in self.STRATUM_GENERIC}
        hits = []
        for w in words:
            matches = [(c, code, label) for c, code, label, low in index
                       if re.search(rf"(?<![a-z-]){re.escape(w)}", low)
                       and not low.startswith("undisclosed")]
            if not (0 < len(matches) <= 60):
                continue
            for c, code, label in matches:
                lab = re.sub(r"[^a-z0-9 ]", " ", label.lower())
                lab_words = [t for t in lab.split() if t]
                phrase_ok = False
                for i, t in enumerate(lab_words):
                    if t.startswith(w):
                        for j in (i - 1, i + 1):
                            if 0 <= j < len(lab_words):
                                pair = " ".join(lab_words[min(i, j):max(i, j) + 1])
                                if f" {pair} " in low_q:
                                    phrase_ok = True
                # (c) a one-word label SEGMENT that is a proper name ("Baku";
                # "Dubai / Private / UK"; "Tashkent/Urban"): the word alone
                # names the entity — common words ("Private", "City") never do
                if not phrase_ok and len(matches) <= 60:
                    body = re.sub(r"^\w{3} - stratum \d+:\s*", "", label)
                    for seg in re.split(r"[/,:;()–]|\s-\s", body):
                        seg = seg.strip()
                        if seg and " " not in seg and seg[0].isupper() and seg.lower() == w:
                            phrase_ok = True
                if phrase_ok:
                    hits.append((c, code, label))
        if not hits:
            return []
        # one entity per economy prefix
        by_econ: dict[str, dict] = {}
        for c, code, label in dict.fromkeys(hits):
            econ = "GBR" if code[:3] in ("QSC", "QUK") else code[:3]
            ent = by_econ.setdefault(econ, {"name": label.split("/")[0].split(":")[-1].strip() or label,
                                            "economy": econ, "codes": {}, "labels": {}})
            if code not in ent["codes"].get(c, []):
                ent["codes"].setdefault(c, []).append(code)
                ent["labels"][code] = label
        return list(by_econ.values())

    def _strata_hits(self, question: str, limit: int = 12) -> list[tuple[str, str, str]]:
        out = []
        for ent in self._strata_entities(question):
            for c, codes in sorted(ent["codes"].items()):
                out += [(c, code, ent["labels"][code]) for code in codes]
        return out[:limit * 3]

    def _apply_strata(self, plan: dict, question: str) -> None:
        """Deterministic stratum plans. One entity ("Nazarbayev Intellectual
        schools"): the plan is filtered to its strata, per cycle, and never
        grouped by STRATUM. Two aliases ("Scotland compared with England"):
        a gap between the two sets of strata. The planner was asked to do
        this and produced a table of every stratum once and a CASE in `by`
        the next time; here it is one rule."""
        if plan.get("action") == "clarify" or plan.get("template") == "raw_sql":
            return
        ents = self._strata_entities(question)
        wheres = [str(plan.get("where") or "")] + [str(ov.get("where") or "") for ov in
                                                   (plan.get("cycle_overrides") or {}).values()
                                                   if isinstance(ov, dict)]
        planner_filter = any(re.search(r"STRATUM", w, re.I) for w in wheres)
        if not ents and planner_filter:
            # "excluding stratum KAZ21": no label named, but the codes are in
            # the planner's filter — build the entity from them
            ents = self._entities_from_codes(wheres)
        if not ents:
            return
        planner_excluded = False
        if planner_filter:
            # The planner wrote the stratum filter itself: strip it and let the
            # entity logic below add the labelled row NEXT TO the economy's own
            # row (a plan filtered to the strata came back labelled as the
            # whole economy); a NOT IN filter means "excluding".
            clause = self.STRATUM_CLAUSE
            planner_excluded = any(re.search(r"STRATUM\s*(NOT\s+IN|<>|!=)", w, re.I) for w in wheres)
            plan["where"] = clause.sub("", str(plan.get("where") or "")).strip() or None
            plan["where"] = re.sub(r"^\s*(AND|OR)\s+", "", plan["where"] or "", flags=re.I).strip() or None
            for ov in (plan.get("cycle_overrides") or {}).values():
                if isinstance(ov, dict) and ov.get("where"):
                    ov["where"] = clause.sub("", str(ov["where"])).strip() or None
                    ov["where"] = re.sub(r"^\s*(AND|OR)\s+", "", ov["where"] or "", flags=re.I).strip() or None
                    if not ov["where"]:
                        ov.pop("where", None)
            self._fire("hook:planner_stratum_filter_stripped")
        cycles = [str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]]
        by = [b for b in (plan.get("by") or ["CNT"])
              if isinstance(b, str) and b.upper() != "STRATUM" and not b.upper().startswith("CASE")]
        if "CNT" not in by:
            by = ["CNT"] + by
        plan["by"] = by
        economy = ents[0]["economy"]
        base_where = str(plan.get("where") or "")
        if f"'{economy}'" not in base_where:
            others = [c for c in dict.fromkeys(re.findall(r"'([A-Z]{3})'", base_where)) if c != economy]
            if others and re.fullmatch(r"\s*CNT\s*(=|IN)\s*\(?\s*(?:'[A-Z]{3}'\s*,?\s*)+\)?\s*", base_where, re.I):
                # "Baku 2022 (QAZ) and the Baku stratum 2025 (AZE)": keep both
                base_where = "CNT IN (" + ", ".join(f"'{c}'" for c in others + [economy]) + ")"
            else:
                base_where = f"CNT = '{economy}'"
        overrides = plan.get("cycle_overrides") if isinstance(plan.get("cycle_overrides"), dict) else {}
        kept = []
        if len(ents) >= 2 and all(e["economy"] == economy for e in ents[:2]):
            a, b = ents[0], ents[1]
            measure = plan.get("measure") or ((plan.get("measures") or [None])[0]) or "PV{pv}SCIE"
            plan.update({"template": "gap", "measure": measure, "measures": None, "minuend": 1,
                         "subtrahend": 0, "group_label": f"{a['name']} minus {b['name']}",
                         "where": base_where, "instrument": plan.get("instrument") or "stu_qqq"})
            for c in cycles:
                if a["codes"].get(c) and b["codes"].get(c):
                    ca = ", ".join(f"'{x}'" for x in a["codes"][c])
                    cb = ", ".join(f"'{x}'" for x in b["codes"][c])
                    overrides.setdefault(c, {})["group_col"] =                         f"CASE WHEN STRATUM IN ({ca}) THEN 1 WHEN STRATUM IN ({cb}) THEN 0 END"
                    kept.append(c)
            plan["group_col"] = overrides[kept[0]]["group_col"] if kept else "STRATUM"
        elif self.STRATA_REST_WORDS.search(question or ""):
            # "Tashkent city versus the rest of the country": a gap between
            # the entity's strata and every other stratum of the economy
            ent = ents[0]
            measure = plan.get("measure") or ((plan.get("measures") or [None])[0]) or "PV{pv}SCIE"
            plan.update({"template": "gap", "measure": measure, "measures": None, "minuend": 1,
                         "subtrahend": 0, "group_label": f"{ent['name']} minus the rest of {economy}",
                         "where": base_where, "instrument": plan.get("instrument") or "stu_qqq"})
            for c in cycles:
                codes = ent["codes"].get(c)
                if codes:
                    lst = ", ".join(f"'{x}'" for x in codes)
                    overrides.setdefault(c, {})["group_col"] = \
                        f"CASE WHEN STRATUM IN ({lst}) THEN 1 WHEN STRATUM IS NOT NULL THEN 0 END"
                    kept.append(c)
            plan["group_col"] = overrides[kept[0]]["group_col"] if kept else "STRATUM"
        else:
            # One entity: the plan stays the economy's (the OECD-reported
            # figure) and the app adds a labelled row for the entity — or, for
            # "excluding <entity>", for the economy without it. A plan filtered
            # to the strata alone came back labelled as the whole economy.
            ent = ents[0]
            plan["where"] = base_where
            exclude = bool(self.STRATA_EXCLUDE_WORDS.search(question or "")) or planner_excluded
            codes_by_cycle = {c: ent["codes"][c] for c in cycles if ent["codes"].get(c)}
            kept = list(cycles) if codes_by_cycle else []   # national rows in every cycle
            if kept:
                plan["_strata_rows"] = {
                    "name": ent["name"], "economy": economy, "mode": "exclude" if exclude else "include",
                    "label": (f"{economy} excl. {ent['name']}" if exclude else f"{economy}/{ent['name']}")[:48],
                    "codes": codes_by_cycle,
                    "labels": {code: ent["labels"].get(code, code) for cs in codes_by_cycle.values() for code in cs}}
        if not kept:
            return
        if len(kept) < len(cycles):
            plan["_strata_cycles_dropped"] = [c for c in cycles if c not in kept]
        plan["cycles"] = kept
        plan["cycle_overrides"] = overrides
        self._fire("hook:strata_plan")

    # "Level 1 or below" / "Level 1 or lower" / "nivel 1 o menos" is the
    # OECD's low-performer group: every student below Level 2 (Levels 1a, 1b,
    # 1c and below 1c). The planner read it as "below Level 1a".
    LEVEL1_OR_BELOW = re.compile(
        r"\blevel\s*1\s*(or|and)\s*(below|lower|less|under|beneath)\b|"
        r"\b(at|in)\s+level\s*1\s+or\s+(below|lower)\b|\bnivel\s*1\s*o\s*(menos|inferior|por debajo)\b",
        re.IGNORECASE)

    def _fix_level_phrases(self, plan: dict, question: str) -> None:
        if plan.get("action") == "clarify" or not self.LEVEL1_OR_BELOW.search(question or ""):
            return
        changed = False
        for key in ("measure", "x", "y"):
            expr = plan.get(key)
            if not isinstance(expr, str):
                continue
            m = re.search(r"PV\{pv\}(MATH|READ|SCIE)\s*<\s*(\d+(?:\.\d+)?)", expr)
            if not m:
                continue
            domain, cut = m.group(1), float(m.group(2))
            bounds = dict(link_errors.LEVELS[domain])
            level2 = bounds.get("2")
            if level2 and abs(cut - level2) > 0.01 and abs(cut - bounds.get("1a", -1)) < 0.01:
                plan[key] = expr.replace(m.group(2), f"{level2}")
                changed = True
        measures = plan.get("measures")
        if isinstance(measures, list):
            fixed = []
            for item in measures:
                expr = item.get("expr") if isinstance(item, dict) else item
                if isinstance(expr, str):
                    m = re.search(r"PV\{pv\}(MATH|READ|SCIE)\s*<\s*(\d+(?:\.\d+)?)", expr)
                    if m:
                        bounds = dict(link_errors.LEVELS[m.group(1)])
                        if abs(float(m.group(2)) - bounds.get("1a", -1)) < 0.01:
                            new = expr.replace(m.group(2), f"{bounds['2']}")
                            item = {**item, "expr": new} if isinstance(item, dict) else new
                            changed = True
                fixed.append(item)
            plan["measures"] = fixed
        if changed:
            plan["_level1_or_below"] = True
            self._fire("hook:level1_or_below")

    # a STRATUM filter the planner wrote, with its leading AND and parentheses
    STRATUM_CLAUSE = re.compile(
        r"(?:\s*\bAND\s+)?\(?\s*STRATUM\s*(?:NOT\s+IN|IN)\s*\([^)]*\)\s*\)?|"
        r"(?:\s*\bAND\s+)?\(?\s*STRATUM\s*(?:=|<>|!=)\s*'[^']*'\s*\)?", re.I)

    def _entities_from_codes(self, wheres: list[str]) -> list[dict]:
        """Entities built from stratum codes in a planner's filter."""
        codes = [c for w in wheres for m in self.STRATUM_CLAUSE.finditer(w)
                 for c in re.findall(r"'([^']+)'", m.group(0))]
        if not codes:
            return []
        index = {code: (c, label) for c, code, label, _ in self._strata_index()}
        by_econ: dict[str, dict] = {}
        for code in dict.fromkeys(codes):
            if code not in index:
                continue
            cycle, label = index[code]
            econ = "GBR" if code[:3] in ("QSC", "QUK") else code[:3]
            name = re.sub(r"^\w{3} - stratum \d+:\s*", "", label).split("/")[0].strip() or code
            ent = by_econ.setdefault(econ, {"name": name[:40], "economy": econ, "codes": {}, "labels": {}})
            ent["codes"].setdefault(cycle, []).append(code)
            ent["labels"][code] = label
        return list(by_econ.values())

    STRATA_EXCLUDE_WORDS = re.compile(
        r"\b(excluding|exclude|without|other than|except|apart from|leaving out|minus the|net of)\b",
        re.IGNORECASE)
    STRATA_REST_WORDS = re.compile(
        r"\b(rest of (the )?(country|economy|nation)|remaining (schools|regions|strata|country)|"
        r"(versus|vs\.?|against|compared (to|with)) (the )?(rest|other (schools|regions|strata)|"
        r"non-\w+ schools)|everyone else|all other (schools|regions))\b", re.IGNORECASE)

    def _with_stratum_rows(self, res: pd.DataFrame, template: str, mplan: dict, tbl: str,
                           by: tuple, cwhere: str | None, cycle: str, plan: dict) -> pd.DataFrame:
        """Rows for a named sampling stratum next to the economy's own row,
        labelled so they can never be read as the economy: "IDN/DKI Jakarta"
        or "KAZ excl. Intellectual schools". A filter the planner wrote itself
        on STRATUM relabels the row the same way."""
        if res is None or "CNT" not in res.columns or "CNT" not in by:
            return res
        strata = plan.get("_strata_rows")
        if strata and cycle in strata["codes"]:
            lst = ", ".join(f"'{x}'" for x in strata["codes"][cycle])
            op = "NOT IN" if strata["mode"] == "exclude" else "IN"
            base = f"({cwhere}) AND " if cwhere else ""
            swhere = f"{base}STRATUM {op} ({lst})"
            if strata["mode"] == "exclude":
                swhere += " AND STRATUM IS NOT NULL"
            sres = self._run_template(template, mplan, tbl, by, swhere)
            sres = self._suppress_small_cells(sres, cycle, plan)
            sres, _ = self._drop_null_groups(sres, by)
            sres = sres[sres["CNT"].astype(str) == strata["economy"]].copy()
            sres["CNT"] = strata["label"]
            self._fire("hook:stratum_rows")
            return pd.concat([res, sres], ignore_index=True)
        if cwhere and re.search(r"\bSTRATUM\s*(=|\bIN\b|\bNOT\s+IN\b)", cwhere, re.I):
            m = re.search(r"\bSTRATUM\s*(=|IN|NOT\s+IN)\s*\(?\s*((?:'[^']*'\s*,?\s*)+)\)?", cwhere, re.I)
            if m:
                codes = re.findall(r"'([^']*)'", m.group(2))
                labels = {code: label for c, code, label, _ in self._strata_index() if c == cycle}
                names = [labels.get(code, code).split("/")[0].split(":")[-1].strip() for code in codes[:2]]
                tag = ("excl. " if "NOT" in m.group(1).upper() else "") + ", ".join(dict.fromkeys(names))
                if len(codes) > 2:
                    tag += f" +{len(codes) - 2}"
                res = res.copy()
                res["CNT"] = res["CNT"].astype(str).map(lambda c: f"{c}/{tag}"[:48] if not c.endswith(" avg") else c)
                plan.setdefault("_strata_relabelled", {})[cycle] = {"codes": codes, "tag": tag}
        return res

    @staticmethod
    def _strata_block(hits) -> str:
        if not hits:
            return ""
        lines = ["SAMPLING STRATA matching the question (a stratum is a region, school "
                 "type or school network inside an economy). The app applies the stratum "
                 "filter itself: plan the STATISTIC as if for the whole economy (by "
                 "[\"CNT\"], where = the economy; never group by STRATUM and never put a "
                 "CASE in `by`) — the app then ADDS the stratum as its own labelled row next "
                 "to the economy's row, so 'Indonesia and Jakarta' or 'Kazakhstan excluding "
                 "the NIS schools' IS one table: never clarify that both cannot be shown "
                 "together. For 'A compared with B' between two such entities plan a gap "
                 "and the app fills the group_col. Strata for reference:"]
        by_cycle: dict[str, list] = {}
        for cycle, code, label in hits:
            by_cycle.setdefault(cycle, []).append((code, label))
        for cycle in sorted(by_cycle):
            items = by_cycle[cycle]
            econ = "GBR" if items[0][0][:3] in ("QSC", "QUK") else items[0][0][:3]
            if len(items) <= 4:
                lines.append(f"- PISA {cycle} (economy {econ}): " + "; ".join(
                    f"STRATUM = '{code}' = {label}" for code, label in items))
            else:
                lines.append(f"- PISA {cycle} (economy {econ}): STRATUM IN (" +
                             ", ".join(f"'{code}'" for code, _ in items) + ") = " +
                             f"{items[0][1]} … ({len(items)} strata)")
        return "\n".join(lines) + "\n"

    WORLD_AVERAGE_WORDS = re.compile(
        r"\b(world|worldwide|global|international)\b.{0,15}\b(average|mean)\b|"
        r"\b(average|mean)\b.{0,25}\b(world|worldwide|globe|global|all (participating |the )?"
        r"(countries|economies|participants|nations))\b|"
        r"\ball[- ](participating |the )?(countries|economies|participants|nations)\b.{0,15}\b(average|mean)\b|"
        r"promedio (mundial|global|de todos)|moyenne (mondiale|globale)", re.IGNORECASE)

    def _comparability_answer(self, codes: list[str] | None = None, text: str = "") -> str:
        mode_note = ""
        if codes:
            years = sorted(set(self.YEAR_RE.findall(text or ""))) or CYCLES
            changed = self._mode_changes(codes, years)
            if changed:
                who = "; ".join(f"{self.economy_names.get(c, c)} ({c}) was tested on {a} in {y1} and on "
                                f"{b} in {y2}" for c, y1, a, y2, b in changed)
                mode_note = (f" One more caveat applies here: {who}. The OECD still reports the linked "
                             "trend with the link error, but the effect of the change of mode is not "
                             "quantified in the public files, and the PISA 2025 Technical Report (Data "
                             "Adjudication) recommends caution in interpreting such trends.")
            else:
                same = [f"{self.economy_names.get(c, c)} ({c})" for c in codes[:4]
                        if any(c in self._mode_index().get(y, {}) for y in years)]
                if same:
                    mode_note = (f" The test mode (paper or computer, from ADMINMODE) did not change "
                                 f"for {', '.join(same)} between the cycles named.")
        example = ("“compare science scores for El Salvador, Sweden and Germany in 2022 and 2025”"
                   if not codes else
                   f"“compare reading scores for {', '.join(self.economy_names.get(c, c) for c in codes[:3])} "
                   f"in {' and '.join(sorted(set(self.YEAR_RE.findall(text or ''))) or ['2018', '2025'])}”")
        return (
            "Yes, with one caveat. PISA scores in mathematics, reading and science are "
            "reported on scales that the OECD links from cycle to cycle, so a 2025 "
            "score can be compared with a 2018 or 2022 score, for the same economy or "
            "for different ones. The caveat is uncertainty: the linking itself adds a "
            "published link error to the standard error of any difference across "
            "cycles, and this app includes it (for mean scores and, derived per economy, "
            "for proficiency-level shares) so that significance matches the OECD's "
            "reports. Questionnaire indices that the OECD re-standardizes in each cycle "
            "(ESCS and most WLE scales) are NOT comparable across cycles; indices the OECD "
            "kept on the earlier cycle's scale, and shares of a response code, are."
            + mode_note +
            " To see the numbers, ask for both economies over the cycles you need — for "
            f"example {example} — and the table lists each economy per cycle with standard errors.")

    def _coverage_rate_answer(self) -> str:
        return (
            "Coverage rates (Coverage Index 3: the share of an economy's 15-year-old "
            "population represented by the PISA sample, after school- and "
            "student-level exclusions) are not variables in the public-use "
            "databases loaded here, so this app cannot compute or compare them. "
            "The OECD publishes them per economy and cycle in PISA 2025 Results "
            "(Volume I), Annex A2, and in the Technical Report's sampling-outcomes "
            "chapter. What the microdata do support: the number of sampled students "
            "and the sum of final student weights (the enrolled 15-year-old "
            "population each sample represents), shown in the provenance card of "
            "every answer. A change in coverage between cycles cannot be separated "
            "from the change in scores with this app.")

    def _participation_answer(self, codes: list[str]) -> str:
        """Which loaded cycles each named economy appears in — from the data."""
        parts = []
        for code in codes:
            cycles = sorted(c for c, present in self.present.items() if code in present)
            name = self.economy_names.get(code, code)
            parts.append(f"{name} ({code}): in PISA " + ", ".join(cycles)
                         + (" only" if len(cycles) == 1 else ""))
        return ("Yes — " + "; ".join(parts) + ". The public-use data for those "
                "cycles are loaded here; ask for a statistic, for example “mean "
                f"science score in {self.economy_names.get(codes[0], codes[0])} in "
                f"{max(c for c, p in self.present.items() if codes[0] in p)}”.")

    def _coverage_answer(self) -> str:
        parts = [f"PISA {c} ({v['economies']} economies, {v['students']:,} students)"
                 for c, v in self.coverage.items()]
        return ("Yes. This app holds the full OECD public-use databases for "
                + ", ".join(parts) + ". The 2025 database was released by the OECD "
                "on 8 September 2026 and is loaded here, including the Learning in "
                "the Digital World domain (2025's innovative domain). Not loaded: the "
                "2025 Foreign Language Assessment, which the OECD releases in 2027. "
                "No PISA cycle has assessed AI literacy yet (announced for PISA 2029), "
                "but the 2025 student questionnaire has AI-use items (AI chatbots for "
                "schoolwork, 84 economies). Ask a data question — for "
                "example “mean science score in Finland in 2025” or “find me data "
                "about well-being in 2025”.")

    # ---------- retrieval ----------

    def _retrieve(self, terms: list[str], per_term: int = 8) -> pd.DataFrame:
        """Top-scoring VARIABLES across the search terms, with every table
        (cycle/instrument) row of each — capped at MAX_CARDS variables."""
        # Term roles: a bare year is not a topic (skipped); a term naming only
        # a respondent type ("teachers") matches hundreds of labels and says
        # nothing about the construct (0.3); the first real concept is the
        # user's own (1.0); the router's broader synonyms that follow
        # ("school climate" for "bullying") must not outrank it (0.5). A
        # variable matching several terms still earns a bonus.
        frames = []
        primary_seen = False
        for i, term in enumerate(terms):
            words = [w for w in re.split(r"\W+", term.lower()) if w]
            if not words or all(re.fullmatch(r"\d{4}", w) for w in words):
                continue
            generic = all(w in catalog.ENTITY_WORDS or w in catalog.STOPWORDS
                          for w in words)
            if generic:
                weight = 0.3
            elif not primary_seen:
                weight, primary_seen = 1.0, True
            else:
                weight = 0.5
            frame = catalog.search(term, limit=per_term)
            frames.append(frame.assign(score=frame.score * weight, _term=i))
        if not frames:
            return pd.DataFrame()
        stacked = pd.concat(frames, ignore_index=True)
        n_terms = stacked.groupby("variable")["_term"].nunique()
        hits = stacked.drop_duplicates(subset=["variable", "table_name"]).drop(columns="_term")
        best = stacked.groupby("variable")["score"].max() * (1 + 0.1 * (n_terms - 1))
        keep = best.sort_values(ascending=False).head(MAX_CARDS).index
        hits = hits[hits.variable.isin(keep)].assign(score=lambda d: d.variable.map(best))
        return hits.sort_values(["score", "variable", "table_name"],
                                ascending=[False, True, True])

    # Fewer than this many economies lacking a variable is noise (a national
    # item dropped by a couple of countries), not an optional questionnaire.
    COVERAGE_NOISE = 8
    COVERAGE_LIST_MAX = 40

    @classmethod
    def _cards(cls, hits: pd.DataFrame, named=()) -> str:
        """One card per variable, listing every table it exists in — so the
        planner sees cross-cycle availability at a glance (and a variable
        absent from a cycle is visibly absent). `named` = economy codes the
        question names: a table where they never collected the variable is
        flagged on the card."""
        if hits is None or hits.empty:
            return "(no extra variables retrieved)"
        named = set(named or ())
        lines = []
        for var, group in hits.groupby("variable", sort=False):
            group = group.sort_values("table_name")
            latest = group.sort_values("cycle").iloc[-1]
            desc = catalog.describe(var, cycle=latest.cycle)
            values = ""
            if not desc.empty and desc.iloc[0].value_labels:
                labels = json.loads(desc.iloc[0].value_labels)
                shown = [f"{k}={v}" for k, v in list(labels.items())[:6]]
                values = f" [values: {', '.join(shown)}]"
            tables = ", ".join(group.table_name)
            lines.append(f"- {var} ({tables}): {latest.label}{values}")
            for tbl in group.table_name:
                line = cls._coverage_line(var, tbl, named)
                if line:
                    lines.append("    " + line)
        return "\n".join(lines)

    @classmethod
    def _coverage_line(cls, var: str, table: str, named: set) -> str:
        cov = catalog.coverage(var, table)
        if not cov or not cov["partial"]:
            return ""
        n, k = cov["n_with_data"], cov["n_economies"]
        named_missing = sorted(named & cov["missing"])
        if named_missing and link_errors.domain_of(var):
            return (f"{table}: the OECD did not release {' '.join(named_missing)}'s "
                    f"{link_errors.DOMAIN_NAMES[link_errors.domain_of(var)]} results "
                    "(took part; no plausible values) — no estimate possible for them")
        if named_missing:
            return (f"{table}: NOT collected for {' '.join(named_missing)} "
                    f"(collected in {n} of {k} economies) — do not use it for them")
        if n <= cls.COVERAGE_LIST_MAX:
            return (f"{table}: collected in only {n} of {k} economies: "
                    f"{' '.join(sorted(cov['with_data']))}")
        if k - n >= cls.COVERAGE_NOISE:
            return (f"{table}: collected in {n} of {k} economies (not in: "
                    f"{' '.join(sorted(cov['missing']))})")
        return ""

    # The two gender codings in the data: the same groups, different codes.
    GENDER_CODES = {"ST004D01T": {"1": "female", "2": "male"},
                    "MALE": {"1": "male", "0": "female"}}

    def _align_gender_direction(self, plan: dict, overrides: dict) -> None:
        """A per-cycle override that swaps ST004D01T for MALE must keep the
        SAME direction (male minus female, or the reverse) — otherwise the
        cross-cycle change adds a sign flip to the real change. The main
        plan's direction wins; a corrected override is recorded for provenance."""
        main_col = str(plan.get("group_col") or "")
        if main_col not in self.GENDER_CODES:
            return
        main = self.GENDER_CODES[main_col]
        want = (main.get(str(plan.get("minuend"))), main.get(str(plan.get("subtrahend"))))
        if None in want or want[0] == want[1]:
            return
        for cycle, ov in overrides.items():
            col = str(ov.get("group_col") or main_col)
            if col not in self.GENDER_CODES:
                continue
            inv = {v: k for k, v in self.GENDER_CODES[col].items()}
            have = (str(ov.get("minuend", plan.get("minuend"))),
                    str(ov.get("subtrahend", plan.get("subtrahend"))))
            if have != (inv[want[0]], inv[want[1]]):
                caster = type(plan.get("minuend")) if isinstance(plan.get("minuend"), int) else str
                ov["group_col"] = col
                ov["minuend"] = caster(inv[want[0]])
                ov["subtrahend"] = caster(inv[want[1]])
                plan.setdefault("_direction_fixed", []).append(
                    f"PISA {cycle}: the gender override pointed the other way "
                    f"({col}: {have[0]} - {have[1]}); the app re-aligned it to "
                    f"the main plan ({want[0]} minus {want[1]}) so the "
                    f"cross-cycle change is comparable.")

    def _missing_estimate_notes(self, table: pd.DataFrame,
                                plan: dict | None = None) -> list[str]:
        plan = plan or {}
        measures = [m for m in (plan.get("measures") or []) if m] or [plan.get("measure")]
        # "the OECD did not release" is only the right reading for a plain
        # mean score; in a gap or quartile table a blank cell usually means an
        # empty group (no private schools sampled, say)
        release_wording = plan.get("template") == "weighted_mean" and \
            all(link_errors.is_mean_score(m) for m in measures)
        measure = measures[0] if release_wording else None
        """Blank estimates explained: an economy that was not in a cycle is
        said so; anything else is a variable not administered / no valid
        responses for that group."""
        estimate_cols = [c for c in table.columns if ESTIMATE_COL.match(c)]
        if not estimate_cols:
            return []
        mask = table[estimate_cols].isna().copy()
        notes = []
        # cells the app blanked itself (reporting minimum) are not "missing"
        suppressed = plan.get("_suppressed_rows") or {}
        if suppressed:
            for col in estimate_cols:
                m = re.fullmatch(r"estimate_(\d{4})", col)
                cyc = m.group(1) if m else (str(table["cycle"].iloc[0]) if "cycle" in table.columns else None)
                for key in suppressed.get(cyc, []):
                    if not isinstance(key, dict):
                        continue
                    cols = [c for c in key if c in table.columns]
                    if not cols:
                        continue
                    hit = pd.Series(True, index=table.index)
                    for c in cols:
                        hit &= table[c].astype(str) == key[c]
                    mask.loc[hit, col] = False
        if "CNT" in table.columns:
            cnt = table["CNT"].astype(str)
            for col in estimate_cols:
                m = re.fullmatch(r"estimate_(\d{4})", col)
                present = self.present.get(m.group(1)) if m else None
                if not present:
                    continue
                absent_rows = mask[col] & ~cnt.isin(present) & ~cnt.str.endswith(" avg")
                if absent_rows.any():
                    cycle = m.group(1)
                    for code in sorted(set(cnt[absent_rows])):
                        kin = [k for k in regions.CODE_SUCCESSION.get(code, ()) if k in present]
                        if kin:
                            notes.append(
                                f"PISA {cycle}: {self._names([code])} appears in that cycle as "
                                f"{self._names(kin)} — a different coverage of the same country. "
                                "The OECD does not compare the two as a trend, so no change is "
                                "computed across them; the levels for each are shown.")
                        else:
                            notes.append(f"PISA {cycle}: {self._names([code])} did not take "
                                         f"part in that cycle — those cells are blank because the "
                                         f"economy was not assessed, not because a variable is missing.")
                    mask.loc[absent_rows, col] = False
        n_missing = int(mask.any(axis=1).sum())
        if n_missing:
            domain = link_errors.domain_of(measure)
            if domain and "CNT" in table.columns:
                rows = table["CNT"].astype(str)[mask.any(axis=1)]
                who = self._names(sorted(c for c in set(rows) if not c.endswith(" avg"))[:12])
                notes.append(
                    f"{n_missing} row(s) have no {link_errors.DOMAIN_NAMES[domain]} "
                    f"estimate ({who}): those economies took part, but the OECD did "
                    f"not release their {link_errors.DOMAIN_NAMES[domain]} results in "
                    "the public database for that cycle (no plausible values — e.g. "
                    "Uzbekistan 2025 in mathematics and reading). They are listed in "
                    "the table but excluded from the chart.")
            else:
                notes.append(
                    f"{n_missing} row(s) have no estimate — the variable was not "
                    "administered (or has no valid responses) for those groups; "
                    "they are listed in the table but excluded from the chart.")
        return notes

    # ---------- standard variables (explorer/standards.py) ----------

    def _with_standard_cards(self, hits: pd.DataFrame, question: str) -> pd.DataFrame:
        """Cards the planner must always see: the standard variable for every
        construct the question names (explorer/standards.py), whether or not
        retrieval surfaced it — retrieval depends on the router's search
        terms, which vary; the standard does not."""
        matches = standards.matching(question or "")
        if not matches:
            return hits
        have = set(hits.variable) if hits is not None and not hits.empty else set()
        extras = []
        for s in matches:
            for code in s.codes():
                if code in have:
                    continue
                desc = catalog.describe(code.replace("{pv}", "1"))
                prefixes = {"stu_sch": ("stu_qqq", "sch_qqq"), "stu_crt": ("stu_qqq", "crt_cog")}.get(
                    s.instrument, (s.instrument,))
                desc = desc[desc.table_name.str.startswith(prefixes)]
                if desc.empty:
                    continue
                have.add(code)
                extras.append(desc.assign(n_value_labels=0, score=1000.0)[
                    ["variable", "table_name", "cycle", "label", "n_value_labels", "score"]])
        if not extras:
            return hits
        self._fire("standard:cards")
        extra = pd.concat(extras, ignore_index=True)
        return pd.concat([extra, hits], ignore_index=True) if hits is not None else extra

    def _apply_standards(self, plan: dict, question: str) -> None:
        """A plan that picked a look-alike of a construct's standard variable
        (PRIVATESCH for public/private, HOMEPOS for socio-economic status) is
        switched to the standard — unless the user named that variable —
        and the switch is stated in the provenance. Only constructs whose
        standard is the same variable in every planned cycle are switched
        here; per-cycle ones (gender) are handled by the planner rules."""
        if plan.get("action") == "clarify":
            return
        cycles = [str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]]
        for s in standards.matching(question or ""):
            if not s.swap_from:
                continue
            targets = {s.variable_for(c) for c in cycles}
            targets.discard(None)
            if len(targets) != 1:
                continue
            target = targets.pop()
            spelled = standards.named_in(question, s.swap_from)
            for fld in s.fields:
                old = plan.get(fld)
                if not isinstance(old, str) or old not in s.swap_from or old in spelled:
                    continue
                plan[fld] = target
                if s.instrument == "sch_qqq" and str(plan.get("instrument") or "stu_qqq") == "stu_qqq":
                    plan["instrument"] = "stu_sch"
                plan.setdefault("_standardized", []).append(
                    f"{s.construct}: {old} replaced by the standard variable {target} — {s.reason}.")
                self._fire(f"standard:swap:{s.construct.split(' ')[0]}")
        if any(line.startswith("public vs private") for line in plan.get("_standardized") or []):
            plan["_school_type_switched"] = True

    def _prefer_reported_school_type(self, plan: dict) -> None:
        """Kept for callers/tests: the public-vs-private rule is now one entry
        of explorer/standards.py, applied by _apply_standards."""
        self._apply_standards(plan, "public vs private school gap")

    # ---------- cross-cycle comparability of questionnaire indices ----------

    INDEX_LABEL = re.compile(r"\(WLE\)|\bindex\b|\bscale\b", re.IGNORECASE)

    def _trend_comparability(self, expr: str | None, plan: dict, cycles: list[str],
                             overrides: dict) -> str | None:
        """None when a cross-cycle change of `expr` is comparable (test scores
        and shares of response codes are, because the PV scales are linked);
        otherwise the reason it is not."""
        expr = str(expr or "")
        if link_errors.domain_of(expr):
            return None
        alt = {str(ov.get("measure")) for ov in overrides.values()
               if isinstance(ov, dict) and ov.get("measure")}
        renamed = alt and alt != {expr} and \
            frozenset({expr.strip().upper(), *[a.strip().upper() for a in alt]}) in self.RENAMED_TREND_INDICES
        if alt and alt != {expr} and not renamed:
            # a share of a response code on items with the SAME wording under
            # different codes (ST184Q01HA 2018 -> ST263Q02JA 2022) is
            # comparable; the OECD itself reports such trends
            if self._same_item_wording(expr, alt):
                plan.setdefault("_same_wording", []).append(f"{expr[:60]} ~ {', '.join(sorted(alt))[:80]}")
                return None
            return ("different variables are used in different cycles "
                    f"({expr} vs {', '.join(sorted(alt))}), so the values are not on one scale")
        tokens = [t for t in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
                  if len(t) >= 3 and t.upper() == t]
        for t in tokens:
            desc = catalog.describe(t)
            if desc.empty:
                continue
            label = str(desc.iloc[-1].label or "")
            if t == "ESCS" or self.INDEX_LABEL.search(label):
                if t != "ESCS":
                    trend = self._trend_scaled(t, cycles, overrides)
                    if trend:
                        plan.setdefault("_trend_index", {})[t] = trend
                        return None
                return (f"{label} ({t}) is a questionnaire index standardized within each "
                        "PISA cycle (OECD mean 0, SD 1), so its level is comparable across "
                        "economies within a cycle but its change across cycles is not")
        return None

    def _same_item_wording(self, expr: str, alternatives: set) -> bool:
        """True when every expression is a share of response codes on items
        whose codebook wording is the same once the stem ("Agree:",
        "Agree/disagree:") and punctuation are removed."""
        import difflib

        def item_key(e: str) -> str | None:
            core = self._unwrap_null_safe(str(e))
            if not re.match(r"^\s*CASE\b", core, re.I) or "100" not in core:
                return None
            vars_ = self._case_variables(core)
            if len(vars_) != 1 or link_errors.domain_of(vars_[0]):
                return None
            desc = catalog.describe(vars_[0])
            if desc.empty:
                return None
            label = str(desc.iloc[-1].label or "").lower()
            label = re.sub(r"^[^:]{0,60}:\s*", "", label)          # drop the stem
            label = label.replace("can't", "cannot").replace("’", "'")
            label = re.sub(r"[^a-z0-9 ]", " ", label)
            m = re.search(rf"\b{re.escape(vars_[0])}\s*(IN\s*\([^)]*\)|[<>=!]+\s*[-\d.]+)", core.split("THEN")[0], re.I)
            codes = re.sub(r"\s+", "", m.group(1)).upper() if m else ""
            return re.sub(r"\s+", " ", label).strip() + "|" + codes

        keys = [item_key(e) for e in [expr, *alternatives]]
        if any(k is None for k in keys):
            return False
        base_label, base_codes = keys[0].split("|")
        for k in keys[1:]:
            label, codes = k.split("|")
            if codes != base_codes or difflib.SequenceMatcher(None, base_label, label).ratio() < 0.85:
                return False
        return True

    # Indices the OECD renamed between cycles while keeping the scale
    RENAMED_TREND_INDICES = {frozenset({"BEINGBULLIED", "BULLIED"})}
    # Known trend scales (OECD reports their change across cycles)
    KNOWN_TREND_INDICES = {"BULLIED", "BEINGBULLIED", "BELONG"}

    def _index_oecd_mean(self, var: str, cycle: str) -> float | None:
        """Unweighted mean over OECD members of the weighted country means of
        an index — ~0 when the OECD re-standardized the index in that cycle."""
        cache = self.__dict__.setdefault("_index_mean_cache", {})
        key = (var, cycle)
        if key not in cache:
            try:
                rows = self.con.execute(
                    f"SELECT CNT, SUM(W_FSTUWT * {var}) / SUM(W_FSTUWT) FROM stu_qqq_{cycle} "
                    f"WHERE {var} IS NOT NULL AND OECD = 1 GROUP BY CNT").fetchall()
                vals = [r[1] for r in rows if r[1] is not None]
                cache[key] = float(np.mean(vals)) if len(vals) >= 5 else None
            except Exception:  # noqa: BLE001 — not in this cycle's file
                cache[key] = None
        return cache[key]

    def _trend_scaled(self, var: str, cycles: list[str], overrides: dict) -> dict | None:
        """Evidence that an index was kept on an earlier cycle's scale rather
        than re-standardized: its OECD-average is clearly not 0 in a later
        cycle (a re-standardized index has OECD mean 0 by construction), or
        it is a documented trend scale. Returns {cycle: OECD mean} or None."""
        var = var.upper()
        cyc = sorted(str(c) for c in cycles)
        if len(cyc) < 2:
            return None
        names = {c: var for c in cyc}
        for c, ov in overrides.items():
            if isinstance(ov, dict) and ov.get("measure") and str(c) in names:
                m = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(ov["measure"]))
                if m:
                    names[str(c)] = m[0].upper()
        means = {c: self._index_oecd_mean(names[c], c) for c in cyc}
        later = [c for c in cyc[1:] if means.get(c) is not None]
        if any(abs(means[c]) >= 0.05 for c in later) or var in self.KNOWN_TREND_INDICES:
            return {c: round(m, 3) for c, m in means.items() if m is not None}
        return None

    def _blank_non_comparable_changes(self, table: pd.DataFrame, plan: dict,
                                      measures: list, cycles: list[str], overrides: dict) -> None:
        if "change" not in table.columns:
            return
        for label, expr in measures:
            reason = self._trend_comparability(expr, plan, cycles, overrides)
            if not reason:
                continue
            mask = (table["measure"] == label) if (label is not None and "measure" in table.columns) \
                else pd.Series(True, index=table.index)
            table.loc[mask, ["change", "se_change"]] = np.nan
            head = ("This measure: " if label is None else
                    "" if label.split(" (")[0] in reason else f"{label}: ")
            plan.setdefault("_non_comparable", []).append(
                f"{head}{reason[0].upper() + reason[1:]}. The change column "
                "is left blank; compare economies within a cycle instead. (For ESCS the "
                "OECD publishes a rescaled trend variable covering 2015–2022; it is in the "
                "database as escs_trend but not yet wired to these templates.)")

    # ---------- small deterministic guards found by adversarial testing ----------

    FALSE_CLAIM_WORDS = re.compile(
        r"(not|n[’']t|never) (collected|administered|available|released|present|provided|"
        r"asked|included)|(is|was|were) (missing|unavailable|absent)", re.IGNORECASE)

    MALE_NOTE = (
        "PISA 2025 uses the derived MALE flag (1 = Male, 0 = Female/Other) for gender in "
        "every economy — the standard convention, not a missing-data workaround: gender "
        "was collected from sampling data everywhere, and ST004D01T is simply not "
        "released in the 2025 public-use file for 14 economies. MALE = 0 pools students "
        "who answered 'other' with female (the OECD's main gender gap leaves them out; "
        "the two are not separable in the public file). 2018 and 2022 use ST004D01T "
        "(1 = Female, 2 = Male).")

    def _verify_substitution_claims(self, plan: dict) -> None:
        # "ST004D01T was not collected for X in 2025" is always the wrong
        # wording: the variable is withheld from the public file, not
        # uncollected — replace it whatever the coverage table says
        note0 = str(plan.get("substitution_note") or "")
        if "2025" in " ".join(str(c) for c in plan.get("cycles") or []) and "ST004D01T" in note0 \
                and self.FALSE_CLAIM_WORDS.search(note0):
            plan["substitution_note"] = self.MALE_NOTE
            plan["_claim_corrected"] = ["ST004D01T"]
            return
        """A substitution_note that says a variable was not collected for the
        named economies is checked against the coverage table; when the
        economies do have it, the false clause is replaced by the plain
        convention statement (the MALE flag is a convention, not a gap)."""
        overrides = {k: v for k, v in (plan.get("cycle_overrides") or {}).items() if isinstance(v, dict)}
        nested = [str(v) for ov in overrides.values() for k, v in ov.items()
                  if k.endswith("_note") and v]
        note = " ".join([str(plan.get("substitution_note") or "")] + nested).strip()
        if not note or not self.FALSE_CLAIM_WORDS.search(note):
            return
        named = set(re.findall(r"'([A-Z]{3})'", str(plan.get("where") or "")))
        if not named:
            return
        cycles = [str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]]
        instrument = plan.get("instrument") or "stu_qqq"
        # only the variable(s) named as the SUBJECT of a "not collected" claim
        # (the 80 characters before each claim), not every code in the note
        claimed = set()
        for m in self.FALSE_CLAIM_WORDS.finditer(note):
            window = note[max(0, m.start() - 80):m.start()]
            subjects = [t for t in re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", window)
                        if not catalog.describe(t).empty]
            if subjects:
                claimed.add(subjects[-1])        # the nearest code is the claim's subject
        claimed = sorted(claimed)
        false = []
        for var in claimed:
            have_all = True
            for cycle in cycles:
                present = named & self.present.get(cycle, set())
                if not present:
                    continue
                cov = self._coverage_for(var, instrument, cycle)
                if cov is None:
                    have_all = False        # unknown: do not judge
                    break
                if cov["partial"] and present & cov["missing"]:
                    have_all = False
                    break
            if have_all:
                false.append(var)
        if not false:
            return
        for ov in overrides.values():                # the corrected wording lives at plan level only
            for k in [k for k in ov if k.endswith("_note")]:
                ov.pop(k, None)
        if "ST004D01T" in false:
            plan["substitution_note"] = self.MALE_NOTE
        else:
            plan["substitution_note"] = (
                f"{', '.join(sorted(false))} is available for the economies in this "
                "question; the planner's substitution was not needed for coverage reasons.")
        plan["_claim_corrected"] = sorted(false)

    @staticmethod
    def _category_label(variable: str, value, table: str) -> str:
        """'MALE = 0 (Female/Other)': the code plus its codebook label."""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        text = f"{variable} = {value}"
        try:
            desc = catalog.describe(variable, cycle=table.rsplit("_", 1)[-1])
            desc = desc[desc.table_name.isin(Agent.underlying_tables(table))]
            if not desc.empty and desc.iloc[0].value_labels:
                labels = json.loads(desc.iloc[0].value_labels)
                key = str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(value)
                label = labels.get(key) or labels.get(f"{key}.0")
                if label:
                    text += f" ({label})"
        except Exception:  # noqa: BLE001 — a label is a nicety, never a failure
            pass
        return text

    CI_WORDS = re.compile(r"\b(confidence intervals?|95\s*%\s*ci\b|\bci\b|intervalos? de confianza|"
                          r"konfidenzintervall\w*|intervalle de confiance)", re.I)
    CHINA_WORDS = re.compile(r"\b(shanghai|beijing|jiangsu|zhejiang|china|chinese)\b", re.I)
    RANK_WORDS = re.compile(r"\b(rank|ranks|ranked|ranking|rankings|order)\b", re.I)

    def _qci_note(self, question: str, plan: dict) -> str | None:
        text = f"{plan.get('where') or ''} {plan.get('regions') or ''}"
        if "QCI" in text and self.CHINA_WORDS.search(question or "") \
                and not re.search(r"\bB-S-J-Z\b", question or "", re.I):
            return ("QCI is B-S-J-Z (China): the four provinces Beijing, Shanghai, Jiangsu "
                    "and Zhejiang assessed together. Shanghai (or any single province) is "
                    "not reported separately, and China as a whole is not in PISA; the "
                    "figures are for the four-province group.")
        return None

    def _keep_full_ranking(self, question: str, plan: dict) -> None:
        """"Rank X and name the largest" asks for the ranking; a planner top_n of
        1 would throw the table away and leave only the winner."""
        if plan.get("top_n") == 1 and plan.get("sort_by") and self.RANK_WORDS.search(question or ""):
            plan["top_n"] = None
            plan["_ranking_kept"] = True

    # The OECD does not report estimates based on fewer than 30 students; the
    # PUF terms forbid anything that could identify a school's responses. A
    # small group is blanked, counted, and explained in the provenance.
    MIN_STUDENTS = 30
    MIN_SCHOOLS = 5
    MIN_COVERAGE = 0.85      # weighted share of the group with an observed value

    def _suppress_small_cells(self, res: pd.DataFrame, cycle: str, plan: dict) -> pd.DataFrame:
        """The OECD reports no estimate based on fewer than 30 students or
        fewer than 5 schools (Reader's Guide, symbol "c"). Such cells are
        blanked, remembered (so later notes do not call them "not released")
        and counted; groups where much of the weighted population has no
        value on the variable are recorded for a coverage note."""
        if res is None or "n" not in res.columns:
            return res
        n = res["n"].fillna(0).astype(int)
        few_students = n < self.MIN_STUDENTS
        few_schools = pd.Series(False, index=res.index)
        if "n_schools" in res.columns:
            ns = res["n_schools"].fillna(-1).astype(int)
            few_schools = (ns >= 0) & (ns < self.MIN_SCHOOLS)
        has_est = res["estimate"].notna() if "estimate" in res.columns else pd.Series(True, index=res.index)
        small = (few_students | few_schools) & has_est
        # identity of a row = its grouping values ("contrast" is one label per
        # table and gets renamed later, so it is not part of the identity)
        keys = [c for c in res.columns if c not in ("estimate", "se", "n_pv", "n", "n_schools",
                                                     "wcov", "contrast", "category")]
        if small.any():
            res = res.copy()
            res.loc[small, [c for c in ("estimate", "se") if c in res.columns]] = np.nan
            plan.setdefault("_suppressed", {})[cycle] = {
                "students": int((few_students & has_est).sum()),
                "schools": int((few_schools & ~few_students & has_est).sum())}
            plan.setdefault("_suppressed_rows", {}).setdefault(cycle, []).extend(
                {c: str(v) for c, v in zip(keys, row)} for row in res.loc[small, keys].itertuples(index=False))
            self._fire("hook:small_cell_suppressed")
        if "wcov" in res.columns:
            low = res["wcov"].notna() & (res["wcov"] < self.MIN_COVERAGE) & has_est & ~small
            if low.any():
                for row, cov in zip(res.loc[low, keys].itertuples(index=False), res.loc[low, "wcov"]):
                    plan.setdefault("_low_coverage", {}).setdefault(cycle, []).append(
                        (", ".join(str(v) for v in row), round(100 * (1 - float(cov)))))
                self._fire("hook:low_coverage")
        return res.drop(columns=[c for c in ("n", "n_schools", "wcov") if c in res.columns])

    @staticmethod
    def _drop_null_groups(res: pd.DataFrame, by) -> tuple[pd.DataFrame, dict]:
        """Rows whose grouping value is missing are not a group (OECD practice
        drops them); returned separately so provenance can say so."""
        dropped = {}
        for col in by:
            if col == "CNT" or col not in res.columns:
                continue
            null = res[col].isna()
            if null.any():
                dropped[col] = int(null.sum())
                res = res[~null]
        return res, dropped

    # ---------- several measures at once ----------

    def _measure_list(self, plan: dict) -> list[tuple[str, str]]:
        """[(label, expression)] when a weighted_mean plan carries two or more
        `measures`; [] otherwise (a single one is folded into `measure`)."""
        if plan.get("template") != "weighted_mean":
            return []
        raw = plan.get("measures")
        if not isinstance(raw, list):
            return []
        exprs = [str(m).strip() for m in raw if m and str(m).strip().lower() not in ("none", "null")]
        exprs = list(dict.fromkeys(exprs))
        if len(exprs) == 1 and not plan.get("measure"):
            plan["measure"] = exprs[0]
        if len(exprs) < 2:
            return []
        for e in exprs:
            self._check_fragment(e)
        labels = []
        for e in exprs:
            label = self._measure_label(e)
            while label in labels:
                label += " (2)"
            labels.append(label)
        return list(zip(labels, exprs))

    # "CASE WHEN <var> <op> <value> THEN 100.0 ELSE 0.0 END" — the planner's
    # form for a share (a proficiency threshold or a response code).
    SHARE_CASE = re.compile(
        r"^\s*CASE\s+WHEN\s+(?:\(?\s*)?(?P<var>[A-Za-z_][A-Za-z0-9_{}]*)\s*\)?\s*"
        r"(?P<op><=|>=|<|>|=|IN)\s*(?P<val>\(?[^)]*?\)?|[-\d.]+)\s+THEN\s+100(?:\.0)?"
        r"(?:\s+WHEN\s+(?P=var)\s+(?:IN\s*\([^)]*\)|IS NOT NULL|[<>=!]+\s*[-\d.]+)\s+THEN\s+0(?:\.0)?)?"
        r"(?:\s+ELSE\s+(?:0(?:\.0)?|NULL))?\s+END\s*$", re.IGNORECASE)
    LEVEL_CUTOFFS = {"420.07": "Level 2 (mathematics)", "407.47": "Level 2 (reading)",
                     "409.54": "Level 2 (science)", "606.99": "Level 5 (mathematics)",
                     "625.61": "Level 5 (reading)", "633.33": "Level 5 (science)"}

    @classmethod
    def _measure_label(cls, expr: str) -> str:
        expr = cls._unwrap_null_safe(str(expr or ""))
        if link_errors.is_mean_score(expr):
            return link_errors.DOMAIN_NAMES[link_errors.domain_of(expr)].capitalize() + " score"
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            desc = catalog.describe(expr)
            if not desc.empty and desc.iloc[-1].label:
                return f"{desc.iloc[-1].label[:50]} ({expr})"
            return expr
        m = cls.SHARE_CASE.match(expr or "")
        if m:
            var, op, val = m.group("var"), m.group("op").upper(), m.group("val").strip()
            if link_errors.domain_of(var):
                domain = link_errors.domain_of(var)
                dom = link_errors.DOMAIN_NAMES[domain]
                level = link_errors.level_name(domain, val)
                side = {"<": "below", "<=": "at or below", ">=": "at or above", ">": "above"}.get(op, "at")
                return f"% {side} {level if level else val + ' points'} in {dom}"
            desc = catalog.describe(var)
            label = desc.iloc[-1].label[:50] if not desc.empty and desc.iloc[-1].label else var
            value_text = val
            try:
                labels = json.loads(desc.iloc[-1].value_labels) if not desc.empty and desc.iloc[-1].value_labels else {}
                key = val.rstrip("0").rstrip(".") if "." in val else val
                if op == "IN":
                    # every code in the list with its label: "(1, 2) [1 =
                    # Strongly disagree; 2 = Disagree]" — the direction of a
                    # share is then in the statement, where the prose check sees it
                    keys = [k.strip().rstrip("0").rstrip(".") if "." in k else k.strip()
                            for k in re.findall(r"[-\d.]+", val)]
                    decoded = [f"{k} = {labels.get(k) or labels.get(k + '.0')}" for k in keys
                               if k in labels or k + ".0" in labels]
                    value_text = f"({', '.join(keys)})" + (f" [{'; '.join(decoded)}]" if decoded else "")
                else:
                    value_text = f"{key} ({labels[key] if key in labels else labels[key + '.0']})" if (key in labels or key + '.0' in labels) else key
            except Exception:  # noqa: BLE001 — a label is a nicety
                pass
            return f"% with {label} ({var}) {op.lower() if op == 'IN' else op} {value_text}"
        if re.match(r"^\s*CASE\b", expr or "", re.IGNORECASE):
            names = []
            for v in Agent._case_variables(Agent, expr)[:3]:
                desc = catalog.describe(v)
                lab = desc.iloc[-1].label[:40] if not desc.empty and desc.iloc[-1].label else v
                names.append(f"{lab} ({v})")
            if names:
                return "% of students meeting a condition on " + ", ".join(names)
        return expr[:60]

    FRACTION_THEN = re.compile(r"\bTHEN\s+(1|0)(?:\.0)?\b(?!\s*\.)", re.IGNORECASE)
    ELSE_ZERO = re.compile(r"\bELSE\s+0(?:\.0)?\s+END\s*$", re.IGNORECASE)

    def _case_variables(self, expr: str) -> list[str]:
        """Catalog variables named inside a CASE expression."""
        out = []
        for tok in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr.replace("{pv}", "1"))):
            if len(tok) >= 3 and tok.upper() == tok and tok not in self.SQL_WORDS \
                    and not catalog.describe(tok).empty:
                out.append(tok)
        return out

    SQL_WORDS = {"CASE", "WHEN", "THEN", "ELSE", "END", "AND", "NOT", "NULL", "BETWEEN",
                 "LIKE", "COALESCE", "CAST", "TRUE", "FALSE", "ABS", "ROUND"}

    def _null_safe_share(self, expr: str | None) -> str | None:
        """A share written as CASE ... ELSE 0 END turns students who never
        saw the question (NULL) into zeros, so a share of "yes" answers in an
        economy that did not administer the item came out as 0.0 (SE 0.0).
        Keep NULL as NULL: the denominator is then valid respondents, exactly
        as weighted_proportion does with valid_values. A share written as
        1/0 (a fraction) is rescaled to 100/0 so it reads as a percentage
        like every other share."""
        if not expr or not re.match(r"^\s*CASE\b", expr, re.IGNORECASE):
            return expr
        text = expr.strip()
        thens = self.FRACTION_THEN.findall(text)
        if thens and set(thens) <= {"1", "0"} and "100" not in text:
            text = self.FRACTION_THEN.sub(lambda m: f"THEN {'100.0' if m.group(1) == '1' else '0.0'}", text)
        if self.ELSE_ZERO.search(text):
            vars_ = self._case_variables(text)
            if vars_:
                guard = vars_[0] if len(vars_) == 1 else "COALESCE(" + ", ".join(vars_) + ")"
                text = f"CASE WHEN ({guard}) IS NULL THEN NULL ELSE ({text}) END"
        return text if text != expr.strip() else expr

    def _null_safe_dummy(self, expr: str | None) -> str | None:
        """A categorical predictor written CASE WHEN IMMIG = 2 THEN 1 ELSE 0 END
        puts students with no value on IMMIG into the reference category
        (natives). Keep NULL as NULL so they drop out listwise, as the method
        note says — no rescaling (a dummy stays 0/1)."""
        if not isinstance(expr, str) or not re.match(r"^\s*CASE\b", expr, re.IGNORECASE):
            return expr
        text = expr.strip()
        if self.ELSE_ZERO.search(text):
            vars_ = self._case_variables(text)
            if vars_:
                guard = vars_[0] if len(vars_) == 1 else "COALESCE(" + ", ".join(vars_) + ")"
                return f"CASE WHEN ({guard}) IS NULL THEN NULL ELSE ({text}) END"
        return expr

    # Measures a question names, matched against what the plan actually
    # uses — a requested measure the plan leaves out is stated, never dropped.
    MEASURE_WORDS = (
        ("mathematics", "MATH", re.compile(r"\bmath(s|ematics|ematical)?\b", re.I)),
        ("reading", "READ", re.compile(r"\breading\b", re.I)),
        ("science", "SCIE", re.compile(r"\bscience\b", re.I)),
        ("socio-economic status (ESCS)", "ESCS",
         re.compile(r"\bescs\b|socio-?economic|\bses\b", re.I)),
    )

    def _dropped_measures(self, question: str, plan: dict) -> list[str]:
        named = [(name, token) for name, token, rx in self.MEASURE_WORDS
                 if rx.search(question or "")]
        if len(named) < 2:
            return []
        fields = {k: v for k, v in plan.items()
                  if not k.startswith("_") and k not in
                  ("explanation", "clarify", "limitation_note", "substitution_note")}
        text = json.dumps(fields, default=str).upper()
        return [name for name, token in named if token not in text]

    def _dropped_economies(self, question: str, plan: dict) -> list[str]:
        """Economies the question names that the plan neither filters to, nor
        covers through a region or a benchmark group — stated, never dropped."""
        named = self._economies_in_data(question)
        if len(named) < 2:
            return []
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        where = " ".join([str(plan.get("where") or "")] + [
            str(ov.get("where") or "") for ov in (plan.get("cycle_overrides") or {}).values()
            if isinstance(ov, dict)])
        if not where.strip() and not plan.get("regions"):
            return []                       # an all-economies plan covers everyone
        covered = set(re.findall(r"'([A-Z]{3})'", where))
        for c in cycles:
            for info in (self._region_filter(plan, [c]) or {}).values():
                covered |= set(info["codes"])
            for bench in self._benchmarks(plan):
                resolved = self._average_members(bench, c)
                if resolved:
                    covered |= resolved[1]
        return [c for c in named if c not in covered]

    # "Why can't I see maths results for Uzbekistan?" — answered from the
    # coverage table plus the OECD's own stated reasons, never improvised.
    WHY_MISSING_WORDS = re.compile(
        r"\bwhy\b.{0,80}\b(no|not|can[’']?t|cannot|don[’']?t|doesn[’']?t|isn[’']?t|aren[’']?t|missing|absent|blank|empty|unavailable|"
        r"omitted|excluded|withheld)\b.{0,80}\b(results?|scores?|data|estimates?|values?|pvs?|"
        r"plausible values)\b|\bwhy (is|are) (there )?(no|not any)\b|"
        r"\b(no|missing|blank|without) (math\w*|reading|science) (results?|scores?|data) for\b",
        re.IGNORECASE | re.DOTALL)
    RELEASE_REASONS = {
        ("UZB", "2025"): ("The OECD's PISA 2025 Technical Report (Data Adjudication chapter) "
                          "states that the adjudication group identified inconsistencies in "
                          "the reading and mathematics response data for Uzbekistan that "
                          "required further investigation before they could be considered fit "
                          "for reporting; only the science results were released."),
    }
    DOMAIN_PV = {"MATH": "PV1MATH", "READ": "PV1READ", "SCIE": "PV1SCIE"}

    def _missing_results_answer(self, codes: list[str]) -> str:
        parts = []
        for code in codes:
            name = self.economy_names.get(code, code)
            for cycle in sorted(self.present):
                if code not in self.present[cycle]:
                    parts.append(f"{name} ({code}) did not take part in PISA {cycle}.")
                    continue
                have, lack = [], []
                for dom, pv in self.DOMAIN_PV.items():
                    cov = catalog.coverage(pv, f"stu_qqq_{cycle}")
                    (lack if cov and cov["partial"] and code in cov["missing"] else have).append(
                        link_errors.DOMAIN_NAMES[dom])
                if not lack:
                    parts.append(f"PISA {cycle}: {name} has results in all three domains "
                                 f"({', '.join(have)}); ask for them directly.")
                    continue
                line = (f"PISA {cycle}: {name} took part, but the public-use database holds "
                        f"its results for {', '.join(have) or 'no domain'} only — the "
                        f"{' and '.join(lack)} plausible values were not released by the OECD, "
                        "so this app cannot compute them.")
                reason = self.RELEASE_REASONS.get((code, cycle))
                line += f" {reason}" if reason else (" The OECD did not publish a reason in the "
                                                     "documents held here.")
                parts.append(line)
        return " ".join(parts)

    # "How many students were tested in X?" with nothing else asked — a fact
    # read from the tables, not a plan.
    COUNT_WORDS = re.compile(
        r"\bhow many (students|pupils|children|kids|schools|participants)\b|"
        r"\bnumber of (students|pupils|schools|participants) (tested|sampled|assessed|"
        r"participat\w*|took part|surveyed)|\bsample sizes?\b", re.IGNORECASE)
    # Anything that makes "how many students" a statistic rather than a
    # sample size: a share, a domain, a group — or a behaviour ("how many
    # students USE AI chatbots" asked for a share and was answered with the
    # sample size until the guard telemetry showed the hijack).
    OTHER_STAT_WORDS = re.compile(
        r"\b(percent\w*|share|proportion|average|mean|score|scores|gap|trend|rank\w*|"
        r"correlat\w*|compare|comparison|girls|boys|female|male|below|above|level|"
        r"use|uses|used|using|report\w*|say|said|have|has|had|feel|felt|agree\w*|"
        r"repeat\w*|skip\w*|speak\w*|attend\w*|ai|chatbots?|immigra\w*|bull\w*|"
        r"belong\w*|satisf\w*)\b",
        re.IGNORECASE)

    def _count_total_answer(self, question: str) -> str:
        years = set(self.YEAR_RE.findall(question))
        parts = []
        for cycle in sorted(self.present):
            if years and cycle not in years:
                continue
            n, schools, econ, wsum = self.con.sql(
                f"SELECT count(*), count(DISTINCT CNTSCHID), count(DISTINCT CNT), sum(W_FSTUWT) "
                f"FROM stu_qqq_{cycle}").fetchone()
            parts.append(f"PISA {cycle}: {int(n):,} students sampled in {int(schools):,} schools "
                         f"across {int(econ)} economies, representing an estimated "
                         f"{int(round(wsum)):,} 15-year-olds (sum of final student weights).")
        return " ".join(parts)

    def _count_answer(self, codes: list[str], question: str) -> str:
        years = set(self.YEAR_RE.findall(question))
        parts = []
        for code in codes:
            name = self.economy_names.get(code, code)
            for cycle in sorted(self.present):
                if years and cycle not in years:
                    continue
                if code not in self.present[cycle]:
                    parts.append(f"{name} ({code}) did not take part in PISA {cycle}.")
                    continue
                n, schools, wsum = self.con.sql(
                    f"SELECT count(*), count(DISTINCT CNTSCHID), sum(W_FSTUWT) "
                    f"FROM stu_qqq_{cycle} WHERE CNT = '{code}'").fetchone()
                parts.append(f"PISA {cycle}: {name} ({code}) — {int(n):,} students sampled in "
                             f"{int(schools):,} schools, representing an estimated "
                             f"{int(round(wsum)):,} 15-year-olds (sum of final student weights).")
        return " ".join(parts) + (" Sample sizes also accompany every analysis in its "
                                  "provenance card." if parts else "")

    # "What does PISA measure?" — the overview is a fact about the data, not
    # a catalog search.
    OVERVIEW_WORDS = re.compile(
        r"\bwhat (does|do|is|are|can) (the )?pisa (measur|assess|cover|test|evaluat|includ)|"
        r"\bwhat (variables|things|subjects|domains|areas|topics|skills|competenc\w*|"
        r"indicators|dimensions) (does|do|is|are) (the )?pisa (measur|assess|cover|test)|"
        r"what is (measured|assessed) in pisa|what does pisa (look at|examine)",
        re.IGNORECASE)

    def _overview_answer(self) -> str:
        innov = "; ".join(f"{c}: {self.INNOVATIVE[c].split(' (')[0]}" for c in self.coverage)
        return (
            "PISA tests 15-year-olds in three core domains every cycle — mathematics, "
            "reading and science — reported as plausible-value scores (PV1–PV10) "
            "on a common scale (OECD mean about 500, Level 2 is the baseline "
            "proficiency), plus one innovative domain per cycle (" + innov + "). "
            "Around the tests are questionnaires: the student questionnaire "
            "(family background and the ESCS socio-economic index, immigrant "
            "background, attitudes such as mathematics anxiety and self-efficacy, "
            "well-being and sense of belonging, learning time, ICT use), the "
            "school questionnaire (public/private, location, size, resources, "
            "climate), and in some economies teacher and parent questionnaires. "
            "Every measure exists for every participating economy — 80 in 2018 "
            "and 2022, 90 in 2025 — so a region such as Latin America is just a "
            "filter. Ask for a statistic (“mean reading score in Latin American "
            "countries in 2025”, “math, reading and ESCS for Argentina”) or list "
            "variables on a topic (“find me data on well-being”).")

    # ---------- joined view plumbing ----------

    @staticmethod
    def underlying_tables(table: str) -> list[str]:
        """The physical tables behind a table name (a joined view maps to
        its student and school tables); catalog lookups use these."""
        m = re.fullmatch(r"stu_sch_(\d{4})", table)
        if m:
            return [f"stu_qqq_{m.group(1)}", f"sch_qqq_{m.group(1)}"]
        m = re.fullmatch(r"stu_crt_(\d{4})", table)
        if m:
            return [f"stu_qqq_{m.group(1)}", f"crt_cog_{m.group(1)}"]
        return [table]

    def _table_columns(self, table: str) -> set[str]:
        cache = self.__dict__.setdefault("_columns_cache", {})
        if table not in cache:
            cache[table] = {r[0] for r in self.con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?", [table]).fetchall()}
        return cache[table]

    def _check_columns(self, plan: dict, table: str) -> None:
        """Every catalog variable a plan uses must exist in the table it runs
        on — otherwise DuckDB's binder error reaches the user. A school
        variable used on a student table gets a message naming the fix."""
        columns = self._table_columns(table)
        if not columns:
            return
        fields = self.COVERAGE_FIELDS + ("where", "by")
        if plan.get("_skip_measures"):
            fields = tuple(f for f in fields if f != "measures")
        text = " ".join(str(plan.get(k) or "") for k in fields).replace("{pv}", "1")
        for token in self._unknown_columns(text, table):
            tables = sorted(set(catalog.describe(token).table_name))
            hint = ""
            if any(t.startswith("sch_qqq") for t in tables) and table.startswith("stu_qqq"):
                hint = (" — it is a SCHOOL questionnaire variable; combine it with "
                        "student outcomes through instrument \"stu_sch\" (students "
                        "joined to their school)")
            elif not any(t.endswith(table.rsplit("_", 1)[-1]) for t in tables):
                hint = f" — it exists only in {', '.join(sorted({t.rsplit('_', 1)[-1] for t in tables}))}"
            raise ValueError(f"{token} is not in {table} (it exists in "
                             f"{', '.join(tables)}){hint}")

    def _unknown_columns(self, text: str, table: str) -> list[str]:
        """Catalog variables named in `text` (any cycle, any instrument) that
        the given table does not have. SQL keywords and literals are not
        catalog variables and are ignored."""
        columns = self._table_columns(table)
        if not columns:
            return []
        out = []
        for token in sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.replace("{pv}", "1")))):
            if token in columns or len(token) < 3:
                continue
            if catalog.describe(token).empty:
                continue                     # not a variable: SQL keyword, literal
            out.append(token)
        return out

    def _coverage_for(self, var: str, instrument: str, cycle: str) -> dict | None:
        for tbl in self.underlying_tables(f"{instrument}_{cycle}"):
            cov = catalog.coverage(var, tbl)
            if cov is not None:
                return cov
        return None

    # ---------- coverage guard (offline-testable, no LLM) ----------

    COVERAGE_FIELDS = ("measure", "measures", "variable", "group_col", "quart_variable",
                       "x", "y", "row_var", "col_var", "predictors")

    def _plan_variables(self, plan: dict) -> list[str]:
        """Identifiers used as analysis variables (not filters/grouping)."""
        text = " ".join(str(plan.get(k) or "") for k in self.COVERAGE_FIELDS)
        text = text.replace("{pv}", "1")
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        return sorted(t for t in tokens if len(t) >= 3 and t.upper() == t)

    def _coverage_check(self, plan: dict, cycles: list[str], instrument: str,
                        region_codes: dict, overrides: dict) -> dict[str, list[dict]]:
        """Per cycle: plan variables that some economies never collected.
        A finding is `blocked` when EVERY economy the plan names lacks the
        variable in that cycle — the cycle cannot yield an estimate."""
        out: dict[str, list[dict]] = {}
        for cycle in cycles:
            cplan = {**plan, **overrides.get(cycle, {})}
            table = f"{instrument}_{cycle}"
            named = set(re.findall(r"'([A-Z]{3})'", str(cplan.get("where") or "")))
            if cycle in region_codes:
                named |= set(region_codes[cycle]["codes"])
            present = self.present.get(cycle)
            if present:
                named &= present
            findings = []
            unlinked = getattr(self, "no_school_link", {}).get(cycle, set())
            if instrument == "stu_sch" and unlinked:
                # students without a school identifier cannot be joined to
                # their school questionnaire: nothing to estimate for them
                affected = sorted(named & unlinked) if named else sorted(unlinked)
                if affected:
                    findings.append({
                        "variable": "CNTSCHID", "label": "student-to-school link", "cycle": cycle,
                        "n_with_data": len(present or ()) - len(unlinked), "n_economies": len(present or ()),
                        "with_data": sorted((present or set()) - unlinked), "missing": sorted(unlinked),
                        "named": sorted(named), "named_missing": affected,
                        "blocked": bool(named) and len(affected) == len(named),
                    })
            for var in self._plan_variables(cplan):
                cov = self._coverage_for(var, instrument, cycle)
                if not cov or not cov["partial"]:
                    continue
                named_missing = sorted(named & cov["missing"])
                if not named_missing and (named or
                        cov["n_economies"] - cov["n_with_data"] < self.COVERAGE_NOISE):
                    continue      # the named economies all have it: nothing to say
                desc = catalog.describe(var, cycle=cycle)
                label = desc.iloc[0].label if not desc.empty else var
                findings.append({
                    "variable": var, "label": label, "cycle": cycle,
                    "n_with_data": cov["n_with_data"], "n_economies": cov["n_economies"],
                    "with_data": sorted(cov["with_data"]), "missing": sorted(cov["missing"]),
                    "named": sorted(named), "named_missing": named_missing,
                    "blocked": bool(named) and len(named_missing) == len(named),
                })
            if findings:
                out[cycle] = findings
        return out

    def _names(self, codes) -> str:
        return ", ".join(f"{self.economy_names.get(c, c)} ({c})" for c in codes)

    def _code_list(self, codes) -> str:
        codes = list(codes)
        if len(codes) <= self.COVERAGE_LIST_MAX:
            return self._names(codes)
        return f"{len(codes)} economies"

    def _school_link_text(self, f: dict) -> str:
        who = self._names(f["named_missing"])
        return (f"PISA {f['cycle']}: the public-use student file carries no school "
                f"identifier for {who}, so students there cannot be linked to their "
                "school questionnaire and no school-variable estimate exists for them "
                "(the school questionnaire itself was administered; student-level "
                "results are unaffected). " +
                ("That cycle is omitted." if f["blocked"] else "Those rows have no estimate."))

    def _coverage_note(self, f: dict) -> str:
        if f["variable"] == "CNTSCHID":
            return self._school_link_text(f)
        if f["variable"] == "ST004D01T" and f["cycle"] == "2025" and f["named_missing"]:
            return (f"PISA 2025: ST004D01T (gender) is not released in the public-use file "
                    f"for {self._names(f['named_missing'])}; gender was collected from "
                    "sampling data everywhere and the derived MALE flag (1 = Male, 0 = "
                    "Female/Other) is complete for all 90 economies — use MALE.")
        head = (f"PISA {f['cycle']}: {f['label']} ({f['variable']}) was collected in "
                f"{f['n_with_data']} of {f['n_economies']} economies")
        domain = link_errors.domain_of(f["variable"])
        if domain and f["named_missing"]:
            dom = link_errors.DOMAIN_NAMES[domain]
            tail = ("that cycle has no estimate and is omitted." if f["blocked"]
                    else "those rows have no estimate.")
            return (f"PISA {f['cycle']}: {self._names(f['named_missing'])} took part, "
                    f"but the OECD did not release its {dom} results in the public "
                    f"database (no plausible values), so {tail}")
        if f["blocked"]:
            return (f"{head} (an optional questionnaire / national option); "
                    f"{self._names(f['named_missing'])} did not administer it, so "
                    f"that cycle has no estimate and is omitted. Economies with data "
                    f"in {f['cycle']}: {self._code_list(f['with_data'])}.")
        if f["named_missing"]:
            return (f"{head}; it was NOT collected for "
                    f"{self._names(f['named_missing'])} — those rows have no "
                    f"estimate (the variable was not administered there, which "
                    f"says nothing about the students).")
        if f["n_with_data"] <= self.COVERAGE_LIST_MAX:
            return (f"{head} (an optional questionnaire): only "
                    f"{self._code_list(f['with_data'])} have estimates; every "
                    f"other economy shows no estimate, so a ranking covers only them.")
        return (f"{head}; not collected in {self._code_list(f['missing'])} — those "
                f"economies show no estimate and are absent from any ranking.")

    def _coverage_message(self, findings: dict) -> str:
        """User-facing statement when every requested cycle is blocked."""
        parts = []
        last = None
        for cycle in sorted(findings):
            for f in findings[cycle]:
                if not f["blocked"]:
                    continue
                who = self._names(f["named_missing"])
                if f["variable"] == "CNTSCHID":
                    parts.append(self._school_link_text(f))
                    last = None
                    continue
                domain = link_errors.domain_of(f["variable"])
                if domain:
                    dom = link_errors.DOMAIN_NAMES[domain]
                    parts.append(
                        f"{who} took part in PISA {cycle}, but the OECD did not "
                        f"release its {dom} results in the public database (no "
                        f"plausible values), so no {dom} estimate exists for that cycle.")
                    last = None
                    continue
                parts.append(
                    f"{f['label']} ({f['variable']}) was collected in only "
                    f"{f['n_with_data']} of {f['n_economies']} economies in PISA "
                    f"{cycle}. It belongs to an optional questionnaire that {who} "
                    f"did not administer, so no estimate exists for {who} in that "
                    f"cycle and none can be computed here.")
                last = f
        if last is not None:
            parts.append(f"Economies with data in PISA {last['cycle']}: "
                         f"{self._code_list(last['with_data'])}.")
        return " ".join(parts)

    def _coverage_alternatives(self, hits: pd.DataFrame, named: set,
                               exclude: set, limit: int = 5) -> list[str]:
        """Retrieved variables that the named economies DID collect (from the
        coverage table — never from the model), best-ranked first."""
        if hits is None or hits.empty:
            return []
        out = []
        for var, group in hits.groupby("variable", sort=False):
            if var in exclude:
                continue
            cycles_ok = []
            for _, r in group.iterrows():
                cov = catalog.coverage(var, r.table_name)
                if cov is None:
                    continue
                if not cov["partial"] or not (named & cov["missing"]):
                    cycles_ok.append(str(r.cycle))
            if cycles_ok:
                label = group.iloc[-1].label
                out.append(f"{label} ({var}; PISA {', '.join(sorted(set(cycles_ok)))})")
            if len(out) >= limit:
                break
        return out

    # User-facing text must not carry the planner's SQL/PV notation.
    PV_CODE = re.compile(r"\bPV(?:\{pv\}|\d{1,2})([A-Z]{4})\b")
    DOMAIN_NAMES = {"MATH": "mathematics score", "READ": "reading score",
                    "SCIE": "science score", "CMPS": "Learning in the Digital World score",
                    "CPPK": "computational practices score", "CMOD": "computational modelling score",
                    "CPRO": "computational problem-solving score",
                    "SEPS": "science (explain phenomena) subscale",
                    "SEDE": "science (design/evaluate) subscale",
                    "SEID": "science (interpret data) subscale",
                    "SENV": "environmental science subscale"}

    @classmethod
    def _plain(cls, text: str) -> str:
        text = cls.PV_CODE.sub(lambda m: cls.DOMAIN_NAMES.get(m.group(1), f"{m.group(1)} score"), text or "")
        return text.replace("{pv}", "").replace("`", "")

    # ---------- execution (offline-testable, no LLM) ----------

    # Fields a template cannot run without. A plan that lacks one used to reach
    # DuckDB as the literal word "None" (Binder Error: column "None") — now it
    # is refused with a message that names the gap.
    REQUIRED_FIELDS = {
        "weighted_mean": [("measure", "measures")],
        "weighted_proportion": [("variable", "measure"), ("value", "measure")],
        "gap": [("measure",), ("group_col",), ("minuend",), ("subtrahend",)],
        "quartile_means": [("measure",), ("quart_variable",)],
        "quartile_gap": [("measure",), ("quart_variable",)],
        "correlation": [("x",), ("y",)],
        "percentiles": [("measure",)],
        "percentile_spread": [("measure",)],
        "crosstab": [("row_var",), ("col_var",)],
        "regression": [("measure",), ("predictors",)],
        "raw_sql": [("sql",)],
    }

    @classmethod
    def _validate_plan(cls, plan: dict) -> None:
        """Raise ValueError naming what an incomplete plan is missing.
        Each entry in REQUIRED_FIELDS is a tuple of alternatives: at least one
        of them must be present and not a null-ish placeholder."""
        template = plan.get("template")
        if template not in cls.REQUIRED_FIELDS:
            raise ValueError(f"unknown template {template!r}")

        def present(key: str) -> bool:
            value = plan.get(key)
            if value is None or value == [] or value == "":
                return False
            return not (isinstance(value, str) and value.strip().lower()
                        in ("none", "null", "nan"))

        missing = [" or ".join(alts) for alts in cls.REQUIRED_FIELDS[template]
                   if not any(present(k) for k in alts)]
        if missing:
            raise ValueError(
                f"the {template} plan is incomplete — missing {', '.join(missing)}. "
                "Try naming the variable or measure you want analyzed.")

    # Plan fields a planner has written into `by` as placeholders for their
    # own value ("by": ["CNT", "group_col"]).
    BY_PLACEHOLDERS = ("group_col", "variable", "quart_variable", "x", "y", "measure")

    def _auto_instrument(self, plan: dict) -> None:
        """A variable that lives only in the school file (RATCMP1, SC001Q01TA)
        or in the creative-thinking file (PV1CRTH_NC) is analysed through the
        joined student view (stu_sch / stu_crt) — the planner sometimes leaves
        the instrument at stu_qqq and the binder error reached the user. A
        single measure that exists in some planned cycles only (creative
        thinking 2022, global competence 2018) keeps those cycles and the
        others are stated, as several measures already are."""
        instrument = str(plan.get("instrument") or "stu_qqq")
        cycles = [str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE] if str(c) in CYCLES]
        if not cycles:
            return
        if instrument in ("sch_qqq", "tch_qqq") and str(plan.get("template")) != "raw_sql":
            # the estimator needs the student weights: school characteristics
            # are reported for the STUDENTS in those schools (the OECD's own
            # convention: "% of students in schools whose principal reports…")
            plan["instrument"] = instrument = "stu_sch"
            plan["_school_file_to_students"] = True
            self._fire("hook:auto_instrument:sch_to_stu_sch")
        text = " ".join(str(plan.get(k) or "") for k in self.COVERAGE_FIELDS + ("where", "by"))
        tokens = [t for t in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.replace("{pv}", "1")))
                  if len(t) >= 3 and t not in self.SQL_WORDS and not catalog.describe(t).empty]
        if instrument == "stu_qqq" and tokens:
            missing = [t for t in tokens if all(t not in self._table_columns(f"stu_qqq_{c}") for c in cycles)]
            if missing:
                in_sch = all(any(t in self._table_columns(f"sch_qqq_{c}") for c in cycles) for t in missing)
                in_crt = all(any(t in self._table_columns(f"crt_cog_{c}") for c in cycles) for t in missing)
                if in_sch:
                    plan["instrument"] = instrument = "stu_sch"
                    plan["_auto_instrument"] = ("stu_sch", missing)
                    self._fire("hook:auto_instrument:stu_sch")
                elif in_crt and self._table_columns("stu_crt_2022"):
                    plan["instrument"] = instrument = "stu_crt"
                    plan["_auto_instrument"] = ("stu_crt", missing)
                    self._fire("hook:auto_instrument:stu_crt")
        # cycles in which the (single) measure does not exist at all
        measure = plan.get("measure")
        if isinstance(measure, str) and measure and not plan.get("measures") and len(cycles) >= 1:
            mtoks = [t for t in dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", measure.replace("{pv}", "1")))
                     if len(t) >= 3 and t not in self.SQL_WORDS and not catalog.describe(t).empty]
            if mtoks:
                overrides = plan.get("cycle_overrides") if isinstance(plan.get("cycle_overrides"), dict) else {}
                have = [c for c in cycles if self._table_columns(f"{instrument}_{c}")
                        and all(t in self._table_columns(f"{instrument}_{c}") for t in mtoks)]
                lacking = [c for c in cycles if self._table_columns(f"{instrument}_{c}") and c not in have
                           and not (isinstance(overrides.get(c), dict) and overrides[c].get("measure"))]
                if have and lacking:
                    plan["cycles"] = have
                    plan["_measure_cycles_dropped"] = lacking
                    self._fire("hook:measure_cycles_dropped")

    def _plain_by(self, cols: list[str], plan: dict) -> list[str]:
        """A CASE expression in `by` ("CASE WHEN MALE = 0 THEN 'Female' …")
        becomes the variable it recodes: the groups are then the variable's
        codes, labelled from the codebook in every statement."""
        out = []
        for b in cols:
            if re.match(r"^\s*CASE\b", b, re.I):
                vars_ = self._case_variables(b)
                if vars_:
                    out.append(vars_[0])
                    plan.setdefault("_case_by_replaced", []).append(f"{b[:60]} → {vars_[0]}")
                    self._fire("hook:case_by_replaced")
                continue
            out.append(b)
        return list(dict.fromkeys(out))

    def _normalize_plan_shape(self, plan: dict, template: str | None) -> None:
        """Deterministic repairs of plan shapes the planner gets wrong in ways
        that would crash or, worse, silently compute the wrong statistic:
        - a `by` entry naming a plan FIELD ("group_col") stands for its value;
        - a gap between two ECONOMIES (group_col CNT) is two per-economy rows,
          whose difference and its change the app states itself (a gap grouped
          by its own group column crashed);
        - a benchmark average ("include_average_of") needs per-economy rows:
          with `by` empty the whole filter was pooled into one student-weighted
          mean of every student in the database and labelled as the group;
        - a filter naming several economies with no CNT grouping is the same
          pooled mean, which is no OECD statistic: the rows are per economy
          and the group's unweighted average is added as a benchmark row."""
        if template == "raw_sql":
            return
        self._auto_instrument(plan)
        by = [b for b in (plan.get("by") or []) if isinstance(b, str)]
        fixed = []
        for b in by:
            if b in self.BY_PLACEHOLDERS and isinstance(plan.get(b), str) and plan.get(b):
                fixed.append(plan[b])
                plan.setdefault("_by_placeholder", []).append(b)
            else:
                fixed.append(b)
        by = self._plain_by(fixed, plan)
        overrides = plan.get("cycle_overrides") if isinstance(plan.get("cycle_overrides"), dict) else {}
        for ov in overrides.values():
            if isinstance(ov, dict) and isinstance(ov.get("by"), list):
                ov["by"] = self._plain_by([b for b in ov["by"] if isinstance(b, str)], plan)
        # a gap grouped by its own group column has one row per group and no
        # contrast (and crashed on duplicate labels)
        if template in ("gap", "quartile_gap") and plan.get("group_col") in by:
            by = [b for b in by if b != plan.get("group_col")]
            plan["_group_col_dropped_from_by"] = True
        where = str(plan.get("where") or "")
        # "ESCS_Q = 4" — an invented quarter column in the filter: the quarters
        # are computed by quartile_means; the wanted quarter(s) filter its rows
        m = re.search(r"(?:\bAND\s+)?\(?\s*\b([A-Z][A-Z0-9]*)_Q(?:UART(?:ER|ILE)?S?)?\s*(=|IN)\s*\(?\s*([\d,\s]+?)\s*\)?\s*\)?",
                      where, re.I)
        if m and template in ("quartile_means", "quartile_gap", "weighted_mean", "weighted_proportion"):
            quarters = [int(x) for x in re.findall(r"\d", m.group(3)) if x in "1234"]
            plan["where"] = re.sub(r"^\s*(AND|OR)\s+", "", where.replace(m.group(0), "").strip(), flags=re.I).strip() or None
            where = plan["where"] or ""
            plan.setdefault("quart_variable", m.group(1).upper())
            if template != "quartile_means":
                plan["template"] = template = "quartile_means"
            plan["_quarter_filter"] = quarters or [4]
            self._fire("hook:quarter_filter")
        if template == "gap" and str(plan.get("group_col") or "").upper() == "CNT":
            a, b = str(plan.get("minuend") or "").upper(), str(plan.get("subtrahend") or "").upper()
            if re.fullmatch(r"[A-Z]{3}", a) and re.fullmatch(r"[A-Z]{3}", b):
                plan.update({"template": "weighted_mean", "where": f"CNT IN ('{a}', '{b}')",
                             "group_col": None, "minuend": None, "subtrahend": None,
                             "_pair": [a, b]})
                template = "weighted_mean"
                by = ["CNT"] + [c for c in by if c.upper() != "CNT"]
                self._fire("hook:economy_gap_to_rows")
        codes = list(dict.fromkeys(re.findall(r"'([A-Z]{3})'", where)))
        several = len(codes) >= 2 and re.search(r"\bCNT\s+IN\b", where, re.I) and \
            not re.search(r"\bNOT\s+IN\b", where, re.I)
        benchmarks = bool(plan.get("include_oecd_average")) or bool(plan.get("include_average_of"))
        if "CNT" not in [c.upper() for c in by] and (benchmarks or several) \
                and template in ("weighted_mean", "weighted_proportion", "gap", "quartile_means",
                                 "quartile_gap", "percentiles", "percentile_spread",
                                 "correlation", "regression"):
            by = ["CNT"] + by
            plan["_pooled_to_rows"] = True
            if several and not benchmarks and len(codes) >= 3:
                # "the average of these countries" = their unweighted average,
                # the OECD-average convention — shown as a benchmark row
                plan["include_average_of"] = [codes]
            self._fire("hook:pooled_to_rows")
        if not where.strip() and not plan.get("regions") and plan.get("_pooled_to_rows"):
            # a benchmark group asked for as ONE number ("average of KAZ, UZB,
            # KGZ, QTJ") with no filter: the rows are the members. A ranking
            # of every economy with the OECD average beside it keeps its rows.
            members = set()
            for bench in self._benchmarks(plan):
                if isinstance(bench, tuple):
                    members |= set(bench)
                elif str(bench).lower() == "oecd":
                    for c in [str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]]:
                        if c in CYCLES:
                            members |= self._oecd_codes(c)
                else:
                    canon = regions.canonical(str(bench))
                    if canon:
                        members |= set(regions.REGIONS.get(canon, []))
            if members:
                plan["where"] = "CNT IN (" + ", ".join(f"'{c}'" for c in sorted(members)) + ")"
                plan["_members_as_rows"] = True
                self._fire("hook:members_as_rows")
        plan["by"] = by

    def execute(self, plan: dict) -> tuple[pd.DataFrame, dict]:
        self._validate_plan(plan)
        template = plan.get("template")
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        unknown = [c for c in cycles if c not in CYCLES]
        if unknown:
            raise ValueError(f"unknown cycle(s) {unknown}; available: {CYCLES}")
        instrument = plan.get("instrument") or "stu_qqq"
        self._normalize_plan_shape(plan, template)
        template = plan.get("template")
        instrument = plan.get("instrument") or "stu_qqq"
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        by = tuple(plan.get("by") or ())
        where = plan.get("where") or None
        self._check_fragment(where)
        for col in by:
            self._check_identifier(col)
        # "cycle"/"year" are not columns: a planner that groups by them wants
        # the cycles side by side, which the app does anyway
        if any(c.lower() in ("cycle", "year", "pisa_cycle") for c in by):
            by = tuple(c for c in by if c.lower() not in ("cycle", "year", "pisa_cycle"))
            plan["by"] = list(by)
            if len(cycles) == 1 and template != "raw_sql":
                cycles = [c for c in CYCLES if c in self.coverage]
                plan["cycles"] = cycles
            self._fire("hook:cycle_by_dropped")
        if template != "raw_sql":
            probe = f"{instrument}_{cycles[-1]}"
            cols = self._table_columns(probe)
            bad = [c for c in by if cols and c not in cols]
            if bad:
                raise ValueError(f"{', '.join(bad)} is not a column of {probe}; group by an "
                                 "existing variable (for example CNT, ST004D01T, IMMIG).")
        overrides = {str(k): v for k, v in (plan.get("cycle_overrides") or {}).items()
                     if isinstance(v, dict) and v}
        for ov in overrides.values():
            for value in ov.values():
                if isinstance(value, str):
                    self._check_fragment(value)
        self._align_gender_direction(plan, overrides)
        # A per-cycle `by` (["CNT", "ST004D01T"] in 2022, ["CNT", "MALE"] in
        # 2025) applies to that cycle; the columns are renamed to the main
        # plan's so the cycles merge. It was silently ignored before.
        by_per_cycle: dict[str, tuple] = {}
        ov_bys = {c: [b for b in ov.get("by") if isinstance(b, str)]
                  for c, ov in overrides.items() if isinstance(ov.get("by"), list) and ov.get("by")}
        if ov_bys and all(c in ov_bys for c in cycles):
            first = ov_bys[cycles[0]]
            if all(len(ov_bys[c]) == len(first) for c in cycles):
                if len(first) > len(by):
                    by = tuple(first)
                    plan["by"] = list(by)
                for c in cycles:
                    if tuple(ov_bys[c]) != by and len(ov_bys[c]) == len(by):
                        by_per_cycle[c] = tuple(ov_bys[c])
                plan["_by_overrides"] = {c: list(v) for c, v in by_per_cycle.items()}
                self._fire("hook:by_override")
        for ov in overrides.values():
            ov.pop("by", None)
        # A stratum fixed by the filter (STRATUM = 'KAZ21', per cycle) must not
        # also be a grouping column: stratum codes differ between cycles, so a
        # trend grouped by STRATUM becomes one blank-riddled row per code.
        wheres = [str(where or "")] + [str(ov.get("where") or "") for ov in overrides.values()]
        if "STRATUM" in by and any(re.search(r"\bSTRATUM\s*=", w, re.I) for w in wheres):
            by = tuple(c for c in by if c != "STRATUM")
            plan["by"] = list(by)
            self._fire("hook:stratum_by_dropped")

        if template == "raw_sql":
            table = self._run_raw_sql(plan["sql"])
            prov = self._provenance(plan, [], raw=True)
            sql_text = str(plan.get("sql") or "")
            pv_single = re.search(r"\bPV(\d{1,2})(MATH|READ|SCIE)\b", sql_text)
            weighted = re.search(r"W_FSTUWT", sql_text, re.I)
            prov["notes"].append(
                "Computed by direct SQL, outside the survey estimator: "
                + ("no student weights were applied, so the figures describe the sample, not "
                   "the population; " if not weighted else "")
                + (f"a single plausible value (PV{pv_single.group(1)}) was used, so the figure "
                   "is illustrative and not a PISA estimate; " if pv_single else "")
                + "no replicate-weight (BRR) standard error is available for it. Ask for the "
                  "statistic by name (mean, share, gap, correlation) to get the official estimate "
                  "with its standard error.")
            return table, prov

        tables = [f"{instrument}_{c}" for c in cycles]
        region_codes = self._region_filter(plan, cycles)
        # Optional-questionnaire coverage: a variable the named economies never
        # collected in a cycle cannot yield an estimate — that cycle is stated
        # (provenance) and skipped; if no cycle is left, the answer is the
        # coverage statement itself, never a summary of an empty table.
        findings = self._coverage_check(plan, cycles, instrument, region_codes, overrides)
        plan["_coverage"] = findings
        blocked = [c for c, fs in findings.items() if any(f["blocked"] for f in fs)]
        if blocked:
            cycles = [c for c in cycles if c not in blocked]
            if not cycles:
                raise CoverageError(self._coverage_message(findings),
                                    self._provenance(plan, tables))
            plan["_no_trend"] = len(cycles) < 2     # no change is computed
        # Several measures for the same groups ("math, reading and ESCS for
        # Argentina"): run each and stack the rows with a `measure` column.
        self._keep_full_ranking(str(plan.get("_question") or ""), plan)
        # A share whose group is a CASE expression belongs in weighted_mean
        # as a 100/0/NULL measure; the planner sometimes files it under
        # weighted_proportion's `variable` (which takes a plain code).
        if template == "weighted_proportion" and re.match(r"^\s*CASE\b", str(plan.get("variable") or ""), re.I):
            cond = str(plan["variable"]).strip()
            valid = plan.get("valid_values") or [0, 1]
            codes = ", ".join(str(v) for v in valid)
            plan["measure"] = (f"CASE WHEN ({cond}) = {plan.get('value')} THEN 100.0 "
                               f"WHEN ({cond}) IN ({codes}) THEN 0.0 ELSE NULL END")
            plan["variable"] = None
            plan["_derived_share"] = True
            self._fire("hook:derived_share")
        measures_all = self._measure_list(plan)
        if template == "regression" and isinstance(plan.get("predictors"), list):
            # categorical dummies keep NULL as NULL (listwise), never as the
            # reference category
            safe = [self._null_safe_dummy(x) if isinstance(x, str) else x for x in plan["predictors"]]
            if safe != plan["predictors"]:
                plan["predictors"] = safe
                plan["_null_safe_dummy"] = True
                self._fire("hook:null_safe_dummy")
        # shares written as CASE ... ELSE 0 END must not count non-respondents
        for key in ("measure", "x", "y"):
            fixed = self._null_safe_share(plan.get(key))
            if fixed != plan.get(key):
                plan[key] = fixed
                plan["_null_safe_share"] = True
                self._fire("hook:null_safe_share")
        if measures_all:
            measures_all = [(lab, self._null_safe_share(e)) for lab, e in measures_all]
            if any(e != o for (_, e), (_, o) in zip(measures_all, self._measure_list(plan))):
                plan["_null_safe_share"] = True
                self._fire("hook:null_safe_share")
        multi = len(measures_all) > 1
        benchmarks = self._benchmarks(plan)
        per_cycle: dict[str, pd.DataFrame] = {}
        bench_basis: dict = {}          # cycle -> measure label -> avg label -> (basis, members)
        cycle_context: dict = {}        # cycle -> (cplan, table, where)  for the share link error
        for cycle in cycles:
            tbl = f"{instrument}_{cycle}"
            cplan = {**plan, **overrides.get(cycle, {})}
            if multi:
                cplan["_skip_measures"] = True      # measures are checked one by one below
            self._check_columns(cplan, tbl)
            cwhere = cplan.get("where") or None
            if cycle in region_codes:
                codes = region_codes[cycle]["codes"]
                clause = ("CNT IN (" + ", ".join(f"'{c}'" for c in codes) + ")"
                          if codes else "FALSE")
                cwhere = f"({cwhere}) AND {clause}" if cwhere else clause
            parts = []
            for label, expr in (measures_all if multi else [(None, None)]):
                if expr is not None and self._unknown_columns(expr, tbl):
                    # e.g. GLOBMIND (2018 only) in a 2018-2025 profile: computed
                    # where it exists, stated where it does not
                    plan.setdefault("_unavailable", {}).setdefault(cycle, []).append(label)
                    continue
                mplan = {**cplan, "measure": expr} if expr is not None else cplan
                cby = by_per_cycle.get(cycle, by)
                res = self._run_template(template, mplan, tbl, cby, cwhere)
                if cby != by:
                    res = res.rename(columns=dict(zip(cby, by)))
                    for src, dst in zip(cby, by):
                        # MALE (1 = male, 0 = female) rows recoded to the main
                        # plan's ST004D01T (2 = male, 1 = female), by label
                        if src != dst and src in self.GENDER_CODES and dst in self.GENDER_CODES \
                                and dst in res.columns:
                            inv = {v: k for k, v in self.GENDER_CODES[dst].items()}
                            res[dst] = res[dst].map(
                                lambda v: float(inv[self.GENDER_CODES[src][str(int(v))]])
                                if not pd.isna(v) and str(int(v)) in self.GENDER_CODES[src]
                                and self.GENDER_CODES[src][str(int(v))] in inv else v)
                for who, why in (getattr(res, "attrs", {}) or {}).get("skipped", []):
                    plan.setdefault("_regression_skipped", []).append((f"{who} (PISA {cycle})", why))
                res = self._suppress_small_cells(res, cycle, plan)
                res, dropped_groups = self._drop_null_groups(res, by)
                for col, n in dropped_groups.items():
                    plan.setdefault("_null_groups", {}).setdefault(col, {})[cycle] = n
                res = self._with_stratum_rows(res, template, mplan, tbl, by, cwhere, cycle, plan)
                if benchmarks and "CNT" in by:
                    # "The OECD / EU average" means ALL members of the group — when
                    # the question is filtered to a few countries, the average must
                    # NOT be taken over just the members inside the filter. The
                    # rows are built after every cycle has run (constant membership
                    # across a trend), so only the basis is kept here.
                    for bench in benchmarks:
                        resolved = self._average_members(bench, cycle, mplan.get("measure"), plan)
                        if resolved is None:
                            plan.setdefault("_unknown_benchmarks", []).append(str(bench))
                            continue
                        label_b, members = resolved
                        if not members:
                            continue
                        clause = "CNT IN (" + ", ".join(f"'{c}'" for c in sorted(members)) + ")"
                        basis = res if cwhere is None else \
                            self._suppress_small_cells(
                                self._run_template(template, mplan, tbl, by, clause), cycle, plan)
                        bench_basis.setdefault(cycle, {}).setdefault(label, {})[label_b] = (basis, members)
                cycle_context[cycle] = (cplan, tbl, cwhere)
                if label is not None:
                    res.insert(1 if "CNT" in res.columns else 0, "measure", label)
                parts.append(res)
            if not parts:
                parts.append(pd.DataFrame(columns=list(by) + ["measure", "estimate", "se", "n_pv"]))
            per_cycle[cycle] = parts[0] if len(parts) == 1 else \
                pd.concat(parts, ignore_index=True)
        # A cycle with no rows for this filter (an economy that had not yet
        # joined PISA, a construct not administered) is reported, not run.
        empty_cycles = [c for c in cycles if per_cycle[c].empty]
        if empty_cycles:
            plan["_empty_cycles"] = empty_cycles
            per_cycle = {c: r for c, r in per_cycle.items() if not r.empty}
            cycles = [c for c in cycles if c not in empty_cycles]
            if not cycles:
                raise ValueError(
                    "no data match this question in any requested cycle — "
                    f"filter {where or 'none'}; check the economy's participation "
                    "(for example El Salvador joined PISA in 2022).")
        self._append_benchmark_rows(per_cycle, bench_basis, cycles, multi, plan)

        if template == "gap" and "contrast" in per_cycle[cycles[0]].columns \
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(plan.get("group_col") or "")):
            # "ST004D01T: 2 - 1" reads as code arithmetic; say who minus whom
            first_tbl = f"{instrument}_{cycles[0]}"
            a = self._category_label(str(plan["group_col"]), plan["minuend"], first_tbl)
            b = self._category_label(str(plan["group_col"]), plan["subtrahend"], first_tbl)
            for res in per_cycle.values():
                res["contrast"] = f"{a} minus {b}"
        elif overrides and "contrast" in per_cycle[cycles[0]].columns:
            # an overridden group variable relabels the contrast (e.g. "MALE:
            # 1 - 0" vs "ST004D01T: 2 - 1"); align on the first cycle's label
            # so the cycles merge — the override itself is stated in provenance
            label = per_cycle[cycles[0]]["contrast"].iloc[0]
            for res in per_cycle.values():
                res["contrast"] = label

        extra_keys = {"gap": ["contrast"], "quartile_gap": ["contrast"],
                      "quartile_means": ["quarter"],
                      "percentiles": ["percentile"],
                      "percentile_spread": ["contrast"],
                      "crosstab": ["row", "col", "row_label", "col_label"],
                      "regression": ["term"]}.get(template, [])
        if template == "crosstab":
            extra_keys = [k for k in extra_keys
                          if k in per_cycle[cycles[0]].columns]
        if multi:
            extra_keys = extra_keys + ["measure"]
        if len(cycles) >= 2:
            plan["_trend_cycles"] = (cycles[0], cycles[-1])
        run_measures = measures_all if multi else [(None, plan.get("measure"))]
        if len(cycles) >= 2 and multi:
            # one trend per measure: each domain has its own link error
            pieces, les, modes = [], {}, {}
            for label, expr in measures_all:
                sub = {c: df[df["measure"] == label] for c, df in per_cycle.items()}
                le, modes[label] = self._link_error_for(template, expr, cycles, by, cycle_context)
                les[label] = le if not isinstance(le, pd.DataFrame) else "per-economy share link error"
                pieces.append(trend(sub, by=[c for c in by] + extra_keys, link_error=le))
            plan["_link_errors"] = les
            plan["_link_error_modes"] = modes
            table = pd.concat(pieces, ignore_index=True)
        elif len(cycles) >= 2:
            le, mode = self._link_error_for(template, plan.get("measure"), cycles, by, cycle_context)
            plan["_link_error"] = le if not isinstance(le, pd.DataFrame) else "per-economy share link error"
            plan["_link_error_modes"] = {None: mode}
            table = trend(per_cycle, by=[c for c in by] + extra_keys, link_error=le)
        else:
            table = per_cycle[cycles[0]].assign(cycle=cycles[0])
        if len(cycles) >= 2 and template in ("weighted_mean", "gap", "quartile_means",
                                             "quartile_gap", "percentiles", "percentile_spread"):
            self._blank_non_comparable_changes(table, plan, run_measures, cycles, overrides)

        if plan.get("_quarter_filter") and "quarter" in table.columns:
            keep = table["quarter"].astype(float).isin([float(q) for q in plan["_quarter_filter"]])
            table = table[keep].drop(columns=["quarter"]).reset_index(drop=True)
        sort_by = plan.get("sort_by")
        if sort_by:
            candidates = [sort_by, f"estimate_{cycles[-1]}" if sort_by == "estimate" else sort_by]
            col = next((c for c in candidates if c in table.columns), None)
            if col:
                table = table.sort_values(col, ascending=not plan.get("sort_desc", True))
        if sort_by and "CNT" in by and len(table) > 1 and not multi:
            # A ranking question: number the rows so "where is Kosovo ranked"
            # has a literal answer in the table, not a position to be counted.
            # Rank always counts from the TOP of the sort measure (highest
            # score or largest change = 1), whichever way the table is shown.
            rank_col = next((c for c in [sort_by, f"estimate_{cycles[-1]}"
                                         if sort_by == "estimate" else sort_by]
                             if c in table.columns), None)
            if rank_col:
                table = table.reset_index(drop=True)
                economies = ~table["CNT"].astype(str).str.endswith(" avg")   # averages are not ranked
                ranks = table.loc[economies, rank_col].rank(ascending=False, method="min")
                table.insert(0, "rank", ranks.reindex(table.index).astype("Int64"))
        if plan.get("top_n"):
            # a benchmark average asked for alongside a top-N stays in the
            # table even when it sits below rank N
            if "CNT" in table.columns:
                is_avg = table["CNT"].astype(str).str.endswith(" avg")
                table = pd.concat([table[~is_avg].head(int(plan["top_n"])), table[is_avg]])
            else:
                table = table.head(int(plan["top_n"]))
        table = table.reset_index(drop=True)

        prov = self._provenance(plan, tables)
        prov["notes"].extend(self._missing_estimate_notes(table, plan))
        is_score = any(link_errors.domain_of(e) for _, e in (measures_all or [(None, plan.get("measure"))]))
        codes_in_table = set(table["CNT"].astype(str)) if "CNT" in table.columns else set()
        if "2018" in cycles and "VNM" in codes_in_table and is_score:
            prov["notes"].append(self.VNM_2018_NOTE)
        if len(cycles) >= 2 and "2025" in cycles and is_score and (codes_in_table & self.MODE_CHANGE_2025):
            who = self._names(sorted(codes_in_table & self.MODE_CHANGE_2025))
            prov["notes"].append(
                f"{who} administered PISA 2025 on computer after paper-based tests in every "
                "earlier cycle. The OECD's PISA 2025 Technical Report (Data Adjudication) states "
                "that the uncertainty around trend comparisons for Guatemala, Paraguay and Viet "
                "Nam is not limited to what the link errors capture and recommends caution in "
                "reporting and interpreting their trends; the change shown here carries only the "
                "sampling and link-error uncertainty.")
        if len(cycles) >= 2 and is_score:
            # any other change of test mode between the first and last cycle
            # shown (Jordan, Lebanon, Moldova, North Macedonia, Romania, Saudi
            # Arabia, Ukraine, Argentina: paper in 2018, computer in 2022)
            plain = [c for c in codes_in_table if re.fullmatch(r"[A-Z]{3}", c)]
            changed = [x for x in self._mode_changes(plain, cycles) if x[0] not in self.MODE_CHANGE_2025]
            if changed:
                who = "; ".join(f"{self._names([c])}: {a} in {y1}, {b} in {y2}" for c, y1, a, y2, b in changed[:8])
                prov["notes"].append(
                    "Test mode changed between the cycles shown (from ADMINMODE in the student "
                    f"files): {who}" + (f"; and {len(changed) - 8} more" if len(changed) > 8 else "")
                    + ". The OECD reports these trends on the linked scale with the link error; the "
                    "effect of the change of mode is not quantified in the public files, so read "
                    "the change with that caution.")
                self._fire("hook:mode_change_note")
        return table, prov

    # Economies whose 2025 test moved from paper to computer (PISA 2025
    # Technical Report, Data Adjudication): trends carry extra, unquantified
    # uncertainty.
    MODE_CHANGE_2025 = {"VNM", "GTM", "PRY"}

    VNM_2018_NOTE = (
        "Viet Nam's 2018 scores come from the plausible values the OECD released "
        "separately after the main database. The OECD states that Viet Nam's PISA 2018 "
        "data \"did not meet the PISA technical standards but were accepted as largely "
        "comparable\" and, in PISA 2018 Results (Volume I), kept Viet Nam out of the "
        "tables that compare performance across countries or over time because full "
        "international comparability could not be assured (Annexes A2, A4 and A6). "
        "Treat its 2018 values, and any 2018-based change, with that caution.")

    AVG_SHORT = {"European Union": "EU", "Latin America and the Caribbean": "LatAm",
                 "Nordic countries": "Nordic", "Middle East and North Africa": "MENA",
                 "Sub-Saharan Africa": "SSA", "Southeast Asia": "SE Asia",
                 "East Asia": "E Asia", "Central Asia": "C Asia", "North America": "N America",
                 "South America": "S America", "Central America": "C America",
                 "Baltic states": "Baltic", "Western Balkans": "W Balkans"}

    def _benchmarks(self, plan: dict) -> list:
        """Benchmark groups a plan asks for: "OECD", region names, or an
        explicit list of codes — deduplicated, in plan order."""
        out = []
        if plan.get("include_oecd_average"):
            out.append("OECD")
        raw = plan.get("include_average_of")
        if isinstance(raw, (str, dict)):
            raw = [raw]
        if isinstance(raw, list):
            labels = plan.setdefault("_group_labels", {})
            flat = []
            for x in raw:
                # a NAMED ad-hoc group: {"label": "Mercosur", "members": [...]}
                # or a bare list of codes — one benchmark row each, so two
                # blocks asked for together never merge into one group
                if isinstance(x, dict):
                    members = x.get("members") or x.get("codes") or []
                    codes_x = tuple(sorted({str(c).upper() for c in members
                                            if isinstance(c, str) and re.fullmatch(r"[A-Za-z]{3}", c)}))
                    if len(codes_x) >= 2:
                        label = str(x.get("label") or "").strip()
                        if label and len(label) <= 28:      # a name, not a member list
                            labels[codes_x] = label
                        out.append(codes_x)
                elif isinstance(x, list):
                    codes_x = tuple(sorted({str(c).upper() for c in x
                                            if isinstance(c, str) and re.fullmatch(r"[A-Za-z]{3}", c)}))
                    if len(codes_x) >= 2:
                        out.append(codes_x)
                elif isinstance(x, str):
                    flat.append(x)
            codes = [str(x).upper() for x in flat if re.fullmatch(r"[A-Za-z]{3}", str(x))]
            names = [x for x in flat if not re.fullmatch(r"[A-Za-z]{3}", x)]
            if len(codes) >= 2 and len(codes) == len(flat):
                out.append(tuple(sorted(set(codes))))        # ad-hoc group
            else:
                out.extend(names)
                if len(codes) >= 2:
                    out.append(tuple(sorted(set(codes))))
        seen, uniq = set(), []
        for b in out:
            key = b if isinstance(b, tuple) else str(b).strip().lower()
            if key not in seen:
                seen.add(key); uniq.append(b)
        return uniq

    # Members the OECD left out of a published average for one cycle and
    # domain (Spain's 2018 reading results were withheld from the
    # comparison tables; Annex A9).
    AVERAGE_EXCLUSIONS = {("OECD avg", "2018", "READ"): {"ESP"}}
    ALL_PARTICIPANTS = {"all participants", "all economies", "all countries", "world",
                        "world average", "global", "global average", "international",
                        "international average", "everyone"}

    def _average_members(self, bench, cycle: str, measure: str | None = None,
                         plan: dict | None = None) -> tuple[str, set[str]] | None:
        """(row label, member codes present in `cycle`) for a benchmark, or
        None when the name is not a known group."""
        present = self.present.get(cycle, set())
        if isinstance(bench, tuple):
            label = ((plan or {}).get("_group_labels") or {}).get(bench) or "Group"
            return f"{label} avg", set(bench) & present
        key = str(bench).strip().lower()
        if key == "oecd":
            members = self._oecd_codes(cycle)
            domain = link_errors.domain_of(measure)
            excluded = self.AVERAGE_EXCLUSIONS.get(("OECD avg", cycle, domain or ""), set()) & members
            if excluded:
                members = members - excluded
                if plan is not None:
                    plan.setdefault("_avg_exclusions", {}).setdefault(cycle, {})["OECD avg"] = sorted(excluded)
            return "OECD avg", members
        if key in self.ALL_PARTICIPANTS:
            return "All-participant avg", set(present)
        canon = regions.canonical(str(bench))
        if canon is None:
            return None
        codes = set(regions.expand([canon], {cycle: present})[cycle]["codes"])
        return f"{self.AVG_SHORT.get(canon, canon)} avg", codes

    def _append_benchmark_rows(self, per_cycle: dict, bench_basis: dict, cycles: list[str],
                               multi: bool, plan: dict) -> None:
        """Build the "<group> avg" rows. In a trend the membership is held
        CONSTANT: only members with an estimate in every computed cycle enter
        the average, so a change is never the arithmetic of a different set
        of countries in each year (the OECD's convention; e.g. its 2018→2022
        mathematics average change is computed on a common membership)."""
        if not bench_basis:
            return
        labels_m = sorted({lab for d in bench_basis.values() for lab in d})
        for mlabel in labels_m:
            avg_labels = sorted({lb for c in cycles for lb in bench_basis.get(c, {}).get(mlabel, {})})
            for label_b in avg_labels:
                have = {c: bench_basis[c][mlabel][label_b] for c in cycles
                        if label_b in bench_basis.get(c, {}).get(mlabel, {})}
                if not have:
                    continue
                with_est = {}
                for c, (basis, members) in have.items():
                    ok = basis[basis["CNT"].isin(members) & basis["estimate"].notna()]
                    with_est[c] = set(ok["CNT"].astype(str))
                if len(have) >= 2:
                    common = set.intersection(*with_est.values())
                    dropped = sorted(set.union(*with_est.values()) - common)
                    if dropped:
                        plan.setdefault("_benchmark_dropped", {})[label_b] = dropped
                else:
                    common = next(iter(with_est.values()))
                for c, (basis, members) in have.items():
                    avg = self._group_average_rows(basis, common, label_b)
                    if avg is None:
                        continue
                    if multi:
                        # the basis may be the result frame itself, which
                        # already carries the measure column
                        if "measure" in avg.columns:
                            avg["measure"] = mlabel
                        else:
                            avg.insert(1, "measure", mlabel)
                    per_cycle[c] = pd.concat([per_cycle[c], avg], ignore_index=True)
                    plan.setdefault("_benchmark_members", {}).setdefault(c, {})[label_b] = sorted(common)

    def _group_average_rows(self, res: pd.DataFrame, members: set[str],
                            label: str) -> pd.DataFrame | None:
        """Official OECD-average convention, for any group: the UNWEIGHTED mean
        of member economies' estimates (each economy counts equally — not a
        pooled student-weighted mean, which would overweight populous
        countries). Economies are independent samples, so SE = sqrt(sum
        SE_i^2) / N, with N the members that have an estimate."""
        sub = res[res["CNT"].isin(members) & res["estimate"].notna()]
        id_cols = [c for c in res.columns
                   if c not in ("CNT", "estimate", "se", "n_pv")]
        if id_cols:
            # a missing grouping value is not a group (no "average of the
            # students with no gender recorded")
            sub = sub.dropna(subset=[c for c in id_cols if c in sub.columns])
        if sub.empty:
            return None

        def agg(group: pd.DataFrame) -> pd.Series:
            n = int(group["estimate"].notna().sum())
            return pd.Series({
                "estimate": group["estimate"].mean(),
                "se": float(np.sqrt((group["se"] ** 2).sum()) / n) if n else float("nan"),
                "n_pv": int(group["n_pv"].iloc[0]),
            })

        if id_cols:
            avg = (sub.groupby(id_cols, dropna=False, observed=True)
                   .apply(agg, include_groups=False).reset_index())
        else:
            avg = agg(sub).to_frame().T
        avg["n_pv"] = avg["n_pv"].astype(int)
        avg.insert(0, "CNT", label)
        return avg

    # Statistics that are levels on the score scale carry the OECD link error
    # across cycles; differences between groups measured in the same cycle
    # (gaps, quartile gaps, spreads, regression slopes) do not — the linking
    # shifts both groups alike and cancels (PISA 2022 Results Vol. I, Annex A7).
    LEVEL_TEMPLATES = ("weighted_mean", "percentiles", "quartile_means")

    def _link_error_for(self, template: str, expr: str | None, cycles: list[str],
                        by: tuple, cycle_context: dict):
        """(link error, mode) for a change of `expr` from the first to the last
        computed cycle: a float (mean score), a per-row DataFrame (a
        proficiency-threshold share, derived from the score link error by
        shifting every plausible value by ±LE, the OECD's slope approach),
        or None with the reason ("cancels", "percentile-approx", "none")."""
        first, last = cycles[0], cycles[-1]
        if template not in self.LEVEL_TEMPLATES:
            return None, "cancels"
        le = link_errors.link_error(first, last, expr)
        if le:
            return le, ("percentile-approx" if template != "weighted_mean" else "mean")
        domain = link_errors.domain_of(expr)
        core = self._unwrap_null_safe(str(expr or ""))
        if template == "weighted_mean" and domain and self.SHARE_CASE.match(core) \
                and last in cycle_context:
            score_le = link_errors.LINK_ERRORS.get((first, last, domain))
            if not score_le:
                return None, "none"
            cplan, tbl, cwhere = cycle_context[last]
            shifted = []
            for sign in ("+", "-"):
                m = re.sub(r"PV\{pv\}(MATH|READ|SCIE)",
                           lambda mm: f"(PV{{pv}}{mm.group(1)} {sign} {score_le})", str(expr))
                res = self._run_template("weighted_mean", {**cplan, "measure": m}, tbl, by, cwhere)
                shifted.append(res.set_index(list(by))["estimate"] if by else res["estimate"])
            diff = (shifted[0] - shifted[1]).abs() / 2.0
            if by:
                frame = diff.reset_index().rename(columns={"estimate": "link_error"})
            else:
                frame = pd.DataFrame({"link_error": [float(diff.iloc[0])]})
            frame["link_error"] = frame["link_error"].fillna(0.0)
            return frame, "share"
        return None, "none"

    @staticmethod
    def _unwrap_null_safe(expr: str) -> str:
        m = re.match(r"^\s*CASE WHEN \(.*?\) IS NULL THEN NULL ELSE \((.*)\) END\s*$", expr, re.S)
        return m.group(1) if m else expr

    def _run_template(self, template: str, plan: dict, tbl: str,
                      by: tuple, where: str | None) -> pd.DataFrame:
        """One template invocation against one cycle table."""
        if template == "weighted_mean":
            return weighted_mean(self.con, tbl, plan["measure"], by=by, where=where)
        if template == "weighted_proportion" and not plan.get("variable") \
                and plan.get("measure"):
            # The planner sometimes files a threshold share ("% below Level
            # 2" = CASE WHEN … THEN 100 ELSE 0) under weighted_proportion;
            # that is exactly weighted_mean of the 0/100 measure.
            return weighted_mean(self.con, tbl, plan["measure"], by=by, where=where)
        if template == "weighted_proportion":
            res = weighted_proportion(
                self.con, tbl, plan["variable"], plan["value"], by=by,
                where=where, valid_values=plan.get("valid_values"))
            # Say WHICH category the percentage is — "MALE = 0 (Female/Other)" —
            # so a summary can never invert it (49% girls is not 49% boys).
            res.insert(1 if "CNT" in res.columns else 0, "category",
                       self._category_label(plan["variable"], plan["value"], tbl))
            return res
        if template == "gap":
            group_col = str(plan["group_col"])
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group_col):
                return gap(self.con, tbl, plan["measure"], group_col,
                           plan["minuend"], plan["subtrahend"], by=by, where=where)
            # A derived two-group split (immigrant vs non-immigrant, …):
            # a CASE expression yielding the two codes compared.
            self._check_fragment(group_col)
            if not re.match(r"^\s*CASE\b", group_col, re.IGNORECASE):
                raise ValueError("group_col must be a column name or a CASE "
                                 f"expression, not {group_col!r}")
            return gap(self.con, tbl, plan["measure"], group_col,
                       plan["minuend"], plan["subtrahend"], by=by, where=where,
                       group_expr=group_col,
                       group_label=str(plan.get("group_label") or "derived group"))
        if template == "quartile_means":
            self._check_fragment(plan["quart_variable"])
            return quartile_means(self.con, tbl, plan["measure"],
                                  plan["quart_variable"], by=by, where=where)
        if template == "quartile_gap":
            self._check_fragment(plan["quart_variable"])
            return quartile_gap(self.con, tbl, plan["measure"],
                                plan["quart_variable"], by=by, where=where)
        if template == "correlation":
            self._check_fragment(plan["x"])
            self._check_fragment(plan["y"])
            return correlation(self.con, tbl, plan["x"], plan["y"],
                               by=by, where=where)
        if template == "percentiles":
            ps = tuple(int(p) for p in plan.get("ps") or (10, 25, 50, 75, 90))
            return percentiles(self.con, tbl, plan["measure"], by=by,
                               where=where, ps=ps)
        if template == "percentile_spread":
            return percentile_spread(
                self.con, tbl, plan["measure"], by=by, where=where,
                upper=int(plan.get("upper") or 90),
                lower=int(plan.get("lower") or 10))
        if template == "crosstab":
            self._check_identifier(plan["row_var"])
            self._check_identifier(plan["col_var"])
            res = crosstab(self.con, tbl, plan["row_var"], plan["col_var"],
                           by=by, where=where,
                           valid_rows=plan.get("valid_rows"),
                           valid_cols=plan.get("valid_cols"))
            return self._label_crosstab(res, plan, tbl)
        if template == "regression":
            for x in plan["predictors"]:
                self._check_fragment(x)
            return regression(self.con, tbl, plan["measure"],
                              plan["predictors"], by=by, where=where,
                              names=plan.get("predictor_names"))
        raise ValueError(f"unknown template {template!r}")

    @staticmethod
    def _label_crosstab(res: pd.DataFrame, plan: dict, table: str) -> pd.DataFrame:
        """Attach human-readable value labels to crosstab codes when the
        catalog has them."""
        for code_col, var_key, label_col in (("row", "row_var", "row_label"),
                                             ("col", "col_var", "col_label")):
            desc = catalog.describe(plan[var_key])
            desc = desc[desc.table_name.isin(Agent.underlying_tables(table))]
            if desc.empty or not desc.iloc[0].value_labels:
                continue
            labels = json.loads(desc.iloc[0].value_labels)
            mapped = res[code_col].map(
                lambda v: labels.get(str(int(v)) if isinstance(v, float)
                                     and float(v).is_integer() else str(v)))
            if mapped.notna().any():
                res.insert(res.columns.get_loc(code_col) + 1, label_col, mapped)
        return res

    # The PUF terms of use forbid distributing the dataset: raw SQL may return
    # aggregates only, never student or school records.
    AGGREGATE_SQL = re.compile(r"\b(count|sum|avg|min|max|median|quantile\w*|stddev\w*|var\w*|"
                               r"corr|group by)\b", re.IGNORECASE)
    IDENTIFIER_COLS = re.compile(r"\b(CNTSTUID|CNTSCHID|CNTTCHID|STUID|SCHID)\b", re.IGNORECASE)

    def _run_raw_sql(self, sql: str) -> pd.DataFrame:
        if not sql:
            raise ValueError("raw_sql plan without sql")
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise ValueError("only a single SQL statement is allowed")
        if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
            raise ValueError("only SELECT queries are allowed")
        if FORBIDDEN_SQL.search(stripped):
            raise ValueError("query uses a forbidden statement type")
        if not self.AGGREGATE_SQL.search(stripped) or re.search(r"\bselect\s+\*", stripped, re.I):
            raise ValueError("raw SQL must aggregate (COUNT/SUM/AVG/... with GROUP BY): "
                             "student- and school-level records are never returned "
                             "(OECD PISA public-use file terms of use)")
        if self.IDENTIFIER_COLS.search(stripped):
            raise ValueError("student and school identifiers cannot be selected or grouped "
                             "on (OECD PISA public-use file terms of use)")
        return self.con.sql(stripped).df().head(MAX_RESULT_ROWS)

    @staticmethod
    def _check_fragment(fragment: str | None) -> None:
        if fragment and (";" in fragment or FORBIDDEN_SQL.search(fragment)):
            raise ValueError(f"unsafe SQL fragment: {fragment!r}")

    @staticmethod
    def _check_identifier(name: str) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid column name: {name!r}")

    # ---------- provenance ----------

    def _referenced_variables(self, plan: dict, tables: list[str]) -> list[dict]:
        fields = ("measure", "measures", "variable", "group_col", "where", "by",
                  "quart_variable", "x", "y", "row_var", "col_var", "predictors")
        parts = [plan] + [ov for ov in (plan.get("cycle_overrides") or {}).values()
                          if isinstance(ov, dict)]
        text = " ".join(str(part.get(k) or "") for part in parts for k in fields)
        text = text.replace("{pv}", "1")  # PV{pv}MATH -> PV1MATH (stands for all 10)
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        tokens |= {"W_FSTUWT"}
        out, seen = [], set()
        physical = [t for tbl in tables for t in self.underlying_tables(tbl)]
        for token in sorted(tokens):
            desc = catalog.describe(token)
            desc = desc[desc.table_name.isin(physical)] if physical else desc
            for _, r in desc.iterrows():
                key = (r.variable, r.table_name)
                if key not in seen:
                    seen.add(key)
                    out.append({"variable": r.variable, "table": r.table_name,
                                "label": r.label})
        return out

    def _provenance(self, plan: dict, tables: list[str], raw: bool = False) -> dict:
        template = plan.get("template")
        where = plan.get("where")
        methods = {
            "weighted_mean": "Weighted mean (final student weight W_FSTUWT). "
                             "SE: Fay's BRR, k=0.5, 80 replicate weights.",
            "weighted_proportion": "Weighted percentage of students with a valid (non-missing) "
                                   "response (W_FSTUWT). SE: Fay's BRR, k=0.5, 80 replicates.",
            "gap": "Group difference computed replicate-wise (correct covariance). "
                   "Weighted by W_FSTUWT; SE: Fay's BRR, k=0.5, 80 replicates.",
            "quartile_means": "Weighted mean per weighted quarter of "
                              f"{plan.get('quart_variable')} (quartile cut points computed "
                              "with W_FSTUWT within each economy, as the OECD does; 1=bottom, "
                              "4=top). SE: Fay's BRR, k=0.5, 80 replicates.",
            "quartile_gap": "Top minus bottom weighted quarter of "
                            f"{plan.get('quart_variable')} (quartile cut points computed with "
                            "W_FSTUWT within each economy), differenced replicate-wise. "
                            "SE: Fay's BRR, k=0.5, 80 replicates.",
            "correlation": "Weighted Pearson correlation (W_FSTUWT). "
                           "SE: Fay's BRR, k=0.5, 80 replicates.",
            "percentiles": "Weighted empirical percentiles (W_FSTUWT; the value at which "
                           "the cumulative weight reaches the percentile, per plausible "
                           "value — the convention of the OECD's published Stata code, "
                           "_pctile [aw=w_fstuwt]). SE: Fay's BRR, k=0.5, 80 replicates.",
            "percentile_spread": "Difference of weighted percentiles (within-group "
                                 "dispersion; OECD _pctile convention), differenced "
                                 "replicate-wise. SE: Fay's BRR, k=0.5, 80 replicates.",
            "crosstab": "Weighted two-way table: row percentages of valid "
                        "respondents (W_FSTUWT), each cell with its own "
                        "Fay-BRR SE (k=0.5, 80 replicates).",
            "regression": "Weighted least-squares regression (W_FSTUWT), "
                          "listwise deletion; plausible values only in the outcome; no R² "
                          "reported. SE: Fay's BRR, k=0.5, 80 replicates per coefficient. "
                          "Association, not causation.",
            "raw_sql": "Direct SQL — NO automatic weighting/PV/BRR treatment; "
                       "results are not population estimates unless the query weights them.",
        }
        if template == "weighted_proportion" and not plan.get("variable") \
                and plan.get("measure"):
            template = "weighted_mean"      # threshold share ran as a 0/100 mean
        method = methods.get(template, "")
        pv_fields = " ".join(str(plan.get(k) or "") for k in ("measure", "measures", "x", "y"))
        if "{pv}" in pv_fields:
            method += (" Point estimate averaged over 10 plausible values; the SE adds the "
                       "between-PV variance with the (1 + 1/M) factor (Rubin's rules).")
        notes = []
        members_by_cycle = plan.get("_benchmark_members") or {}
        if members_by_cycle:
            labels = sorted({lab for d in members_by_cycle.values() for lab in d})
            detail = "; ".join(
                f"{lab}: " + ", ".join(
                    f"PISA {c} = {len(d[lab])} economies" for c, d in sorted(members_by_cycle.items()) if lab in d)
                for lab in labels)
            notes.append(f"The \"{'\", \"'.join(labels)}\" row(s) follow the official OECD-average "
                         "convention: the unweighted mean of the member economies' estimates "
                         f"(SE = sqrt(sum of member SEs squared) / N, N = members with an "
                         "estimate), over all members, not only those in the question's "
                         f"filter ({detail}).")
            for lab, dropped in (plan.get("_benchmark_dropped") or {}).items():
                notes.append(f"To keep the {lab} comparable across cycles it is computed on the "
                             f"members with an estimate in every cycle shown; excluded: "
                             f"{self._names(dropped)} (not in every cycle, or no estimate). The "
                             "OECD's published trend averages use a constant membership in the "
                             "same way.")
            for cycle_x, d in sorted((plan.get("_avg_exclusions") or {}).items()):
                for lab, codes in d.items():
                    notes.append(f"PISA {cycle_x}: {self._names(codes)} is left out of the {lab} "
                                 "for this domain, as in the OECD's published tables (its reading "
                                 "results were withheld from the 2018 comparison tables, Annex A9).")
        contrast_case = any(
            isinstance(v, str) and re.match(r"^\s*CASE\b", v, re.I) and not re.search(r"\bELSE\b", v, re.I)
            for v in [plan.get("group_col"), plan.get("measure")] +
            [ov.get("group_col") for ov in (plan.get("cycle_overrides") or {}).values() if isinstance(ov, dict)])
        for cycle_x, rows in sorted((plan.get("_low_coverage") or {}).items()):
            shown = "; ".join(f"{k}: {pct}% of the weighted population has no value" for k, pct in rows[:6])
            if contrast_case:
                notes.append(f"PISA {cycle_x}: a large part of the group is outside the contrast "
                             f"({shown}{'; …' if len(rows) > 6 else ''}) — students in categories the "
                             "contrast does not name (for example towns in a rural-vs-city contrast) "
                             "plus non-respondents; the estimate describes the compared categories only.")
            else:
                notes.append(f"PISA {cycle_x}: the variable is missing for a large part of the group "
                             f"({shown}{'; …' if len(rows) > 6 else ''}); the estimate describes the "
                             "students who have a value, not the whole population.")
        if plan.get("_null_safe_dummy"):
            notes.append("Categorical predictors written as CASE dummies keep students with no value "
                         "on the underlying variable as missing (dropped listwise), not as members of "
                         "the reference category.")
        if plan.get("_same_wording"):
            notes.append("The item code differs between cycles but the item wording and response codes "
                         "are the same, so the share is compared across cycles (as the OECD does for "
                         "retained items); the stem wording changed slightly (e.g. \"Agree:\" to "
                         "\"Agree/disagree:\"), which is a caveat, not a break in the series. Sampling SE "
                         "only — no link error applies to a response-code share.")
        if plan.get("_trend_index"):
            for var, means in plan["_trend_index"].items():
                shown = ", ".join(f"{c}: {m:+.2f}" for c, m in sorted(means.items()))
                notes.append(f"{var} is treated as a trend scale: its OECD-average is not 0 in every "
                             f"cycle ({shown}), so the OECD did not re-standardize it in each cycle but "
                             "kept it on the earlier cycle's scale. The change shown carries sampling "
                             "error only — the OECD publishes no link error for questionnaire indices.")
        if plan.get("_measure_cycles_dropped"):
            notes.append("The measure does not exist in PISA " + ", ".join(plan["_measure_cycles_dropped"])
                         + " (not assessed in that cycle); only the cycles that have it are shown.")
        if plan.get("_school_file_to_students"):
            notes.append("School-questionnaire variables are reported for the students in those "
                         "schools (students joined to their school, weighted by the final student "
                         "weight), the OECD's convention for school characteristics — not as a "
                         "share of schools.")
        if plan.get("_auto_instrument"):
            view, vars_ = plan["_auto_instrument"]
            notes.append(f"{', '.join(vars_)} live(s) in the "
                         + ("school questionnaire file" if view == "stu_sch" else "creative-thinking cognitive file")
                         + "; the analysis runs on students joined to it, with the students' official weights.")
        for who, why in (plan.get("_regression_skipped") or [])[:6]:
            notes.append(f"No regression estimate for {who}: {why}.")
        if not members_by_cycle and plan.get("include_oecd_average"):
            notes.append('The "OECD avg" row follows the official convention: '
                         "the unweighted mean of OECD member countries' "
                         "estimates (SE = sqrt(sum of country SEs squared) / N).")
        for col, per_cycle_n in sorted((plan.get("_null_groups") or {}).items()):
            desc = catalog.describe(col)
            label = desc.iloc[-1].label if not desc.empty else col
            cycles_txt = ", ".join(f"PISA {c}" for c in sorted(per_cycle_n))
            notes.append(f"Students with no value on {label} ({col}) — no response, or no "
                         f"school questionnaire for their school — are excluded from the "
                         f"groups in {cycles_txt}, following OECD practice; there is no "
                         "\"overall\" row in this table.")
        qci = self._qci_note(str(plan.get("_question") or ""), plan)
        if qci:
            notes.append(qci)
        if plan.get("_ranking_kept"):
            notes.append("The full ranking is shown (the question asked for a ranking as "
                         "well as the top economy).")
        for bench in plan.get("_unknown_benchmarks") or []:
            notes.append(f"NOT DONE: \"{bench}\" is not a group this app knows (OECD or a "
                         "region from its fixed list), so no average row was added for it.")
        if plan.get("substitution_note") and not plan.get("_school_type_switched"):
            notes.append(f"VARIABLE SUBSTITUTION: {plan['substitution_note']}")
        for line in plan.get("_standardized") or []:
            notes.append(f"STANDARD VARIABLE: {line}")
        for cycle, k in sorted((plan.get("_suppressed") or {}).items()):
            if isinstance(k, dict):
                bits = []
                if k.get("students"):
                    bits.append(f"{k['students']} on fewer than {self.MIN_STUDENTS} students")
                if k.get("schools"):
                    bits.append(f"{k['schools']} on fewer than {self.MIN_SCHOOLS} schools")
                what = " and ".join(bits)
            else:
                what = f"{k} on fewer than {self.MIN_STUDENTS} students"
            notes.append(f"PISA {cycle}: estimate(s) suppressed — {what} — the OECD's minimum "
                         "for reporting (30 students and 5 schools), which also protects "
                         "individual schools' responses; those cells are blank, not zero.")
        all_wheres = " ".join([str(where or ""), str(plan.get("group_col") or "")] +
                              [str(ov.get("where") or "") + " " + str(ov.get("group_col") or "")
                               for ov in (plan.get("cycle_overrides") or {}).values()
                               if isinstance(ov, dict)])
        if re.search(r"\bSTRATUM\s*(=|\bIN\b)", all_wheres, re.I):
            notes.append("The filter selects a SAMPLING STRATUM (a region, school type or school "
                         "network inside an economy, as coded by the national centre for that "
                         "cycle). Estimates use the same student weights and replicate weights; "
                         "a stratum is a sampling unit rather than an official OECD reporting "
                         "category, stratum codes differ between cycles, and its sample can be small.")
        strata_rows = plan.get("_strata_rows")
        if strata_rows:
            codes_text = "; ".join(
                f"PISA {c}: " + ", ".join(f"{code} = {strata_rows['labels'].get(code, code)}" for code in cs[:6])
                + (f" … ({len(cs)} strata)" if len(cs) > 6 else "")
                for c, cs in sorted(strata_rows["codes"].items()))
            who = self._names([strata_rows["economy"]])
            if strata_rows["mode"] == "exclude":
                notes.append(
                    f"Rows labelled \"{strata_rows['label']}\" are {who} WITHOUT the sampling "
                    f"stratum/strata of {strata_rows['name']} ({codes_text}); the row "
                    f"\"{strata_rows['economy']}\" is the whole economy as the OECD reports it. "
                    "Both use the official student and replicate weights.")
            else:
                notes.append(
                    f"Rows labelled \"{strata_rows['label']}\" are the sampling stratum/strata of "
                    f"{strata_rows['name']} inside {who} ({codes_text}); the row "
                    f"\"{strata_rows['economy']}\" is the whole economy as the OECD reports it. "
                    "A stratum is a sampling unit coded by the national centre for that cycle, "
                    "not an official OECD reporting category; its sample can be small, its "
                    "codes differ between cycles, and both rows use the official weights.")
        if plan.get("_strata_relabelled"):
            notes.append("Rows whose label carries a \"/\" are filtered to the sampling strata "
                         "named after the slash (\"excl.\" = the economy without them), not the "
                         "whole economy.")
        if plan.get("_pooled_to_rows"):
            notes.append("The statistic is computed per economy (never as one pooled mean of "
                         "students across economies, which is no OECD statistic); a group "
                         "average, where asked, is the unweighted mean of the members' estimates.")
        if plan.get("_members_as_rows"):
            notes.append("The rows are the members of the requested group; the \"avg\" row is "
                         "their unweighted average with SE = sqrt(sum of SE²)/N.")
        if plan.get("_pair"):
            a, b = plan["_pair"]
            notes.append(f"The difference {self._names([a])} minus {self._names([b])} and, across "
                         "cycles, the change in that difference are stated in the verified "
                         "statements (independent samples; the link error cancels in a "
                         "difference of two economies measured on the same linked scale).")
        if plan.get("_invented_codes"):
            notes.append("Economy codes corrected to the PISA codes: " + ", ".join(
                f"{a} → {b} ({self._names([b])})" for a, b in sorted(plan["_invented_codes"].items())) + ".")
        if plan.get("_quarter_filter"):
            qs = ", ".join(str(q) for q in plan["_quarter_filter"])
            notes.append(f"Only quarter {qs} of {plan.get('quart_variable')} is shown (1 = bottom, 4 = top; "
                         "quarters cut within each economy with the student weights); the other quarters "
                         "were computed and left out as the question asked.")
        if plan.get("_level1_or_below"):
            notes.append("\"Level 1 or below\" is read the OECD way: every student below Level 2 "
                         "(Levels 1a, 1b, 1c and below), the low-performer group.")
        if plan.get("_by_overrides"):
            notes.append("The grouping variable differs by cycle (" + "; ".join(
                f"PISA {c}: {', '.join(v)}" for c, v in sorted(plan["_by_overrides"].items()))
                + "); rows are aligned on the first cycle's column names.")
        if plan.get("_null_safe_share"):
            notes.append("Shares are percentages of students with a valid response: students "
                         "who did not answer the item (or were not asked it) are excluded "
                         "from the denominator, and a group with no valid responses shows "
                         "no estimate rather than 0.")
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        all_blank = bool(plan.get("_non_comparable")) and not plan.get("_link_error") \
            and not any((plan.get("_link_errors") or {}).values())
        if len(cycles) >= 2 and not plan.get("_no_trend") and not all_blank:
            first, last = plan.get("_trend_cycles") or (cycles[0], cycles[-1])
            method += (f" Cross-cycle change = {last} minus {first}: independent "
                       f"samples, SE = sqrt(SE{first[2:]}^2 + SE{last[2:]}^2).")
            modes = plan.get("_link_error_modes") or {}
            les = plan.get("_link_errors") if plan.get("_link_errors") is not None \
                else {None: plan.get("_link_error")}
            by_mode: dict[str, list[str]] = {}
            for lab, mode in modes.items():
                by_mode.setdefault(mode, []).append(lab)
            lab_of = lambda lab: (f"{lab}: " if lab else "")  # noqa: E731
            if "mean" in by_mode or "percentile-approx" in by_mode:
                def _dom(lab):
                    expr = plan.get("measure") if lab is None else next(
                        (e for l_, e in (self._measure_list(plan) or []) if l_ == lab), plan.get("measure"))
                    d = link_errors.domain_of(expr)
                    return link_errors.DOMAIN_NAMES.get(d, "this measure")
                parts_le = [f"{lab_of(lab)}{_dom(lab)} {first}→{last} ({les.get(lab)} points)" for lab in
                            by_mode.get("mean", []) + by_mode.get("percentile-approx", []) if les.get(lab)]
                method += f" Plus the OECD link error for {last} vs {first}."
                notes.append("Trend SEs include the OECD link error for " + "; ".join(parts_le) +
                             f" ({link_errors.SOURCE}), so significance of changes matches the "
                             "OECD's reports." +
                             (" For percentiles and quartile means the mean-score link error is "
                              "used as an approximation of the percentile-specific link errors "
                              "the OECD publishes." if "percentile-approx" in by_mode else ""))
            if "share" in by_mode:
                notes.append("For the share of students at a proficiency threshold, the link "
                             "error is derived per economy from the published score link error "
                             "(every plausible value shifted by ± the link error; half the "
                             "difference in the share), following the OECD's approach for "
                             "cumulative-distribution statistics (PISA 2022 Results Vol. I, "
                             "Annex A7), and included in the change SE.")
            if "cancels" in by_mode:
                notes.append("No link error is added to this change: for a difference between "
                             "groups measured in the same cycle (a gap, a quartile gap, a "
                             "percentile spread, a regression coefficient) the scale-linking "
                             "uncertainty shifts both groups alike and cancels (PISA 2022 "
                             "Results Vol. I, Annex A7); the change SE is the sampling SE.")
            if "none" in by_mode:
                notes.append("Sampling-only SE for " + ", ".join(lab or "this measure" for lab in by_mode["none"]) +
                             ": no link error applies to a questionnaire index or a share of a "
                             "response code (the linking concerns the achievement scales).")
        for cycle, ov in sorted((plan.get("cycle_overrides") or {}).items()):
            if isinstance(ov, dict) and ov:
                fields = ", ".join(f"{k} = {v}" for k, v in ov.items() if not k.endswith("_note"))
                notes.append(f"PISA {cycle} uses a per-cycle variable override: "
                             f"{fields} (the variable differs in that cycle; "
                             "other cycles use the main plan fields).")
        unknown = self._unknown_country_codes(where, tables)
        if unknown:
            notes.append("The filter names economy code(s) that exist in none of "
                         f"the queried tables: {', '.join(unknown)} — those rows "
                         "are absent, not zero. Check the code (e.g. B-S-J-Z "
                         "(China) is QCI, Chinese Taipei is TAP).")

        cycles_for_regions = [t.rsplit("_", 1)[-1] for t in tables] or cycles
        region_codes = self._region_filter(plan, cycles_for_regions) if tables else {}
        if region_codes:
            names = ", ".join(str(n) for n in plan.get("regions") or [])
            for cycle in cycles_for_regions:
                info = region_codes[cycle]
                line = (f"Region \"{names}\" in PISA {cycle} = {len(info['codes'])} "
                        f"economies: {', '.join(info['codes']) or 'none'}")
                if info["absent"]:
                    line += (f"; not in the {cycle} data: {', '.join(info['absent'])}")
                notes.append(line + ".")
        for cycle in plan.get("_empty_cycles") or []:
            named = self._economies_in_data(str(plan.get("where") or ""))
            who = ", ".join(self.economy_names.get(c, c) for c in named) or "this filter"
            joined = ""
            if len(named) == 1:
                first = min(c for c, p in self.present.items() if named[0] in p)
                joined = f" ({who} first took part in PISA {first})"
            notes.append(f"PISA {cycle}: no data for {who}{joined}; that cycle is "
                         "omitted from the table and the change is computed between "
                         "the cycles that have data.")
        notes.extend(plan.get("_direction_fixed") or [])
        for cycle in sorted(plan.get("_coverage") or {}):
            for f in plan["_coverage"][cycle]:
                notes.append(self._coverage_note(f))
        notes.extend(plan.get("_non_comparable") or [])
        for cycle in sorted(plan.get("_unavailable") or {}):
            labels = plan["_unavailable"][cycle]
            notes.append(f"PISA {cycle}: {', '.join(labels)} — not in the {cycle} file "
                         "(the index or item exists only in other cycles), so no "
                         f"{cycle} value is computed for it; the other measures are.")
        if plan.get("limitation_note"):
            notes.append(f"NOT DONE: {plan['limitation_note']}")
        if self.WHY_WORDS.search(str(plan.get("_question") or "")) \
                and not plan.get("limitation_note"):
            notes.append(self.WHY_NOTE)

        sample = []
        for tbl in tables:
            try:
                cycle = tbl.rsplit("_", 1)[-1]
                clause_where = where
                if cycle in region_codes:
                    codes = region_codes[cycle]["codes"]
                    reg = ("CNT IN (" + ", ".join(f"'{c}'" for c in codes) + ")"
                           if codes else "FALSE")
                    clause_where = f"({where}) AND {reg}" if where else reg
                clause = f" WHERE {clause_where}" if clause_where else ""
                has_school = "CNTSCHID" in self._table_columns(tbl)
                schools_sql = ", count(DISTINCT CNTSCHID)" if has_school else ", NULL"
                n, wsum, schools = self.con.sql(
                    f"SELECT count(*), sum(W_FSTUWT){schools_sql} FROM {tbl}{clause}").fetchone()
                sample.append({"table": tbl, "students": int(n),
                               "schools": int(schools) if schools else None,
                               "weighted_students": round(wsum) if wsum else None})
            except Exception:
                sample.append({"table": tbl})

        return {
            "source": SOURCE_LINE,
            "tables": tables or ["(raw SQL — see query)"],
            "variables": self._referenced_variables(plan, tables),
            "filter": (where or "none (all rows)")
                      + (f" + region(s): {', '.join(str(n) for n in plan['regions'])}"
                         if plan.get("regions") else ""),
            "method": method.strip(),
            "sample": sample,
            "sql": plan.get("sql") if raw else None,
            "notes": notes,
            "build": stamp(),
        }

    # ---------- the full loop ----------

    def _unknown_country_codes(self, where: str | None, tables: list[str]) -> list[str]:
        """Codes quoted in the filter that no queried table contains."""
        codes = sorted(set(re.findall(r"'([A-Z]{3})'", where or "")))
        if not codes or not tables:
            return []
        present: set[str] = set()
        for tbl in tables:
            try:
                present |= {r[0] for r in self.con.sql(
                    f"SELECT DISTINCT CNT FROM {tbl}").fetchall()}
            except Exception:  # noqa: BLE001 — a table without CNT
                return []
        return [c for c in codes if c not in present]

    YEAR_RE = re.compile(r"\b(2018|2022|2025)\b")
    ITEM_WORDS = re.compile(r"\b(item|items|test|cognitive|timing|process|log|logs|"
                            r"response time|actions)\b", re.IGNORECASE)
    MISSING_LABEL = re.compile(r"valid skip|system missing|not administered|"
                               r"not applicable|invalid|no response|not reached|"
                               r"^missing", re.IGNORECASE)
    MAX_EXPLORE_ROWS = 40
    EXPLORE_MIN_SCORE = 3.5     # a whole-word label match on a synonym term (8 x 0.5)
                                # passes; entity-only and substring noise does not

    def _explore(self, question: str, terms: list[str]) -> AgentResult:
        """Answer 'what data is there about X' from the catalog itself: one
        row per matching variable with its label, the tables (cycles and
        instruments) it exists in, and its response codes. Deterministic —
        no plan, no statistic, no LLM summary that could invent variables."""
        hits = self._retrieve(terms, per_term=12)
        if not hits.empty:      # weak, scattered token matches are not "data on X"
            hits = hits[hits.score >= self.EXPLORE_MIN_SCORE]
        years = sorted(set(self.YEAR_RE.findall(question)))
        if years and not hits.empty:
            hits = hits[hits.cycle.isin(years)]
        # "What data is there on X" means questionnaire content (items and
        # derived indices); test items, timing and process logs only when
        # the question asks for them.
        if not hits.empty and not self.ITEM_WORDS.search(question):
            questionnaire = hits.table_name.str.split("_").str[1].eq("qqq")
            hits = hits[questionnaire] if questionnaire.any() else hits
        rows = []
        for var, group in hits.groupby("variable", sort=False):
            group = group.sort_values("table_name")
            latest = group.sort_values("cycle").iloc[-1]
            desc = catalog.describe(var, cycle=latest.cycle)
            values = ""
            if not desc.empty and desc.iloc[0].value_labels:
                labels = {k: v for k, v in json.loads(desc.iloc[0].value_labels).items()
                          if not self.MISSING_LABEL.search(str(v))}
                shown = [f"{k}={v}" for k, v in list(labels.items())[:4]]
                values = ("; ".join(shown) + (" …" if len(labels) > 4 else "")
                          if labels else "continuous index / numeric")
            rows.append({"variable": var, "label": latest.label,
                         "tables": ", ".join(group.table_name),
                         "cycles": ", ".join(sorted(set(group.cycle))),
                         "values": values, "score": float(group.score.iloc[0])})
        table = (pd.DataFrame(rows).sort_values(["score", "variable"],
                                                ascending=[False, True])
                 .head(self.MAX_EXPLORE_ROWS).drop(columns="score")
                 .reset_index(drop=True)) if rows else pd.DataFrame(
            columns=["variable", "label", "tables", "cycles", "values"])
        scope = f"PISA {', '.join(years)}" if years else "PISA 2018, 2022 and 2025"
        topic = ", ".join(t for t in terms if not self.YEAR_RE.fullmatch(t)) or "that topic"
        if table.empty:
            answer = (f"No catalog variables matched “{topic}” in {scope}. The "
                      "public-use files hold questionnaire responses, derived "
                      "indices and test results — not the published report tables "
                      "(coverage or exclusion rates, OECD averages, trend tables). "
                      "Try other words for the construct (PISA labels use the "
                      "OECD's wording, e.g. “sense of belonging”, “bullying”, "
                      "“life satisfaction”).")
        else:
            top = "; ".join(f"{r.variable} ({r.label[:60]}{'…' if len(r.label) > 60 else ''})"
                            for r in table.head(5).itertuples())
            answer = (f"{len(table)} catalog variables relate to “{topic}” in {scope}; "
                      f"the closest matches: {top}. The table lists each one with "
                      "its codebook label, the tables it exists in, and its response "
                      "codes. To analyze one, ask for a statistic — for example "
                      f"“mean {table.variable.iloc[0]} by country in "
                      f"{years[-1] if years else '2025'}” or “what share of students "
                      "in Brazil chose each answer?”.")
        plan = {"template": "explore", "cycles": years or None,
                "search_terms": terms,
                "explanation": f"Catalog search for variables about {topic}"}
        provenance = {
            "source": SOURCE_LINE,
            "tables": sorted({t for r in table.itertuples()
                              for t in r.tables.split(", ")}) if not table.empty else [],
            "variables": [{"variable": r.variable, "table": r.tables.split(", ")[-1],
                           "label": r.label} for r in table.head(15).itertuples()],
            "filter": f"search terms: {', '.join(terms)}"
                      + (f"; cycles: {', '.join(years)}" if years else ""),
            "method": "Keyword search over the OECD codebooks (variable names and "
                      "labels). No statistic was computed.",
            "sample": [], "sql": None, "notes": [], "build": stamp(),
        }
        return AgentResult(question, answer, plan=plan, table=table,
                           provenance=provenance, retrieved=hits, route="explore")

    SUMMARY_ROWS = 30          # default window handed to the summarizer
    RANKING_ROWS = 120         # a ranked per-economy table is passed whole up to this

    def _summary_view(self, table: pd.DataFrame, question: str,
                      history: list | None = None) -> tuple[pd.DataFrame, str, str]:
        """What the summarizer is allowed to see.

        A ranked per-economy table (it carries a `rank` column) is passed in
        full, in a compact form (rank, economy, estimates), because 90 short
        rows are cheap and truncating them made "where is Kosovo ranked"
        unanswerable. Any other large table keeps the 30-row window plus the
        do-not-guess warning. In both cases, economies named in the question
        (or in the follow-up's context) get their rows spelled out as FOCUS
        ROWS, computed here — never left for the model to count."""
        compact_cols = [c for c in table.columns
                        if c in ("rank", "CNT", "contrast", "quarter", "percentile",
                                 "term", "cycle") or ESTIMATE_COL.match(c)
                        or c.startswith("se") or c in ("change", "se_change")]
        ranked = "rank" in table.columns and len(table) <= self.RANKING_ROWS
        if ranked:
            shown = table[compact_cols].round(1)
            truncation = ""
        else:
            shown = table.head(self.SUMMARY_ROWS).round(2)
            truncation = (
                f"WARNING: the table has {len(table)} rows but only the first "
                f"{self.SUMMARY_ROWS} are shown below. If the question needs rows "
                "beyond these (rankings, extremes, totals), say the full table is "
                "in the result — NEVER answer it from this partial view.\n"
            ) if len(table) > self.SUMMARY_ROWS else ""

        focus = ""
        if "CNT" in table.columns:
            text = question + " " + " ".join(h.get("question", "") for h in (history or [])[-2:])
            hits = self._economies_mentioned(text, table["CNT"].astype(str).unique())
            if hits:
                rows = table[table["CNT"].isin(hits)]
                if not rows.empty:
                    cols = [c for c in compact_cols if c in rows.columns]
                    lines = []
                    for r in rows[cols].round(1).itertuples(index=False):
                        d = r._asdict()
                        pos = (f"rank {int(d['rank'])} of {int(table['rank'].notna().sum())}"
                               if "rank" in d and not pd.isna(d["rank"]) else "")
                        vals = ", ".join(f"{k} {v}" for k, v in d.items()
                                         if k not in ("rank", "CNT") and not pd.isna(v))
                        lines.append(f"{d['CNT']}: {pos}{'; ' if pos else ''}{vals}")
                    focus = ("FOCUS ROWS (economies named in the question, located by "
                             "the app — state these positions and values exactly):\n"
                             + "\n".join(lines) + "\n")
        return shown, truncation, focus

    def _economies_in_data(self, text: str) -> list[str]:
        """Economies named in the text that exist in at least one loaded cycle."""
        present = getattr(self, "present", None) or {}
        every = set().union(*present.values()) if present else set()
        return self._economies_mentioned(text, sorted(every))

    def _economies_mentioned(self, text: str, codes) -> list[str]:
        """Codes of economies named in the text, by code or by name (the
        catalog's CNT labels), longest names first so 'Korea' does not match
        inside 'North Korea'-style labels."""
        names = {}
        for cycle in ("2025", "2022", "2018"):
            desc = catalog.describe("CNT", cycle=cycle)
            desc = desc[desc.table_name.str.startswith("stu_qqq")]
            if not desc.empty and desc.iloc[0].value_labels:
                for k, v in json.loads(desc.iloc[0].value_labels).items():
                    names.setdefault(k, v)
        low = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
        # codes only as written in capitals: ARE, CAN, PER are also English words
        caps = " " + re.sub(r"[^A-Za-z0-9 ]", " ", text) + " "
        found = []
        for code in codes:
            if re.search(rf"\b{re.escape(code)}\b", caps):
                found.append(code)
                continue
            negatives = regions.NEGATIVE_ALIASES.get(code, [])
            if negatives and any(f" {n} " in low for n in negatives):
                # "north korea" names a country that is not KOR — unless the
                # text also names the economy itself
                positives = [p for p in regions.ECONOMY_ALIASES.get(code, []) + [names.get(code, "").lower()]
                             if p.strip() and not any(p in n for n in negatives)]
                if not any(f" {re.sub(r'[^a-z0-9 ]', ' ', p).strip()} " in low for p in positives):
                    continue
            label = names.get(code, "")
            base = re.sub(r"\s*\(.*?\)\s*", " ", label).strip().lower()   # "Macao (China)" -> "macao"
            cands = {base, label.lower(), *regions.ECONOMY_ALIASES.get(code, [])}
            for cand in cands:
                cand = re.sub(r"[^a-z0-9 ]", " ", cand).strip()
                cand = re.sub(r"\s+", " ", cand)
                if len(cand) >= 3 and f" {cand} " in low:
                    found.append(code)
                    break
        return found

    @staticmethod
    def _country_legend(shown: pd.DataFrame) -> str:
        """CNT code -> economy name for the codes in the shown table, so the
        summary can say 'B-S-J-Z (China)' instead of the bare code 'QCI'."""
        if "CNT" not in shown.columns:
            return ""
        codes = [c for c in shown["CNT"].dropna().astype(str).unique() if not c.endswith(" avg")]
        if not codes:
            return ""
        names: dict[str, str] = {}
        for cycle in ("2025", "2022", "2018"):
            desc = catalog.describe("CNT", cycle=cycle)
            desc = desc[desc.table_name.str.startswith("stu_qqq")]
            if not desc.empty and desc.iloc[0].value_labels:
                for k, v in json.loads(desc.iloc[0].value_labels).items():
                    names.setdefault(k, v)
        pairs = [f"{c} = {names[c]}" for c in codes if c in names]
        if not pairs:
            return ""
        return ("COUNTRY CODES (use the name, with the code in parentheses on "
                f"first mention): {'; '.join(pairs)}\n")

    @staticmethod
    def _transcript(history) -> str:
        """Compact conversation context: last few exchanges, truncated."""
        if not history:
            return ""
        lines = []
        for h in history[-4:]:
            lines.append(f"User: {h['question'][:400]}")
            lines.append(f"Assistant: {(h.get('answer') or '')[:400]}")
            if h.get("explanation"):
                lines.append(f"  (analysis run: {h['explanation'][:200]})")
        return "CONVERSATION SO FAR (use it to resolve follow-ups and answers " \
               "to clarifying questions):\n" + "\n".join(lines) + "\n\n"

    # Deterministic, honest answer for "can you chart/visualize that?" —
    # the router LLM cannot be trusted to describe the app's own abilities
    # (it reverts to "I am a text-based AI"), so the app answers this itself.
    VIZ_WORDS = re.compile(r"\b(visuali[sz]|chart|graph|plot|diagram|draw)", re.IGNORECASE)
    VIZ_ANSWER = (
        "Charts render automatically whenever a result compares things: country "
        "or group comparisons draw bar charts with 95%-confidence whiskers, "
        "cross-cycle questions (2018, 2022, 2025) draw dumbbell charts with one "
        "dot per cycle, and group gaps draw diverging bars. A single-number "
        "result (like a lone correlation or one average) has no chart form, so "
        "only its table is shown. To get a chart, ask for a comparison — across "
        "countries, groups (gender, immigrant background, grade repetition…), "
        "or cycles. For example: “compare science scores for boys and girls in "
        "Spain, France and Germany in 2025” or “how did reading in Finland "
        "change from 2018 to 2025?”."
    )

    def ask(self, question: str, history: list | None = None) -> AgentResult:
        """Answer one question; attaches route + timing for analytics."""
        llm.reset_stats()
        self._fired = []
        started = time.time()
        try:
            result = self._ask(question, history)
        except Exception as e:  # noqa: BLE001 — surface as a result, not a crash
            result = AgentResult(question, f"Something went wrong: {e}",
                                 error=str(e), route="error")
        result.timing = {"total_ms": round((time.time() - started) * 1000),
                         **llm.stats()}
        result.guards = list(self._fired)
        return result

    # Deterministic answers with one correct wording: checked before any
    # model runs (on the question as typed) and again after routing on the
    # router's English rendering, so a Japanese or Spanish question gets the
    # same safeguards as an English one.
    # "How is ESCS constructed / which components changed" is a question
    # about an index, not about comparing scores across cycles
    INDEX_QUESTION_WORDS = re.compile(
        r"\b(constructed|construction|computed|components?|built|derived|composition|"
        r"how is .{0,30}(index|escs|scale) (made|calculated|built))\b", re.IGNORECASE)
    MODE_WORDS = re.compile(
        r"\b(paper|computer)[- ]based\b|\bon paper\b|\bpaper (test|version|form|and pencil)|"
        r"\b(paper|pencil)\b.{0,25}\b(computer|screen|digital)\b|\bcomputer\b.{0,25}\bpaper\b|"
        r"\btest(ing)? mode\b|\badministration mode\b|\badminmode\b|\bcomputer[- ]delivered\b|"
        r"\b(en papel|por computadora|en computadora|na papel|no computador)\b", re.IGNORECASE)
    AVG_SE_WORDS = re.compile(
        r"\b(se|standard error|variance|uncertainty)\b.{0,80}\b(difference|minus|gap|deviation)\b"
        r".{0,60}\b(oecd|average|mean of)\b|\b(oecd|average)\b.{0,40}\b(se|standard error)\b.{0,60}"
        r"\b(difference|formula|computed|calculated)\b", re.IGNORECASE)

    def _intercept(self, text: str, history: list | None = None) -> str | None:
        # a follow-up ("why is there no math for 2025?") inherits the economies
        # of the last exchanges — from the answers too, whose codes are in
        # capitals whatever language the questions were asked in
        recent = " ".join(f"{h.get('question', '')} {(h.get('answer') or '')[:600]}" for h in (history or [])[-2:])
        named_recent = self._economies_in_data(text) or self._economies_in_data(recent)
        if self.LINK_WORDS.search(text) and not (self.LINK_DATA_REQUEST.search(text)
                                                 and self._economies_in_data(text)):
            self._fire("intercept:link_error")
            return self._link_error_answer(self._last_link_mode)
        if self.AVG_SE_WORDS.search(text) and not self._economies_in_data(text):
            self._fire("intercept:avg_se_method")
            return self._avg_se_answer()
        if self.OECD_AVG_WHY.search(text):
            self._fire("intercept:oecd_average_method")
            return self._oecd_average_answer(text)
        if self.OVERVIEW_WORDS.search(text):
            self._fire("intercept:overview")
            return self._overview_answer()
        if self.WHY_MISSING_WORDS.search(text):
            if named_recent:
                self._fire("intercept:why_missing")
                return self._missing_results_answer(named_recent)
        if self.COUNT_WORDS.search(text) and not self.OTHER_STAT_WORDS.search(text):
            named = self._economies_in_data(text)
            if named:
                self._fire("intercept:count")
                return self._count_answer(named, text)
            if re.search(r"\b(in total|overall|altogether|all (countries|economies)|worldwide)\b", text, re.I):
                self._fire("intercept:count_total")
                return self._count_total_answer(text)
        if self.COVERAGE_RATE_WORDS.search(text):
            self._fire("intercept:coverage_rate")
            return self._coverage_rate_answer()
        if self.MODE_WORDS.search(text) and not re.search(r"\b(score|scores|mean|average|gap|share|percent)\b.{0,40}\b(by|per|for) (paper|computer)", text, re.I):
            self._fire("intercept:test_mode")
            return self._mode_answer(named_recent, text)
        if self.COMPARABLE_WORDS.search(text) and not self.INDEX_QUESTION_WORDS.search(text) \
                and (len(set(self.YEAR_RE.findall(text))) >= 2
                     or re.search(r"\b(cycles?|years?|over time)\b", text, re.I)):
            self._fire("intercept:comparability")
            return self._comparability_answer(named_recent, text)
        return None

    _last_link_mode: str | None = None

    OECD_AVG_WHY = re.compile(
        r"\b(why|how)\b.{0,60}\boecd (average|mean)\b.{0,80}\b(differ|different|match|computed|calculated|"
        r"defined|built|members?|table|volume)\b|\bhow (is|do you (compute|calculate)) the oecd (average|mean)\b|"
        r"\bwhich (countries|economies|members) (are|is) in (the|your) oecd (average|mean)\b", re.IGNORECASE)

    def _oecd_average_answer(self, text: str) -> str:
        counts = {c: len(self._oecd_codes(c)) for c in CYCLES if c in self.present}
        members = ", ".join(f"PISA {c}: {n} members" for c, n in sorted(counts.items()))
        years = sorted(set(self.YEAR_RE.findall(text or "")))
        value = ""
        if years and years[0] in counts:
            try:
                domain = "READ" if re.search(r"\bread", text, re.I) else "SCIE" if re.search(r"\bscien", text, re.I) else "MATH"
                excluded = self.AVERAGE_EXCLUSIONS.get(("OECD avg", years[0], domain), set())
                codes = sorted(self._oecd_codes(years[0]) - excluded)
                lst = ", ".join(f"'{c}'" for c in codes)
                rows = self.con.execute(
                    f"SELECT CNT, SUM(W_FSTUWT * PV1{domain}) / SUM(W_FSTUWT) FROM stu_qqq_{years[0]} "
                    f"WHERE CNT IN ({lst}) AND PV1{domain} IS NOT NULL GROUP BY CNT").fetchall()
                vals = [r[1] for r in rows if r[1] is not None]
                if vals:
                    value = (f" For {years[0]} {link_errors.DOMAIN_NAMES[domain]}, the app's OECD average is "
                             f"{np.mean(vals):.1f} (first plausible value, {len(vals)} members"
                             + (f", {self._names(sorted(excluded))} excluded as in the OECD's tables" if excluded else "")
                             + "), which should round to the published figure; ask for it as a data question "
                             "to get the 10-PV estimate with its standard error.")
            except Exception:  # noqa: BLE001 — the method text stands on its own
                value = ""
        return (
            "The OECD average in this app follows the OECD's own convention: the unweighted mean of "
            "the OECD member countries' estimates (each member counts once, whatever its size), with "
            f"SE = sqrt(sum of the members' SE²) / N — over the members present in the cycle ({members}), "
            "never over all participating economies. In a trend the same members enter both cycles; "
            "Spain is left out of the 2018 reading average as in the OECD's tables (Annex A9)." + value +
            " Reasons a figure can still differ from a published table: rounding; a table that uses a "
            "different member set (e.g. the 'OECD average-35' of earlier reports); a table computed on "
            "the trend membership; or a Volume printed before a data correction. Every OECD-average row "
            "in an answer states the members it was computed on in the provenance card.")

    REGION_LIST_WORDS = re.compile(
        r"\b(per|by|each|every|across|all|for the different|broken down by|breakdown by) "
        r"(region|regions|province|provinces|state|states|oblast|oblasts|governorate|governorates|"
        r"emirate|emirates|department|departments|canton|cantons|prefecture|prefectures|"
        r"district|districts|county|counties)\b|\bregional (results|breakdown|ranking|scores)\b|"
        r"\b(results|scores) (per|by) region\b|\bsub-?national (results|breakdown)\b", re.IGNORECASE)

    def _regions_answer(self, codes: list[str], question: str) -> str | None:
        """PISA reports economies nationally; the only sub-national handles
        in the public files are the sampling strata. Name them for the
        economy asked about, with the caveat, instead of "no regional data"."""
        index = self._strata_index()
        years = sorted(set(self.YEAR_RE.findall(question or ""))) or [c for c in reversed(CYCLES) if c in self.present][:1]
        for code in codes[:1]:
            for cycle in years:
                rows = [(c, code_s, label) for c, code_s, label, _ in index
                        if c == cycle and (code_s.startswith(code) or
                                           (code == "GBR" and code_s[:3] in ("QSC", "QUK")))]
                if not rows:
                    continue
                segments = []
                for _, _, label in rows:
                    body = re.sub(r"^\w{3} - stratum \d+:\s*", "", label)
                    first = re.split(r"[/,:;()–]|\s-\s", body)[0].strip()
                    if first and not first.lower().startswith("undisclosed"):
                        segments.append(first)
                distinct = list(dict.fromkeys(segments))
                if not distinct:
                    continue
                name = self.economy_names.get(code, code)
                listed = "; ".join(distinct[:30]) + (f"; … ({len(distinct)} in all)" if len(distinct) > 30 else "")
                return (f"PISA reports {name} ({code}) as one economy: the OECD publishes no official "
                        f"regional results for it, and the public file carries no region variable. What "
                        f"the file does carry is the SAMPLING STRATUM each school was drawn from — in "
                        f"PISA {cycle}, {len(rows)} strata labelled by the national centre: {listed}. A "
                        "stratum estimate uses the official weights but is a sampling unit, not an "
                        "OECD reporting category; its sample can be small (the app blanks anything "
                        "under 30 students or 5 schools) and the labels change between cycles. Name "
                        f"one to see it beside the national figure — for example “mean mathematics "
                        f"score in {name} {cycle} and in {distinct[0]}”.")
        return None

    def _avg_se_answer(self) -> str:
        return (
            "The standard error of an economy's difference from an OECD (or any group) average "
            "is computed as the OECD's Data Analysis Manual prescribes for an average of "
            "independent samples. The average is the unweighted mean of the members' estimates, "
            "so SE_avg = sqrt(sum of the members' SE²) / N. For an economy that is NOT a member, "
            "var(economy − avg) = SE_economy² + SE_avg² (independent samples). For a member, its "
            "own estimate is inside the average, so var(economy − avg) = SE_economy² × (1 − 2/N) "
            "+ SE_avg². Within one economy, differences between groups (a gender gap, a quartile "
            "gap) are instead computed replicate by replicate with the 80 Fay-BRR weights, which "
            "carries the covariance between the groups. Every difference the app states in its "
            "verified statements names which of the two it used.")

    # ---------- test administration mode (paper / computer) ----------

    def _mode_index(self) -> dict[str, dict[str, str]]:
        """{cycle: {CNT: 'paper' | 'computer' | 'mixed'}} from ADMINMODE in the
        student files (1 = paper, 2 = computer in 2018/2022; the reverse in
        2025 — read from each cycle's value labels)."""
        cache = self.__dict__.get("_mode_cache")
        if cache is not None:
            return cache
        out: dict[str, dict[str, str]] = {}
        for cycle in CYCLES:
            if cycle not in self.present:
                continue
            try:
                desc = catalog.describe("ADMINMODE", cycle=cycle)
                desc = desc[desc.table_name.str.startswith("stu_qqq")]
                labels = json.loads(desc.iloc[0].value_labels) if not desc.empty and desc.iloc[0].value_labels else {}
                meaning = {}
                for k, v in labels.items():
                    code = str(int(float(k)))
                    meaning[code] = "paper" if "paper" in str(v).lower() else "computer"
                rows = self.con.execute(
                    f"SELECT CNT, CAST(ADMINMODE AS INTEGER), COUNT(*) FROM stu_qqq_{cycle} "
                    "WHERE ADMINMODE IS NOT NULL GROUP BY 1, 2").fetchall()
                per: dict[str, dict[str, int]] = {}
                for cnt, code, n in rows:
                    per.setdefault(cnt, {})[meaning.get(str(code), "computer")] = per.get(cnt, {}).get(
                        meaning.get(str(code), "computer"), 0) + int(n)
                out[cycle] = {cnt: (max(d, key=d.get) if max(d.values()) >= 0.9 * sum(d.values()) else "mixed")
                              for cnt, d in per.items()}
            except Exception:  # noqa: BLE001 — a missing variable means no mode information
                out[cycle] = {}
        self._mode_cache = out
        return out

    def _mode_changes(self, codes, cycles) -> list[tuple[str, str, str, str, str]]:
        """(code, first cycle, mode, last cycle, mode) for economies whose
        mode differs between the first and last of `cycles`."""
        modes = self._mode_index()
        cycles = sorted(str(c) for c in cycles if str(c) in modes)
        if len(cycles) < 2:
            return []
        first, last = cycles[0], cycles[-1]
        out = []
        for code in codes:
            a, b = modes[first].get(code), modes[last].get(code)
            if a and b and a != b:
                out.append((code, first, a, last, b))
        return out

    def _mode_answer(self, codes: list[str], text: str) -> str:
        modes = self._mode_index()
        if codes:
            parts = []
            for code in codes[:6]:
                per = [f"PISA {c}: {modes[c][code]}" for c in sorted(modes) if code in modes[c]]
                if per:
                    parts.append(f"{self.economy_names.get(code, code)} ({code}) — " + ", ".join(per))
            if not parts:
                return ("The administration mode is recorded in the public files (ADMINMODE), but "
                        "none of the named economies appears in the loaded cycles.")
            changed = self._mode_changes(codes[:6], sorted(modes))
            caution = ""
            if changed:
                caution = (" A change of mode matters for trends: the OECD reports the linked trend "
                           "with the link error, but mode effects are not quantified in the public "
                           "files; the PISA 2025 Technical Report (Data Adjudication) recommends "
                           "caution for economies that moved from paper to computer.")
            return ("From the ADMINMODE variable in the student files (the mode each sampled "
                    "student was tested in): " + "; ".join(parts) + "." + caution)
        lines = []
        for c in sorted(modes):
            paper = sorted(k for k, v in modes[c].items() if v == "paper")
            mixed = sorted(k for k, v in modes[c].items() if v == "mixed")
            lines.append(f"PISA {c}: paper-based in {self._names(paper) if paper else 'no economy'}"
                         + (f"; mixed in {self._names(mixed)}" if mixed else "")
                         + "; computer-based everywhere else")
        return ("From the ADMINMODE variable in the student files: " + ". ".join(lines) +
                ". Name an economy to see its mode in each cycle.")

    AI_WORDS = re.compile(r"\b(ai|a\.i\.|artificial intelligence|chatgpt|chatbots?|"
                          r"generative ai|llm|llms)\b|人工知能|生成AI|チャットボット", re.IGNORECASE)
    AI_DATA_WORDS = re.compile(r"\b(use|uses|usage|using|used|adoption|rate|rates|share|percent\w*|"
                               r"how (many|often|much)|data|compare|students?|schools?|country|"
                               r"countries|japan|relationship|associat\w*|correlat\w*)\b",
                               re.IGNORECASE)

    ENGLISH_WORDS = re.compile(r"\b(the|is|are|was|were|what|how|which|of|and|in|for|with|do|does|did|between|from)\b", re.I)

    @classmethod
    def _looks_english(cls, text: str) -> bool:
        return text.isascii() and bool(cls.ENGLISH_WORDS.search(text or ""))

    def _localize(self, text: str, language: str | None) -> str:
        """Deterministic answers are written in English; a non-English question
        gets a faithful rendering in its own language (numbers, codes and
        source names unchanged)."""
        if not text or not language or language.strip().lower() in ("english", "en", ""):
            return text
        try:
            out = generate(
                f"Translate the following answer into {language}. Keep every number, "
                f"standard error, economy code (like USA, QCI), variable code (like "
                f"ST438Q01DA) and source title exactly as written; translate nothing "
                f"else than the prose. Return only the translation.\n\n{text}",
                system="You are a precise translator for a statistics app.")
            out = out.strip()
            # a translation that changes a number is worse than English
            source_numbers = {(v, d) for _, v, d in summ._numbers_in(text)}
            if any((v, d) not in source_numbers for _, v, d in summ._numbers_in(out)):
                self._fire("localize:numbers_changed")
                return text
            return out or text
        except Exception:  # noqa: BLE001 — never lose the answer over a translation
            return text

    BEST_COUNTRY_WORDS = re.compile(
        r"\b(which|what|who)\b.{0,20}\b(country|countries|economy|economies|nation|nations|system|systems)\b"
        r".{0,25}\b(best|top|highest|strongest|leads?|leading|number one|first|winner|wins?)\b|"
        r"\b(best|top) (performing |scoring )?(country|countries|economy|economies|nation|system)\b|"
        r"\b(who|which) (is|are|was|were) (the )?(best|top|number one|winner|leader)\b|"
        r"\bcu[aá]l (es el|fue el) mejor pa[ií]s\b|\bqu[eé] pa[ií]s (es|fue) (el )?mejor\b", re.IGNORECASE)

    # Economy codes a planner invents from ISO habits for economies whose PISA
    # code differs (Tajikistan is Dushanbe QTJ; Taiwan is Chinese Taipei TAP)
    INVENTED_CODES = {"TJK": "QTJ", "TWN": "TAP", "IRQ": "QKI", "XKX": "KSV", "XKO": "KSV", "RKS": "KSV",
                      "UKB": "GBR", "UK": "GBR", "PAL": "PSE", "PS": "PSE", "KOS": "KSV", "MO": "MAC",
                      "CHN": "QCI", "MCO": "QMC", "RUM": "QMR"}

    def _fix_invented_codes(self, plan: dict) -> None:
        every = set().union(*self.present.values()) if getattr(self, "present", None) else set()

        def fix_text(text: str) -> str:
            def repl(m):
                code = m.group(1)
                if code not in every and code in self.INVENTED_CODES and self.INVENTED_CODES[code] in every:
                    plan.setdefault("_invented_codes", {})[code] = self.INVENTED_CODES[code]
                    return f"'{self.INVENTED_CODES[code]}'"
                return m.group(0)
            return re.sub(r"'([A-Z]{2,3})'", repl, text)

        for key in ("where",):
            if isinstance(plan.get(key), str):
                plan[key] = fix_text(plan[key])
        for ov in (plan.get("cycle_overrides") or {}).values():
            if isinstance(ov, dict) and isinstance(ov.get("where"), str):
                ov["where"] = fix_text(ov["where"])
        groups = plan.get("include_average_of")
        if isinstance(groups, list):
            fixed = []
            for g in groups:
                if isinstance(g, dict) and isinstance(g.get("members"), list):
                    g = {**g, "members": [self.INVENTED_CODES.get(c, c) if c not in every else c for c in g["members"]]}
                elif isinstance(g, list):
                    g = [self.INVENTED_CODES.get(c, c) if c not in every else c for c in g]
                elif isinstance(g, str) and g not in every and g in self.INVENTED_CODES:
                    plan.setdefault("_invented_codes", {})[g] = self.INVENTED_CODES[g]
                    g = self.INVENTED_CODES[g]
                fixed.append(g)
            plan["include_average_of"] = fixed
        if plan.get("_invented_codes"):
            self._fire("hook:invented_code")

    CANNOT_COMPUTE = re.compile(
        r"\b(cannot|can'?t|unable to|not able to|does not|doesn'?t) (directly )?(compute|calculate|"
        r"provide|determine|perform|support)\b.{0,80}\b(difference|gap|change|significan|standard error|"
        r"tied|compar)", re.IGNORECASE)

    # Methods asked for by name that the app does not implement: each gets an
    # explicit NOT DONE line, never a silent substitute.
    UNSUPPORTED = [
        (re.compile(r"\b(country|economy)?[- ]?fixed[- ]effects?\b|\bpooled (regression|model) (across|over)\b", re.I),
         "Pooled regressions with country fixed effects are not available: the app estimates each "
         "economy separately (and, when asked, the unweighted OECD average of the per-economy "
         "coefficients), never one pooled model across economies."),
        (re.compile(r"\br\s*(²|\^2|squared)\b|\bvariance explained\b", re.I),
         "R² / variance explained is not reported by the regression template."),
        (re.compile(r"\b(icc|intra-?class correlation|between-school variance|variance decomposition|"
                    r"multilevel|hierarchical linear|hlm|random effects?)\b", re.I),
         "Variance decomposition (ICC, between-school share) and multilevel models are not "
         "available; the app reports single-level weighted estimates."),
        (re.compile(r"\b(cohen'?s d|effect size in (sd|standard deviation)|standardi[sz]ed (effect|coefficient|beta))\b", re.I),
         "Effect sizes in standard-deviation units (Cohen's d, standardized betas) are not computed: "
         "coefficients are per unit of the predictor and outcomes are on the PISA score scale."),
        (re.compile(r"\bdesign effect\b|\bdeff\b|\bsimple random sampl", re.I),
         "Design effects and simple-random-sampling standard errors are not reported; every SE is "
         "the Fay-BRR replicate-weight SE, which already reflects the sample design."),
        (re.compile(r"\bunweighted (mean|average|share|percentage)\b|\bwithout (the )?weights?\b", re.I),
         "Unweighted statistics are not reported: every estimate uses the final student weight, "
         "as the OECD requires for population estimates."),
        (re.compile(r"\bcoverage[- ]adjust|adjust(ed|ing)? for (coverage|exclusion)", re.I),
         "No coverage adjustment is applied: the OECD's Coverage Index is a published table, not "
         "a variable in the files, so the estimate cannot be corrected for it here."),
        (re.compile(r"\b(resilien(t|ce)|academically resilient)\b", re.I),
         "The OECD's 'academically resilient' share (disadvantaged students in the top quarter of "
         "performance) is not a template here; the app can give the mean by ESCS quarter and the "
         "share above a level within the bottom ESCS quarter instead."),
    ]

    # "explain that more simply", "give me five bullets", "summarize the
    # takeaways": a rewrite of the previous answer, not a new analysis
    REWRITE_WORDS = re.compile(
        r"\b(simpler|simplify|simple terms|plain (english|language|words)|layman|in other words|"
        r"bullet|bullets|bullet points|takeaways?|key points|summari[sz]e (that|this|it|the (above|previous|last))|"
        r"tl;?dr|shorter|shorten|rephrase|reword|explain (that|this|it) (again|to me|more|better)|"
        r"(five|5|three|3|ten|10) (points|bullets|lines|sentences)|for my (staff|team|class|boss|students|principal)|"
        r"sin siglas|m[aá]s sencillo|m[aá]s simple|resum[ei]|en pocas palabras|plus simple|einfacher|"
        r"zusammenfass\w*|mais simples|resumo)\b", re.IGNORECASE)

    def _rewrite_previous(self, question: str, history: list | None, language: str | None) -> str | None:
        """The previous answer restated as asked; numbers are checked against
        it and, when the rewrite invents any, the previous answer is repeated."""
        # the last SUBSTANTIVE answer (one that came from an analysis, or at
        # least carries numbers), not a "no advice" reply that followed it
        answered = [h for h in reversed(history or []) if (h.get("answer") or "").strip()]
        last = next((h for h in answered if h.get("explanation")
                     or any(v >= 13 for _, v, _ in summ._numbers_in(str(h["answer"])))), None) \
            or (answered[0] if answered else None)
        if not last:
            return None
        previous = str(last["answer"])
        try:
            draft = self._plain(generate(
                f"REQUEST: {question}\n\nPREVIOUS ANSWER (restate this as requested; keep every "
                f"number, standard error and economy exactly; add nothing that is not in it; "
                f"if asked for bullets, write bullets; if asked for simpler language, drop the "
                f"acronyms and explain 'SE' as 'margin of error'):\n{previous}",
                system=("You restate a statistics answer in PISA Explorer. You never add "
                        "numbers, causes or countries that are not in the previous answer."
                        + ("" if not language or language.lower() == "english"
                           else f" Write in {language}.")))).strip()
        except llm.LLMError:
            return previous
        allowed = {(v, d) for _, v, d in summ._numbers_in(previous)}
        allowed |= {(round(v), 0) for v, _ in allowed}
        bad = [raw for raw, v, d in summ._numbers_in(draft)
               if not summ.YEAR.match(raw) and not (d == 0 and v <= 12) and (v, d) not in allowed
               and not any(abs(v - a) <= 0.5 * 10 ** (-d) + 1e-9 for a, _ in allowed)]
        if bad or not draft:
            self._fire("rewrite:numbers_changed")
            return previous
        return draft

    def _ask(self, question: str, history: list | None) -> AgentResult:
        context = self._transcript(history)
        if history and self.REWRITE_WORDS.search(question) and not self._economies_in_data(question) \
                and len(question) < 160:
            self._fire("intercept:rewrite")
            lang = None
            if not self._looks_english(question):
                try:
                    lang = str(generate_json(f"Question: {question}", system=TERMS_SYSTEM + self.facts).get("language") or None)
                except llm.LLMError:
                    lang = None
            rewritten = self._rewrite_previous(question, history, lang)
            if rewritten:
                return AgentResult(question, rewritten, route="conversational")
        early = self._intercept(question, history)
        if early:
            if not self._looks_english(question):
                # the deterministic answers are English; find the user's
                # language with the router and translate (numbers checked)
                try:
                    lang = str(generate_json(f"Question: {question}",
                                             system=TERMS_SYSTEM + self.facts).get("language") or "English")
                except llm.LLMError:
                    lang = "English"
                early = self._localize(early, lang)
            return AgentResult(question, early, route="conversational")
        route = generate_json(f"{context}Question: {question}",
                              system=TERMS_SYSTEM + self.facts)
        language = str(route.get("language") or "English")
        q_en = str(route.get("question_en") or question).strip() or question
        if q_en != question:
            early = self._intercept(q_en, history)
            if early:
                return AgentResult(question, self._localize(early, language),
                                   route="conversational")
        # "Did X participate / do you have data on X?" — answered from the
        # participant lists, never by the model, whichever way it was routed.
        if self._is_participation_question(q_en):
            named = self._economies_in_data(q_en)
            if named:
                self._fire("intercept:participation")
                return AgentResult(question, self._localize(self._participation_answer(named), language),
                                   route="conversational")
            outside = regions.non_pisa_named(q_en)
            if outside:
                self._fire("intercept:participation_none")
                return AgentResult(question, self._localize(
                    f"No — {', '.join(outside)} has not taken part in PISA 2018, 2022 or 2025 "
                    "(nor in any earlier cycle, for the countries never assessed) and is not in "
                    "the PISA 2018, 2022 or 2025 public-use databases loaded here, so no statistic "
                    "can be computed. Ask “did <economy> participate?” "
                    "for any economy, or name one from the participant lists.", language),
                    route="conversational")
        if not route.get("data_question"):
            if self.VIZ_WORDS.search(q_en):
                self._fire("intercept:viz")
                return AgentResult(question, self._localize(self.VIZ_ANSWER, language),
                                   route="conversational")
            if self.YEAR_RE.search(q_en) and self.COVERAGE_WORDS.search(q_en):
                self._fire("intercept:coverage")
                return AgentResult(question, self._localize(self._coverage_answer(), language),
                                   route="conversational")
            named = self._economies_in_data(q_en)
            ai_data = bool(self.AI_WORDS.search(q_en) and self.AI_DATA_WORDS.search(q_en)
                           and not re.search(r"literacy", q_en, re.I))
            strata = self._strata_hits(q_en)
            best = bool(self.BEST_COUNTRY_WORDS.search(q_en))
            if not named and not ai_data and not strata and not best:
                return AgentResult(question, route.get("direct_answer")
                                   or "Could you rephrase that?", route="conversational")
            if best and not named and not ai_data and not strata:
                # "which country is best": a ranking in the three subjects,
                # not an opinion about the word "best"
                self._fire("route:force_analyze:best")
                route = {"data_question": True, "intent": "analyze",
                         "search_terms": ["mathematics", "reading", "science"]}
                q_en = q_en + " (rank all economies by their mean score in mathematics, reading and science in the latest cycle, with the OECD average)"
            # The router called it conversational, but the question names an
            # economy that IS in the data (or asks about AI use, which the 2025
            # questionnaire covers, or a school network / region that is a
            # sampling stratum): the data answer, not the model's memory.
            self._fire("route:force_analyze:" + ("ai_data" if ai_data else "economy" if named else "stratum"))
            route = {"data_question": True, "intent": "analyze",
                     "search_terms": (["artificial intelligence chatbot", "AI use school"]
                                      if ai_data else None)
                     or route.get("search_terms") or ["science", "mathematics", "reading"]}

        if self.ITEM_ASK_WORDS.search(q_en) and route.get("intent") != "explore":
            # "were there questions about X?" is a catalog question, not one
            # the model may answer from memory (it invented LDW content once)
            self._fire("route:force_explore")
            route = {**route, "data_question": True, "intent": "explore",
                     "search_terms": route.get("search_terms") or
                     [w for w in re.split(r"\b(?:questions?|items?)\s+(?:about|on|regarding)\s+", q_en, flags=re.I)[-1:]
                      if w.strip()]}
        if route.get("intent") == "explore" and standards.matching(q_en) and self.ANALYSIS_WORDS.search(q_en) \
                and not self.ITEM_ASK_WORDS.search(q_en):
            # "which economies took the global competence test and what was the
            # mean in Colombia": a statistic on a standard measure, not a listing
            self._fire("route:explore_to_analyze")
            route = {**route, "intent": "analyze"}
        if route.get("intent") == "explore":
            # a variable code the user (or the thread) names is listed first
            recent = q_en + " " + " ".join(h.get("question", "") for h in (history or [])[-2:])
            codes = [c for c in dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9_]{4,}\b", recent))
                     if not catalog.describe(c).empty]
            result = self._explore(q_en, codes + list(route.get("search_terms") or []))
            result.answer = self._localize(result.answer, language)
            return result

        named = self._economies_in_data(q_en)
        if named and self.REGION_LIST_WORDS.search(q_en) and not self._strata_entities(q_en):
            regional = self._regions_answer(named, q_en)
            if regional:
                self._fire("intercept:regions_list")
                return AgentResult(question, self._localize(regional, language), route="conversational")
        outside = regions.non_pisa_named(q_en)
        if outside and not named:
            # "students in India": a country that has never been in PISA —
            # the fixed answer, before a planner can invent a code for it
            self._fire("hook:unknown_economy")
            return AgentResult(question, self._localize(
                f"{', '.join(outside)} is not in the PISA 2018, 2022 or 2025 public-use "
                "databases loaded here, so no statistic can be computed. Ask “did <economy> "
                "participate?” to check any economy, or name one from the participant lists.",
                language), route="conversational")
        terms = list(route.get("search_terms") or [])
        quoted = [q.strip() for q in re.findall(r"[“\"']([^”\"']{12,160})[”\"']", q_en)
                  if len(q.split()) >= 3]
        hits = self._retrieve(quoted + terms) if quoted else self._retrieve(terms)
        if quoted:
            # item wording quoted by the user: the closest label in EVERY cycle
            # (the 2018 form of a 2022 item was being lost to the cap)
            self._fire("retrieve:quoted_phrase")
            extras = []
            for phrase in quoted[:2]:
                for cycle in CYCLES:
                    if cycle in self.present:
                        extras.append(catalog.search(phrase, cycle=cycle, limit=2))
            if extras:
                extra = pd.concat(extras, ignore_index=True)
                hits = pd.concat([hits, extra], ignore_index=True) if hits is not None and not hits.empty else extra
                hits = hits.drop_duplicates(subset=["variable", "table_name"])
        hits = self._with_standard_cards(hits, q_en)
        unsupported = [note for rx, note in self.UNSUPPORTED if rx.search(q_en)]
        question_block = (f"QUESTION: {question}" if q_en == question
                          else f"QUESTION (original, {language}): {question}\nQUESTION (English): {q_en}")
        standard_block = standards.prompt_block(standards.matching(q_en))
        strata_block = self._strata_block(self._strata_hits(q_en))
        if strata_block:
            self._fire("standard:strata")
        plan = generate_json(
            f"{context}{question_block}\n\n{standard_block}{strata_block}VARIABLE CARDS:\n{self._cards(hits, named)}",
            system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS),
                                      regions=self.regions_block,
                                      levels=link_errors.levels_prompt()),
        )
        if plan.get("action") != "clarify" and history and plan.get("template") in self.REQUIRED_FIELDS:
            # a follow-up ("and 2018?") whose plan lost the measure of the
            # analysis it continues: one re-plan with the continuation spelled out
            try:
                self._validate_plan(plan)
            except ValueError as e:
                self._fire("hook:replan_incomplete")
                try:
                    plan = generate_json(
                        f"{context}{question_block}\n\n{standard_block}{strata_block}VARIABLE CARDS:\n{self._cards(hits, named)}"
                        f"\n\nYOUR PREVIOUS PLAN WAS INCOMPLETE ({e}). This question continues the "
                        "analysis in the conversation: copy its template, measure(s), grouping and "
                        "filter from the previous exchange and change only what the question asks "
                        "(the cycle, the economy, the subject).",
                        system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS),
                                                  regions=self.regions_block,
                                                  levels=link_errors.levels_prompt()))
                except llm.LLMError:
                    pass
        clarify_text = str(plan.get("clarify") or "")
        if plan.get("action") == "clarify" and strata_block and \
                re.search(r"\b(stratum|strata|both|single table|one table|one output|separately)\b", clarify_text, re.I):
            # the planner declined to show an economy and its stratum together:
            # the app does exactly that — re-plan for the economy alone
            self._fire("hook:replan_stratum")
            try:
                plan = generate_json(
                    f"{context}{question_block}\n\n{standard_block}{strata_block}VARIABLE CARDS:\n{self._cards(hits, named)}"
                    "\n\nDO NOT CLARIFY: plan the statistic for the whole economy (by [\"CNT\"], where = "
                    "the economy); the app adds the stratum's own labelled row next to it.",
                    system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS),
                                              regions=self.regions_block,
                                              levels=link_errors.levels_prompt()))
            except llm.LLMError:
                pass
        if plan.get("action") == "clarify" and self.CANNOT_COMPUTE.search(str(plan.get("clarify") or "")) \
                and len(named) >= 2:
            # the planner declined a pairwise question the app answers from
            # its own statements: one re-plan with the rule spelled out
            self._fire("hook:replan_pairwise")
            try:
                plan = generate_json(
                    f"{context}{question_block}\n\n{standard_block}{strata_block}VARIABLE CARDS:\n{self._cards(hits, named)}"
                    "\n\nDO NOT CLARIFY: plan the statistic for every economy named, in one table "
                    f"(where = \"CNT IN ({', '.join(repr(c) for c in named)})\", by = [\"CNT\"]); the app "
                    "computes every pairwise difference, the change in a difference and ties itself.",
                    system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS),
                                              regions=self.regions_block,
                                              levels=link_errors.levels_prompt()))
            except llm.LLMError:
                pass
        plan["_question"] = q_en
        plan["_language"] = language
        self._fix_invented_codes(plan)
        self._apply_standards(plan, q_en)
        self._apply_strata(plan, q_en)
        self._fix_level_phrases(plan, q_en)
        if self.WORLD_AVERAGE_WORDS.search(q_en + " " + question) and plan.get("action") != "clarify":
            # "world average" is the average of every participant, never the OECD's
            groups = plan.get("include_average_of") or []
            groups = groups if isinstance(groups, list) else [groups]
            changed = False
            if not any(str(g).lower() in self.ALL_PARTICIPANTS for g in groups):
                groups = groups + ["All participants"]
                changed = True
            # judge "did the user say OECD" on the user's own words: the router's
            # English rendering sometimes glosses "world average" as the OECD's
            own_words = question if self._looks_english(question) else q_en
            if not re.search(r"\boecd|ocde\b", own_words, re.I):
                # the user asked for the world, not the OECD: one benchmark
                if plan.get("include_oecd_average") or any(str(g).lower() == "oecd" for g in groups):
                    changed = True
                plan["include_oecd_average"] = False
                groups = [g for g in groups if str(g).lower() != "oecd"]
            plan["include_average_of"] = groups
            if changed:
                self._fire("hook:world_average")
        self._verify_substitution_claims(plan)
        if plan.get("_claim_corrected"):
            self._fire("hook:verify_substitution_claims")
        if unsupported and plan.get("action") != "clarify":
            # a method the app does not offer, asked for by name: said in the
            # notes rather than silently replaced by something else
            self._fire("hook:unsupported_method")
            note = " ".join(unsupported)
            plan["limitation_note"] = (f"{plan['limitation_note']} {note}"
                                       if plan.get("limitation_note") else note)
        dropped = self._dropped_measures(q_en, plan)
        if dropped and plan.get("action") != "clarify":
            self._fire("hook:dropped_measures")
            note = ("Also asked but not computed in this answer: " + ", ".join(dropped)
                    + " — ask for them together (e.g. “math, reading and ESCS for …”) "
                      "or one at a time.")
            plan["limitation_note"] = (f"{plan['limitation_note']} {note}"
                                       if plan.get("limitation_note") else note)
        if plan.get("action") != "clarify":
            left_out = self._dropped_economies(q_en, plan)
            if left_out:
                self._fire("hook:dropped_economies")
                note = ("Economies named in the question but not in this table: "
                        f"{self._names(left_out)} — ask again naming them as rows, or as a "
                        "group average (e.g. “… with the EU average”).")
                plan["limitation_note"] = (f"{plan['limitation_note']} {note}"
                                           if plan.get("limitation_note") else note)
        if plan.get("action") == "clarify":
            text = self._plain(plan.get("clarify")) or "I need more detail to answer that."
            return AgentResult(question, self._localize(text, language),
                               plan=plan, retrieved=hits, route="clarify")

        # An economy the model invented a code for ("India" -> CNT = 'IND')
        # is answered from the participant lists, never as an engine error.
        every = set().union(*self.present.values()) if self.present else set()
        absent = [c for c in sorted(set(re.findall(r"'([A-Z]{3})'", str(plan.get("where") or ""))))
                  if c not in every]
        if absent and not any(c in every for c in re.findall(r"'([A-Z]{3})'", str(plan.get("where") or ""))):
            self._fire("hook:unknown_economy")
            names = ", ".join(self.economy_names.get(c, c) for c in absent)
            return AgentResult(question, self._localize(
                f"{names} is not in the PISA 2018, 2022 or 2025 public-use databases loaded "
                f"here (no code {', '.join(absent)} appears in any cycle), so no statistic can be "
                "computed. Ask “did <economy> participate?” to check any economy, or name one "
                "from the participant lists.", language), plan=plan, retrieved=hits,
                route="conversational")
        try:
            table, provenance = self.execute(plan)
            for key, name in (("_direction_fixed", "hook:align_gender_direction"),
                              ("_ranking_kept", "hook:keep_full_ranking"),
                              ("_null_groups", "hook:drop_null_groups"),
                              ("_non_comparable", "hook:blank_non_comparable"),
                              ("_empty_cycles", "hook:empty_cycles"),
                              ("_unavailable", "hook:unavailable_measures")):
                if plan.get(key):
                    self._fire(name)
            if any(f["blocked"] for fs in (plan.get("_coverage") or {}).values() for f in fs):
                self._fire("hook:coverage_block")
        except CoverageError as e:
            self._fire("hook:coverage_block")
            answer = str(e)
            findings = plan.get("_coverage") or {}
            lacking = {c for fs in findings.values() for f in fs for c in f["named_missing"]}
            blocked_vars = {f["variable"] for fs in findings.values() for f in fs if f["blocked"]}
            alts = self._coverage_alternatives(hits, lacking, blocked_vars)
            if alts:
                answer += (f" Variables on this topic that {self._names(sorted(lacking))} "
                           f"did collect: {'; '.join(alts)}. Ask again naming one of them.")
            else:
                answer += (" None of the variables retrieved for this topic has data "
                           f"for {self._names(sorted(lacking))}.")
            return AgentResult(question, self._localize(answer, language), plan=plan,
                               retrieved=hits, provenance=e.provenance,
                               notes=(e.provenance or {}).get("notes", []),
                               route="coverage")
        except ValueError as e:
            if "terms of use" in str(e):
                self._fire("hook:raw_sql_refused")
                return AgentResult(question, self._localize(
                    "PISA Explorer returns aggregate statistics only. Student- and school-level "
                    "records, identifiers and raw rows are never shown: the OECD's terms of use "
                    "for the public-use files forbid redistributing the dataset, and the app "
                    "protects individual schools' responses. Ask for a statistic (a mean, share, "
                    "gap, correlation or ranking) and it will be computed with the official "
                    "weights.", language), plan=plan, retrieved=hits, route="conversational")
            return AgentResult(question, f"The analysis failed: {e}",
                               plan=plan, retrieved=hits, error=str(e), route="error")
        except Exception as e:
            return AgentResult(question, f"The analysis failed: {e}",
                               plan=plan, retrieved=hits, error=str(e), route="error")

        shown, truncation, focus = self._summary_view(table, q_en, history)
        legend = self._country_legend(shown)
        if focus and "rank" in table.columns:
            positions = [line for line in focus.splitlines()[1:] if "rank" in line]
            if positions:
                provenance["notes"].append(
                    "Position(s) located by the app in the ranked table: "
                    + "; ".join(positions) + ".")
        sample_line = "; ".join(
            f"{smp['table']}: {smp['students']:,} students sampled"
            + (f" in {smp['schools']:,} schools" if smp.get("schools") else "")
            + (f", representing {smp['weighted_students']:,} 15-year-olds (sum of weights)"
               if smp.get("weighted_students") else "")
            for smp in provenance.get("sample") or [] if smp.get("students") is not None)
        if self.CI_WORDS.search(q_en):
            # a confidence interval asked for: computed here (estimate ± 1.96 SE),
            # so the summary never has to do the arithmetic
            for est_col in [c for c in table.columns if ESTIMATE_COL.match(c) or c == "change"]:
                se_col = summ.SE_FOR.get(est_col) or est_col.replace("estimate", "se")
                if se_col in table.columns and f"ci95_low{est_col.replace('estimate', '')}" not in table.columns:
                    suffix = "" if est_col == "estimate" else ("_change" if est_col == "change" else est_col.replace("estimate", ""))
                    table[f"ci95_low{suffix}"] = (table[est_col] - summ.Z * table[se_col]).round(2)
                    table[f"ci95_high{suffix}"] = (table[est_col] + summ.Z * table[se_col]).round(2)
            provenance["notes"].append("95% confidence intervals = estimate ± 1.96 × SE (ci95_low, ci95_high columns).")
            self._fire("hook:ci_columns")
        sample_block = (f"SAMPLE SIZES (within the question's filter): {sample_line}" + chr(10)
                        if sample_line else "")
        focus_codes = self._economies_mentioned(
            q_en + " " + " ".join(h.get("question", "") for h in (history or [])[-2:]),
            table["CNT"].astype(str).unique()) if "CNT" in table.columns else []
        label_tables = [f"{plan.get('instrument') or 'stu_qqq'}_{c}"
                        for c in sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]}, reverse=True)]

        def category_label(col, value):
            # the label from whichever cycle's codebook has it (a 2025 row
            # recoded to the 2022 gender column still gets "Female")
            for tbl in label_tables:
                text = self._category_label(col, value, tbl)
                if "(" in text:
                    return text
            return text

        facts = summ.fact_sentences(table, plan, provenance, self.economy_names,
                                    self._measure_label, focus_codes=focus_codes,
                                    category_label=category_label,
                                    low_coverage=plan.get("_low_coverage"))
        provenance["facts"] = facts
        summary_prompt = (
            f"QUESTION: {question}\n"
            f"PLANNED: {plan.get('explanation')}\n"
            f"METHOD: {provenance['method']}\n"
            f"{sample_block}"
            f"NOTES: {'; '.join(provenance['notes']) or 'none'}\n"
            f"VERIFIED STATEMENTS (computed by the app — every number, rank, change and "
            f"significance verdict in your answer must come from these or the table; "
            f"never compute a new number):\n- " + "\n- ".join(facts) + "\n"
            f"{legend}{focus}{truncation}"
            f"RESULT TABLE (CSV):\n{shown.to_csv(index=False)}"
        )
        answer, mode, issues = self._summarize(summary_prompt, facts, table, provenance,
                                               plan, q_en, language)
        provenance["summary_mode"] = mode
        modes = plan.get("_link_error_modes") or {}
        if modes:
            self._last_link_mode = next(iter(modes.values()))
        return AgentResult(question, answer.strip(), plan=plan, table=table,
                           provenance=provenance, retrieved=hits,
                           notes=provenance["notes"], summary_mode=mode,
                           prose_issues=issues)

    def _allowed_codes(self, table: pd.DataFrame, plan: dict, provenance: dict,
                       question: str) -> set[str]:
        """Economies a summary may name: result rows, the question, the
        filter, and anything the app's own notes mention."""
        codes = set()
        if "CNT" in table.columns:
            codes |= {c for c in table["CNT"].astype(str) if not c.endswith(" avg")}
        codes |= set(re.findall(r"'([A-Z]{3})'", str(plan.get("where") or "")))
        codes |= set(self._economies_in_data(question))
        codes |= set(self._economies_in_data(" ".join(provenance.get("notes") or [])))
        return codes

    def _summarize(self, prompt: str, facts: list[str], table: pd.DataFrame,
                   provenance: dict, plan: dict, question: str,
                   language: str) -> tuple[str, str, list[str]]:
        """The model phrases the app's verified statements; its draft is
        checked against the result (explorer/summary.py); one retry naming
        the problems; then the statements themselves are the answer."""
        allowed = self._allowed_codes(table, plan, provenance, question)
        lang_rule = ("" if language.lower() == "english"
                     else f"\nAnswer in {language} (the user's language); keep codes and numbers as they are.")
        every = set().union(*self.present.values()) if self.present else set()

        def mentioned(text):
            return self._economies_mentioned(text, sorted(every))

        def check(text):
            return summ.check_prose(text, table, provenance, plan, question,
                                    mentioned, allowed, language=language)

        first_issues: list[str] = []
        try:
            draft = self._plain(generate(prompt, system=SUMMARY_SYSTEM + lang_rule)).strip()
            first_issues = check(draft)
            if not first_issues and draft:
                return draft, "llm", []
            self._fire("summary:retry")
            retry_prompt = (prompt + "\n\nYOUR PREVIOUS DRAFT WAS REJECTED for these reasons: "
                            + "; ".join(first_issues) +
                            ".\nRewrite it using ONLY the verified statements and the table: "
                            "copy numbers exactly, name only economies in the result, and "
                            "state significance only as the verified statements do.")
            draft = self._plain(generate(retry_prompt, system=SUMMARY_SYSTEM + lang_rule)).strip()
            if draft and not check(draft):
                return draft, "llm-retry", first_issues
        except llm.LLMError as e:
            first_issues = first_issues or [f"LLM error: {e}"]
        self._fire("summary:app_authored")
        answer = " ".join(facts)
        samples = [s for s in provenance.get("sample") or [] if s.get("students") is not None]
        if samples:
            answer += " Sample: " + "; ".join(
                f"{s['students']:,} students" + (f" in {s['schools']:,} schools" if s.get("schools") else "")
                + f" ({s['table']})" for s in samples) + "."
        if provenance.get("notes"):
            answer += " Notes: " + " ".join(provenance["notes"][:3])
        return self._localize(answer, language), "app-authored", first_issues
