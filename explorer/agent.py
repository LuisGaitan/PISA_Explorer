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

from . import catalog, llm
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

MAX_CARDS = 40
MAX_RESULT_ROWS = 500

INSTRUMENTS = ["stu_qqq", "sch_qqq", "tch_qqq", "stu_cog", "stu_tim", "stu_ttm",
               "flt_qqq", "flt_cog", "flt_tim", "crt_cog", "ldw_cog"]
CYCLES = ["2018", "2022", "2025"]
DEFAULT_CYCLE = "2025"          # a question that names no cycle means the latest
SOURCE_LINE = ("OECD PISA public-use databases: 2018 (CY07MSU), 2022 (CY08MSP), "
               "2025 (CY09MS)")
ESTIMATE_COL = re.compile(r"^estimate(_\d{4})?$")

FORBIDDEN_SQL = re.compile(
    r"\b(copy|attach|detach|install|load|pragma|export|import|create|insert|"
    r"update|delete|alter|drop|call|set|reset)\b", re.IGNORECASE)

TERMS_SYSTEM = """You are the router inside PISA Explorer, a local app where a
user chats with the OECD PISA 2018/2022/2025 databases. How the app works (answer
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
"search_terms": [str, ...], "direct_answer": str|null}.
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
stu_qqq_2025 (instruments: {instruments}; cycles: 2018, 2022, 2025). Student
questionnaire (stu_qqq_*) holds achievement plausible values PV1..PV10 for
MATH/READ/SCIE, final weight W_FSTUWT, replicate weights, ESCS (socio-economic
index), and CNT (ISO-3 country code, e.g. 'USA', 'KOR', 'DEU').
There is also escs_trend (OECD comparable-ESCS across cycles).

ECONOMY CODES that are NOT ISO-3 (use exactly these): QCI = B-S-J-Z (China:
Beijing, Shanghai, Jiangsu, Zhejiang; 2018 and 2025 only), TAP = Chinese
Taipei, MAC = Macao (China), HKG = Hong Kong (China), KSV = Kosovo, QAT =
Qatar, QAZ = Baku (Azerbaijan; 2018/2022) vs AZE = Azerbaijan (2025), QUR =
Ukrainian regions (2022) / QUA = Ukrainian regions (2025), QKI = Kurdistan
Region (Iraq; 2025), QTJ = Dushanbe (Tajikistan; 2025), QMR = Moscow and QRT =
Tatarstan (2018), RUS = Russia (2018). Never invent codes: an economy is a
single CNT value. Not every economy is in every cycle (e.g. ARM, KEN, KGZ,
MUS, RWA, ZMB, ECU joined in 2025; JAM only 2022; BIH, BLR, UKR only 2018).

GENDER: ST004D01T (1=Female, 2=Male) in 2018 and 2022. In 2025 fourteen
economies (ARG, AUS, BEL, CAN, CHL, COL, DEU, DNK, ESP, IRL, ISL, NLD, NZL,
URY) release only the derived flag MALE (1=Male, 0=Female/Other), which is
complete for all 90 economies — so for 2025 use MALE (gap: group_col="MALE",
minuend=1, subtrahend=0; dummy: "CASE WHEN MALE = 1 THEN 1 ELSE 0 END"). In a
plan that spans 2018/2022 AND 2025, keep ST004D01T in the main fields and put
the 2025 variant in cycle_overrides (see below).

PISA 2025 SPECIFICS: 90 economies (80 in 2018/2022). Science was the major
domain; the same proficiency-level cutoffs apply in every cycle (the scales are
linked). Uzbekistan (UZB) has science PVs only in 2025 (no MATH, no READ —
its rows show no estimate for those domains). Extra 2025 PVs:
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
 "by": [grouping columns, usually ["CNT"]],
 "where": SQL boolean filter or null, e.g. "CNT IN ('USA','KOR')",
 "sql": str|null  (raw_sql only: one read-only SELECT; use this ONLY when no
        template fits — raw SQL gets no automatic weighting/PV/BRR treatment),
 "sort_by": "estimate"|"change"|null, "sort_desc": bool, "top_n": int|null
        (for ranking/top-N/bottom-N questions: the system sorts the result and
        keeps top_n rows, so the reported rows ARE the answer),
 "substitution_note": str|null,
 "explanation": one sentence of what will be computed
}}

Reply with EXACTLY ONE JSON object — never a JSON array, never multiple plans.
A request to compare the averages of two DIFFERENT variables side by side has
no template yet: either action="clarify" asking which variable to analyze
first, or (if the user insists on both at once) raw_sql computing both
weighted means per group as sum(W_FSTUWT * var) / sum(W_FSTUWT) — and note in
the explanation that raw SQL carries no standard errors.

Rules: comparisons across cycles => list every cycle asked about, in
chronological order ("over time" / "trend" / "since 2018" => all three unless
the user narrows it); the system runs the template per cycle, reports every
cycle side by side, and adds the change from the FIRST to the LAST listed
cycle. A question that names no cycle means the latest one (2025) — say so in
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
"CASE WHEN PV{{pv}}MATH < 420.07 THEN 100.0 ELSE 0.0 END"
(Level 2 lower bounds — math 420.07, reading 407.47, science 409.54;
 Level 5 lower bounds — math 606.99, reading 625.61, science 633.33)."""

SUMMARY_SYSTEM = """You summarize PISA analysis results for a general audience.
Write 2-5 sentences. Cite the key numbers with their standard errors like
"465 points (SE 4.0)". Treat |estimate| > 1.96*SE as statistically significant
and say so in plain language. In a multi-cycle table the columns are
estimate_<cycle> per cycle and `change` = last cycle minus first cycle;
describe the path over time (e.g. 2018 → 2022 → 2025), not just the endpoints.
All three cycles — including PISA 2025, released in 2026 — are real, published
survey results: never describe 2025 figures as projections, forecasts or
expectations. If a substitution_note or method note is present,
state it plainly. Do not invent numbers not in the table."""


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
            "oecd_average": bool(plan.get("include_oecd_average")),
            "missing_rows": any("no estimate" in n for n in prov.get("notes", [])),
            "error": self.error,
            "answer_chars": len(self.answer or ""),
            **self.timing,
        }


class Agent:
    def __init__(self):
        self.con = connect(read_only=True)

    # ---------- retrieval ----------

    def _retrieve(self, terms: list[str], per_term: int = 8) -> pd.DataFrame:
        """Top-scoring VARIABLES across the search terms, with every table
        (cycle/instrument) row of each — capped at MAX_CARDS variables."""
        # The router lists the user's own concept first and broader synonyms
        # after it; weight them so a synonym ("school climate") cannot outrank
        # the word the user actually used ("bullying"), while a variable that
        # matches several terms still gets a bonus.
        frames = []
        for i, term in enumerate(terms):
            frame = catalog.search(term, limit=per_term)
            frames.append(frame.assign(score=frame.score * (1.0 if i == 0 else 0.6),
                                       _term=i))
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

    @staticmethod
    def _cards(hits: pd.DataFrame) -> str:
        """One card per variable, listing every table it exists in — so the
        planner sees cross-cycle availability at a glance (and a variable
        absent from a cycle is visibly absent)."""
        if hits is None or hits.empty:
            return "(no extra variables retrieved)"
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
        return "\n".join(lines)

    # ---------- execution (offline-testable, no LLM) ----------

    # Fields a template cannot run without. A plan that lacks one used to reach
    # DuckDB as the literal word "None" (Binder Error: column "None") — now it
    # is refused with a message that names the gap.
    REQUIRED_FIELDS = {
        "weighted_mean": [("measure",)],
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

    def execute(self, plan: dict) -> tuple[pd.DataFrame, dict]:
        self._validate_plan(plan)
        template = plan.get("template")
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        unknown = [c for c in cycles if c not in CYCLES]
        if unknown:
            raise ValueError(f"unknown cycle(s) {unknown}; available: {CYCLES}")
        instrument = plan.get("instrument") or "stu_qqq"
        by = tuple(plan.get("by") or ())
        where = plan.get("where") or None
        self._check_fragment(where)
        for col in by:
            self._check_identifier(col)
        overrides = {str(k): v for k, v in (plan.get("cycle_overrides") or {}).items()
                     if isinstance(v, dict) and v}
        for ov in overrides.values():
            for value in ov.values():
                if isinstance(value, str):
                    self._check_fragment(value)

        if template == "raw_sql":
            table = self._run_raw_sql(plan["sql"])
            prov = self._provenance(plan, [], raw=True)
            return table, prov

        tables = [f"{instrument}_{c}" for c in cycles]
        per_cycle: dict[str, pd.DataFrame] = {}
        for cycle, tbl in zip(cycles, tables):
            cplan = {**plan, **overrides.get(cycle, {})}
            cwhere = cplan.get("where") or None
            res = self._run_template(template, cplan, tbl, by, cwhere)
            if plan.get("include_oecd_average") and "CNT" in by:
                avg = self._oecd_average_rows(res, tbl)
                if avg is not None:
                    res = pd.concat([res, avg], ignore_index=True)
            per_cycle[cycle] = res
        if overrides and "contrast" in per_cycle[cycles[0]].columns:
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
        if len(cycles) >= 2:
            table = trend(per_cycle, by=[c for c in by] + extra_keys)
        else:
            table = per_cycle[cycles[0]].assign(cycle=cycles[0])

        sort_by = plan.get("sort_by")
        if sort_by:
            candidates = [sort_by, f"estimate_{cycles[-1]}" if sort_by == "estimate" else sort_by]
            col = next((c for c in candidates if c in table.columns), None)
            if col:
                table = table.sort_values(col, ascending=not plan.get("sort_desc", True))
        if plan.get("top_n"):
            table = table.head(int(plan["top_n"]))
        table = table.reset_index(drop=True)

        prov = self._provenance(plan, tables)
        estimate_cols = [c for c in table.columns if ESTIMATE_COL.match(c)]
        n_missing = int(table[estimate_cols].isna().any(axis=1).sum()) \
            if estimate_cols else 0
        if n_missing:
            prov["notes"].append(
                f"{n_missing} row(s) have no estimate — the variable was not "
                "administered (or has no valid responses) for those groups; "
                "they are listed in the table but excluded from the chart.")
        return table, prov

    def _oecd_average_rows(self, res: pd.DataFrame, tbl: str) -> pd.DataFrame | None:
        """Official OECD-average convention: the UNWEIGHTED mean of member
        countries' estimates (each country counts equally — not a pooled
        student-weighted mean, which would overweight populous countries).
        Countries are independent samples, so SE = sqrt(sum SE_i^2) / N."""
        members = {r[0] for r in self.con.sql(
            f"SELECT DISTINCT CNT FROM {tbl} WHERE OECD = 1").fetchall()}
        sub = res[res["CNT"].isin(members)]
        if sub.empty:
            return None
        id_cols = [c for c in res.columns
                   if c not in ("CNT", "estimate", "se", "n_pv")]

        def agg(group: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "estimate": group["estimate"].mean(),
                "se": float(np.sqrt((group["se"] ** 2).sum()) / len(group)),
                "n_pv": int(group["n_pv"].iloc[0]),
            })

        if id_cols:
            avg = (sub.groupby(id_cols, dropna=False, observed=True)
                   .apply(agg, include_groups=False).reset_index())
        else:
            avg = agg(sub).to_frame().T
        avg["n_pv"] = avg["n_pv"].astype(int)
        avg.insert(0, "CNT", "OECD avg")
        return avg

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
            return weighted_proportion(
                self.con, tbl, plan["variable"], plan["value"], by=by,
                where=where, valid_values=plan.get("valid_values"))
        if template == "gap":
            self._check_identifier(plan["group_col"])
            return gap(self.con, tbl, plan["measure"], plan["group_col"],
                       plan["minuend"], plan["subtrahend"], by=by, where=where)
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
            desc = desc[desc.table_name == table]
            if desc.empty or not desc.iloc[0].value_labels:
                continue
            labels = json.loads(desc.iloc[0].value_labels)
            mapped = res[code_col].map(
                lambda v: labels.get(str(int(v)) if isinstance(v, float)
                                     and float(v).is_integer() else str(v)))
            if mapped.notna().any():
                res.insert(res.columns.get_loc(code_col) + 1, label_col, mapped)
        return res

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
        fields = ("measure", "variable", "group_col", "where", "by",
                  "quart_variable", "x", "y", "row_var", "col_var", "predictors")
        parts = [plan] + [ov for ov in (plan.get("cycle_overrides") or {}).values()
                          if isinstance(ov, dict)]
        text = " ".join(str(part.get(k) or "") for part in parts for k in fields)
        text = text.replace("{pv}", "1")  # PV{pv}MATH -> PV1MATH (stands for all 10)
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        tokens |= {"W_FSTUWT"}
        out, seen = [], set()
        for token in sorted(tokens):
            desc = catalog.describe(token)
            desc = desc[desc.table_name.isin(tables)] if tables else desc
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
            "weighted_proportion": "Weighted percentage of valid respondents "
                                   "(W_FSTUWT). SE: Fay's BRR, k=0.5, 80 replicates.",
            "gap": "Group difference computed replicate-wise (correct covariance). "
                   "Weighted by W_FSTUWT; SE: Fay's BRR, k=0.5, 80 replicates.",
            "quartile_means": "Weighted mean per weighted quarter of "
                              f"{plan.get('quart_variable')} (quartiles computed with "
                              "W_FSTUWT within each group, OECD convention; 1=bottom, "
                              "4=top). SE: Fay's BRR, k=0.5, 80 replicates.",
            "quartile_gap": "Top minus bottom weighted quarter of "
                            f"{plan.get('quart_variable')} (quartiles computed with "
                            "W_FSTUWT within each group), differenced replicate-wise. "
                            "SE: Fay's BRR, k=0.5, 80 replicates.",
            "correlation": "Weighted Pearson correlation (W_FSTUWT). "
                           "SE: Fay's BRR, k=0.5, 80 replicates.",
            "percentiles": "Weighted empirical percentiles (W_FSTUWT). "
                           "SE: Fay's BRR, k=0.5, 80 replicates.",
            "percentile_spread": "Difference of weighted percentiles "
                                 "(within-group dispersion), differenced "
                                 "replicate-wise. SE: Fay's BRR, k=0.5, 80 replicates.",
            "crosstab": "Weighted two-way table: row percentages of valid "
                        "respondents (W_FSTUWT), each cell with its own "
                        "Fay-BRR SE (k=0.5, 80 replicates).",
            "regression": "Weighted least-squares regression (W_FSTUWT), "
                          "listwise deletion. SE: Fay's BRR, k=0.5, 80 "
                          "replicates. Association, not causation.",
            "raw_sql": "Direct SQL — NO automatic weighting/PV/BRR treatment; "
                       "results are not population estimates unless the query weights them.",
        }
        if template == "weighted_proportion" and not plan.get("variable") \
                and plan.get("measure"):
            template = "weighted_mean"      # threshold share ran as a 0/100 mean
        method = methods.get(template, "")
        pv_fields = " ".join(str(plan.get(k) or "") for k in ("measure", "x", "y"))
        if "{pv}" in pv_fields:
            method += " Point estimate averaged over 10 plausible values (Rubin's rules)."
        notes = []
        if plan.get("include_oecd_average"):
            notes.append('The "OECD avg" row follows the official convention: '
                         "the unweighted mean of OECD member countries' "
                         "estimates (SE = sqrt(sum of country SEs squared) / N).")
        if plan.get("substitution_note"):
            notes.append(f"VARIABLE SUBSTITUTION: {plan['substitution_note']}")
        cycles = sorted({str(c) for c in plan.get("cycles") or [DEFAULT_CYCLE]})
        if len(cycles) >= 2:
            first, last = cycles[0], cycles[-1]
            method += (f" Cross-cycle change = {last} minus {first}: independent "
                       f"samples, SE = sqrt(SE{first[2:]}^2 + SE{last[2:]}^2).")
            notes.append("Trend SEs exclude the OECD link error (scale equating "
                         "uncertainty); they are slightly understated.")
        for cycle, ov in sorted((plan.get("cycle_overrides") or {}).items()):
            if isinstance(ov, dict) and ov:
                fields = ", ".join(f"{k} = {v}" for k, v in ov.items())
                notes.append(f"PISA {cycle} uses a per-cycle variable override: "
                             f"{fields} (the variable differs in that cycle; "
                             "other cycles use the main plan fields).")
        unknown = self._unknown_country_codes(where, tables)
        if unknown:
            notes.append("The filter names economy code(s) that exist in none of "
                         f"the queried tables: {', '.join(unknown)} — those rows "
                         "are absent, not zero. Check the code (e.g. B-S-J-Z "
                         "(China) is QCI, Chinese Taipei is TAP).")

        sample = []
        for tbl in tables:
            try:
                clause = f" WHERE {where}" if where else ""
                n, wsum = self.con.sql(
                    f"SELECT count(*), sum(W_FSTUWT) FROM {tbl}{clause}").fetchone()
                sample.append({"table": tbl, "students": int(n),
                               "weighted_students": round(wsum) if wsum else None})
            except Exception:
                sample.append({"table": tbl})

        return {
            "source": SOURCE_LINE,
            "tables": tables or ["(raw SQL — see query)"],
            "variables": self._referenced_variables(plan, tables),
            "filter": where or "none (all rows)",
            "method": method.strip(),
            "sample": sample,
            "sql": plan.get("sql") if raw else None,
            "notes": notes,
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

    def _explore(self, question: str, terms: list[str]) -> AgentResult:
        """Answer 'what data is there about X' from the catalog itself: one
        row per matching variable with its label, the tables (cycles and
        instruments) it exists in, and its response codes. Deterministic —
        no plan, no statistic, no LLM summary that could invent variables."""
        hits = self._retrieve(terms, per_term=12)
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
            answer = (f"No catalog variables matched “{topic}” in {scope}. Try "
                      "other words for the construct (PISA labels use the OECD's "
                      "wording, e.g. “sense of belonging”, “bullying”, “life "
                      "satisfaction”).")
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
            "sample": [], "sql": None, "notes": [],
        }
        return AgentResult(question, answer, plan=plan, table=table,
                           provenance=provenance, retrieved=hits, route="explore")

    @staticmethod
    def _country_legend(shown: pd.DataFrame) -> str:
        """CNT code -> economy name for the codes in the shown table, so the
        summary can say 'B-S-J-Z (China)' instead of the bare code 'QCI'."""
        if "CNT" not in shown.columns:
            return ""
        codes = [c for c in shown["CNT"].dropna().astype(str).unique() if c != "OECD avg"]
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
        started = time.time()
        try:
            result = self._ask(question, history)
        except Exception as e:  # noqa: BLE001 — surface as a result, not a crash
            result = AgentResult(question, f"Something went wrong: {e}",
                                 error=str(e), route="error")
        result.timing = {"total_ms": round((time.time() - started) * 1000),
                         **llm.stats()}
        return result

    def _ask(self, question: str, history: list | None) -> AgentResult:
        context = self._transcript(history)
        route = generate_json(f"{context}Question: {question}", system=TERMS_SYSTEM)
        if not route.get("data_question"):
            if self.VIZ_WORDS.search(question):
                return AgentResult(question, self.VIZ_ANSWER, route="conversational")
            return AgentResult(question, route.get("direct_answer")
                               or "Could you rephrase that?", route="conversational")

        if route.get("intent") == "explore":
            return self._explore(question, route.get("search_terms") or [])

        hits = self._retrieve(route.get("search_terms") or [])
        plan = generate_json(
            f"{context}QUESTION: {question}\n\nVARIABLE CARDS:\n{self._cards(hits)}",
            system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS)),
        )
        if plan.get("action") == "clarify":
            return AgentResult(question, plan.get("clarify")
                               or "I need more detail to answer that.",
                               plan=plan, retrieved=hits, route="clarify")

        try:
            table, provenance = self.execute(plan)
        except Exception as e:
            return AgentResult(question, f"The analysis failed: {e}",
                               plan=plan, retrieved=hits, error=str(e), route="error")

        shown = table.head(30).round(2)
        legend = self._country_legend(shown)
        truncation = (
            f"WARNING: the table has {len(table)} rows but only the first 30 are "
            f"shown below. If the question needs rows beyond these (rankings, "
            f"extremes, totals), say the full table is in the result — NEVER "
            f"answer it from this partial view.\n"
        ) if len(table) > 30 else ""
        summary_prompt = (
            f"QUESTION: {question}\n"
            f"PLANNED: {plan.get('explanation')}\n"
            f"METHOD: {provenance['method']}\n"
            f"NOTES: {'; '.join(provenance['notes']) or 'none'}\n"
            f"{legend}{truncation}"
            f"RESULT TABLE (CSV):\n{shown.to_csv(index=False)}"
        )
        answer = generate(summary_prompt, system=SUMMARY_SYSTEM)
        return AgentResult(question, answer.strip(), plan=plan, table=table,
                           provenance=provenance, retrieved=hits,
                           notes=provenance["notes"])
