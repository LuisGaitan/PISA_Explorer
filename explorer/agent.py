"""The conversational layer: natural-language question -> catalog retrieval ->
Gemini fills an analysis plan -> survey-correct execution -> summary.

Design rules (each traces to a defect in the old prototype, see handoff §4):
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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import catalog
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

INSTRUMENTS = ["stu_qqq", "sch_qqq", "tch_qqq", "stu_cog", "stu_tim",
               "flt_qqq", "flt_cog", "flt_tim", "crt_cog"]

FORBIDDEN_SQL = re.compile(
    r"\b(copy|attach|detach|install|load|pragma|export|import|create|insert|"
    r"update|delete|alter|drop|call|set|reset)\b", re.IGNORECASE)

TERMS_SYSTEM = """You are the router inside PISA Explorer, a local app where a
user chats with the OECD PISA 2018/2022 databases. How the app works (answer
questions about it truthfully): the user's question becomes an analysis plan;
validated survey statistics run locally; the app itself then AUTOMATICALLY
renders a chart when the result has a chartable shape (group comparisons ->
bar chart with 95% CI whiskers; 2018-vs-2022 -> dumbbell chart; group gaps ->
diverging bars), plus the data table, a CSV export button, and a provenance
card. Single-number results and raw-SQL results get a table but no chart — to
get a chart, the user should ask for a comparison across groups, countries, or
cycles. Never claim "I cannot visualize data": charts appear automatically for
comparison-shaped results.

Reply with JSON: {"data_question": bool, "search_terms": [str, ...], "direct_answer": str|null}.
If the question needs data — including a follow-up that continues or answers a
clarification from the conversation context — set data_question=true and give
2-6 short catalog search terms (constructs, topics, variable ideas — e.g.
"gender", "mathematics", "socioeconomic status", "bullying", "immigrant").
If it is conversational, about PISA in general, or about this app's abilities
(no data needed), set data_question=false and write direct_answer.

direct_answer speaks AS the app ("PISA Explorer"), never as a language model:
never say "I am an AI / text-based model", never mention routing or search
terms. Example:
  User: "can you show me a visualization / chart of that?"
  -> {"data_question": false, "search_terms": [],
      "direct_answer": "Charts appear automatically when a result compares
      groups: country or group comparisons draw bars with confidence whiskers,
      2018-vs-2022 questions draw dumbbells, and gaps draw diverging bars.
      The last result was a single number, which has no chart form — ask for a
      comparison instead (for example: 'compare reading scores for students
      with high vs low ICT autonomy, top vs bottom quartile, by country') and
      the chart will render with it."}
A follow-up like "yes, top vs bottom quartile" that answers a clarifying
question IS a data question — combine it with the context and route it to data."""

PLAN_SYSTEM = """You plan analyses of the OECD PISA 2018 and 2022 databases.

DATABASE: DuckDB. Tables are <instrument>_<cycle>, e.g. stu_qqq_2018, stu_qqq_2022
(instruments: {instruments}). Student questionnaire (stu_qqq_*) holds achievement
plausible values PV1..PV10 for MATH/READ/SCIE, final weight W_FSTUWT, replicate
weights, ESCS (socio-economic index), and CNT (ISO-3 country code, e.g. 'USA',
'KOR', 'DEU'). Gender is ST004D01T (1=Female, 2=Male) in both cycles.
There is also escs_trend (OECD comparable-ESCS across cycles).

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
 "cycles": ["2018","2022"] or ["2022"] or ["2018"],
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

Rules: comparisons across cycles => cycles=["2018","2022"] (the system runs the
template per cycle and differences them). Achievement questions always use the
PV{{pv}} form. Filter to the countries the user names; if none named, ask
yourself whether all 80 economies is really wanted — for rankings it is.
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
and say so in plain language. If a substitution_note or method note is present,
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


class Agent:
    def __init__(self):
        self.con = connect(read_only=True)

    # ---------- retrieval ----------

    def _retrieve(self, terms: list[str]) -> pd.DataFrame:
        frames = [catalog.search(t, limit=8) for t in terms]
        hits = (pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["variable", "table_name"])
                .sort_values("score", ascending=False)
                .head(MAX_CARDS)) if frames else pd.DataFrame()
        return hits

    @staticmethod
    def _cards(hits: pd.DataFrame) -> str:
        if hits is None or hits.empty:
            return "(no extra variables retrieved)"
        lines = []
        for _, r in hits.iterrows():
            desc = catalog.describe(r.variable, cycle=r.cycle)
            values = ""
            if not desc.empty and desc.iloc[0].value_labels:
                labels = json.loads(desc.iloc[0].value_labels)
                shown = [f"{k}={v}" for k, v in list(labels.items())[:6]]
                values = f" [values: {', '.join(shown)}]"
            lines.append(f"- {r.variable} ({r.table_name}): {r.label}{values}")
        return "\n".join(lines)

    # ---------- execution (offline-testable, no LLM) ----------

    def execute(self, plan: dict) -> tuple[pd.DataFrame, dict]:
        template = plan.get("template")
        cycles = [str(c) for c in plan.get("cycles") or ["2022"]]
        instrument = plan.get("instrument") or "stu_qqq"
        by = tuple(plan.get("by") or ())
        where = plan.get("where") or None
        self._check_fragment(where)
        for col in by:
            self._check_identifier(col)

        if template == "raw_sql":
            table = self._run_raw_sql(plan["sql"])
            prov = self._provenance(plan, [], raw=True)
            return table, prov

        tables = [f"{instrument}_{c}" for c in cycles]
        per_cycle: dict[str, pd.DataFrame] = {}
        for cycle, tbl in zip(cycles, tables):
            res = self._run_template(template, plan, tbl, by, where)
            if plan.get("include_oecd_average") and "CNT" in by:
                avg = self._oecd_average_rows(res, tbl)
                if avg is not None:
                    res = pd.concat([res, avg], ignore_index=True)
            per_cycle[cycle] = res

        extra_keys = {"gap": ["contrast"], "quartile_gap": ["contrast"],
                      "quartile_means": ["quarter"],
                      "percentiles": ["percentile"],
                      "percentile_spread": ["contrast"],
                      "crosstab": ["row", "col", "row_label", "col_label"],
                      "regression": ["term"]}.get(template, [])
        if template == "crosstab":
            extra_keys = [k for k in extra_keys
                          if k in per_cycle[cycles[0]].columns]
        if len(cycles) == 2:
            table = trend(per_cycle["2018"], per_cycle["2022"],
                          by=[c for c in by] + extra_keys)
        else:
            table = per_cycle[cycles[0]].assign(cycle=cycles[0])

        sort_by = plan.get("sort_by")
        if sort_by:
            candidates = [sort_by, "estimate_2022" if sort_by == "estimate" else sort_by]
            col = next((c for c in candidates if c in table.columns), None)
            if col:
                table = table.sort_values(col, ascending=not plan.get("sort_desc", True))
        if plan.get("top_n"):
            table = table.head(int(plan["top_n"]))
        table = table.reset_index(drop=True)

        prov = self._provenance(plan, tables)
        estimate_cols = [c for c in ("estimate", "estimate_2018", "estimate_2022")
                         if c in table.columns]
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
        if template == "weighted_proportion":
            return weighted_proportion(
                self.con, tbl, plan["variable"], plan["value"], by=by,
                where=where, valid_values=plan.get("valid_values"))
        if template == "gap":
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
        text = " ".join(str(plan.get(k) or "") for k in
                        ("measure", "variable", "group_col", "where", "by",
                         "quart_variable", "x", "y", "row_var", "col_var",
                         "predictors"))
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
        if len(plan.get("cycles") or []) == 2:
            method += (" Cross-cycle change: independent samples, "
                       "SE = sqrt(SE18^2 + SE22^2).")
            notes.append("Trend SEs exclude the OECD link error (scale equating "
                         "uncertainty); they are slightly understated.")

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
            "source": "OECD PISA public-use databases: 2018 (CY07MSU), 2022 (CY08MSP)",
            "tables": tables or ["(raw SQL — see query)"],
            "variables": self._referenced_variables(plan, tables),
            "filter": where or "none (all rows)",
            "method": method.strip(),
            "sample": sample,
            "sql": plan.get("sql") if raw else None,
            "notes": notes,
        }

    # ---------- the full loop ----------

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
        "2018-vs-2022 questions draw dumbbell charts, and group gaps draw "
        "diverging bars. A single-number result (like a lone correlation or one "
        "average) has no chart form, so only its table is shown. To get a chart, "
        "ask for a comparison — across countries, groups (gender, immigrant "
        "background, grade repetition…), or the two cycles. For example: "
        "“compare reading scores for boys and girls in Spain, France and "
        "Germany in 2022”."
    )

    def ask(self, question: str, history: list | None = None) -> AgentResult:
        context = self._transcript(history)
        route = generate_json(f"{context}Question: {question}", system=TERMS_SYSTEM)
        if not route.get("data_question"):
            if self.VIZ_WORDS.search(question):
                return AgentResult(question, self.VIZ_ANSWER)
            return AgentResult(question, route.get("direct_answer")
                               or "Could you rephrase that?")

        hits = self._retrieve(route.get("search_terms") or [])
        plan = generate_json(
            f"{context}QUESTION: {question}\n\nVARIABLE CARDS:\n{self._cards(hits)}",
            system=PLAN_SYSTEM.format(instruments=", ".join(INSTRUMENTS)),
        )
        if plan.get("action") == "clarify":
            return AgentResult(question, plan.get("clarify")
                               or "I need more detail to answer that.",
                               plan=plan, retrieved=hits)

        try:
            table, provenance = self.execute(plan)
        except Exception as e:
            return AgentResult(question, f"The analysis failed: {e}",
                               plan=plan, retrieved=hits, error=str(e))

        shown = table.head(30).round(2)
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
            f"{truncation}"
            f"RESULT TABLE (CSV):\n{shown.to_csv(index=False)}"
        )
        answer = generate(summary_prompt, system=SUMMARY_SYSTEM)
        return AgentResult(question, answer.strip(), plan=plan, table=table,
                           provenance=provenance, retrieved=hits,
                           notes=provenance["notes"])
