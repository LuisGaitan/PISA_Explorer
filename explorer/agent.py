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

import pandas as pd

from . import catalog
from .analysis import gap, trend, weighted_mean, weighted_proportion
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
 "template": "weighted_mean"|"weighted_proportion"|"gap"|"raw_sql",
 "cycles": ["2018","2022"] or ["2022"] or ["2018"],
 "instrument": "stu_qqq" etc.,
 "measure": SQL expression; write {{pv}} for the plausible-value slot, e.g. "PV{{pv}}MATH",
            or a plain expression like "ESCS" (weighted_mean and gap only),
 "variable": str|null, "value": num|null, "valid_values": [nums]|null   (weighted_proportion:
            % of valid respondents with variable=value; valid_values = the non-missing codes),
 "group_col": str|null, "minuend": num|null, "subtrahend": num|null     (gap: mean of
            minuend group minus subtrahend group),
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

Rules: comparisons across cycles => cycles=["2018","2022"] (the system runs the
template per cycle and differences them). Achievement questions always use the
PV{{pv}} form. Filter to the countries the user names; if none named, ask
yourself whether all 80 economies is really wanted — for rankings it is.
Percentages of a category => weighted_proportion with valid_values listed."""

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
            if template == "weighted_mean":
                res = weighted_mean(self.con, tbl, plan["measure"], by=by, where=where)
            elif template == "weighted_proportion":
                res = weighted_proportion(
                    self.con, tbl, plan["variable"], plan["value"], by=by,
                    where=where, valid_values=plan.get("valid_values"))
            elif template == "gap":
                res = gap(self.con, tbl, plan["measure"], plan["group_col"],
                          plan["minuend"], plan["subtrahend"], by=by, where=where)
            else:
                raise ValueError(f"unknown template {template!r}")
            per_cycle[cycle] = res

        if len(cycles) == 2:
            table = trend(per_cycle["2018"], per_cycle["2022"],
                          by=[c for c in by] + (["contrast"] if template == "gap" else []))
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
        return table, prov

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
                        ("measure", "variable", "group_col", "where", "by"))
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
            "raw_sql": "Direct SQL — NO automatic weighting/PV/BRR treatment; "
                       "results are not population estimates unless the query weights them.",
        }
        method = methods.get(template, "")
        if "{pv}" in str(plan.get("measure") or ""):
            method += " Point estimate averaged over 10 plausible values (Rubin's rules)."
        notes = []
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
