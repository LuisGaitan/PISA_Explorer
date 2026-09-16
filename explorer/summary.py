"""App-authored statements of a result, and a check of any prose against it.

The seam where the app used to be least protected: the summary sentence was
whatever the language model wrote from a CSV. Now the app writes the facts
itself — every number, rank, change and significance verdict — and the model
is only allowed to phrase them. Its draft is then checked mechanically:

  - every number in the prose must exist in the table, the notes, the sample
    sizes or the question (at the precision written);
  - every economy named must be in the result (or the question / notes);
  - a "significant" / "not significant" claim must have a matching verdict
    in the table;
  - a short list of phrasings the app must never produce (a variable "not
    collected", an economy that "did not participate", 2025 as a projection,
    a cause attributed to a change) is rejected unless the app's own notes
    say it.

A draft that fails is regenerated once with the problems named; if it fails
again the app-authored statements are the answer. Nothing here calls a model.
"""

import re

import numpy as np
import pandas as pd

Z = 1.96
ESTIMATE_COL = re.compile(r"^estimate(_\d{4})?$")
SE_FOR = {"estimate": "se", "change": "se_change"}


# ---------- formatting ----------

def fmt(value, decimals: int | None = None) -> str:
    """Numbers as the summarizer sees them: one decimal for score-sized
    values, two for indices, shares and correlations."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "no estimate"
    v = float(value)
    if decimals is None:
        decimals = 1 if abs(v) >= 10 else 2
    return f"{v:.{decimals}f}"


def _sig(est, se) -> str | None:
    """A verdict for a difference/change/coefficient, None when unknown."""
    if est is None or se is None or pd.isna(est) or pd.isna(se) or float(se) <= 0:
        return None
    return ("statistically significant" if abs(float(est)) > Z * float(se)
            else "not statistically significant")


def _name(code, names: dict) -> str:
    code = str(code)
    if code.endswith(" avg") or code not in names:
        return code
    return f"{names[code]} ({code})"


def _measure_word(plan: dict, label_of) -> str:
    if plan.get("template") == "weighted_proportion" and plan.get("variable"):
        return "share"
    m = plan.get("measure")
    return label_of(m) if m else "value"


# ---------- app-authored statements ----------

def fact_sentences(table: pd.DataFrame, plan: dict, provenance: dict,
                   names: dict, label_of, focus_codes=(), max_rows: int = 8) -> list[str]:
    """Deterministic sentences stating what the table holds. `label_of`
    renders a measure expression ("PV{pv}MATH" -> "Mathematics score")."""
    if table is None or table.empty:
        return ["The analysis returned no rows."]
    t = table.reset_index(drop=True)
    template = str(plan.get("template") or "")
    cycles = sorted(m.group(1) for c in t.columns for m in [re.fullmatch(r"estimate_(\d{4})", c)] if m)
    multi_cycle = bool(cycles)
    single_cycle = str(t["cycle"].iloc[0]) if "cycle" in t.columns else \
        (sorted({str(c) for c in plan.get("cycles") or []}) or [""])[0]
    key_cols = [c for c in ("CNT", "measure", "contrast", "quarter", "percentile",
                            "term", "row_label", "row", "col_label", "col", "category")
                if c in t.columns]
    ranked = "rank" in t.columns
    out = []

    def row_key(r) -> str:
        parts = []
        for c in key_cols:
            v = r[c]
            if pd.isna(v):
                continue
            if c == "CNT":
                parts.append(_name(v, names))
            elif c in ("row", "col") and f"{c}_label" in key_cols:
                continue
            elif c == "quarter":
                parts.append(f"quarter {int(v)}")
            elif c == "percentile":
                parts.append(f"P{int(v)}")
            elif c == "term" and str(v) == "(intercept)":
                parts.append("intercept (predicted score when every predictor is 0 — not the mean)")
            else:
                parts.append(str(v))
        return ", ".join(parts) or "overall"

    def level(r, est_col, se_col) -> str:
        est, se = r.get(est_col), r.get(se_col)
        if pd.isna(est):
            return "no estimate"
        return f"{fmt(est)} (SE {fmt(se)})" if se_col in r and not pd.isna(se) else fmt(est)

    contrastive = template in ("gap", "quartile_gap", "percentile_spread", "correlation",
                               "regression")
    head = {
        "gap": "Difference", "quartile_gap": "Top-minus-bottom quarter difference",
        "percentile_spread": "Percentile spread", "correlation": "Weighted correlation",
        "regression": "Coefficient", "weighted_proportion": "Share",
        "percentiles": "Percentile", "quartile_means": "Mean by quarter",
        "crosstab": "Row percentage",
    }.get(template, "Mean")
    what = _measure_word(plan, label_of)
    if template == "weighted_proportion" and "category" in t.columns:
        what = f"of students with {t['category'].iloc[0]}"
    elif template in ("gap", "quartile_gap", "quartile_means", "percentiles",
                      "percentile_spread") and plan.get("measure"):
        what = f"in {label_of(plan['measure'])}"
    elif template == "weighted_mean" and "measure" not in t.columns and plan.get("measure"):
        what = f"of {label_of(plan['measure'])}"
    elif template == "correlation":
        what = f"between {label_of(plan.get('x'))} and {label_of(plan.get('y'))}"
    elif template == "regression":
        what = f"for {label_of(plan.get('measure'))}"
    else:
        what = ""

    # which rows to spell out: all when few; else top, bottom and focus rows
    rows = t
    if len(t) > max_rows:
        idx = list(t.index[:3]) + list(t.index[-3:])
        if "CNT" in t.columns and focus_codes:
            idx += list(t.index[t["CNT"].astype(str).isin(set(focus_codes))])
            idx += list(t.index[t["CNT"].astype(str).str.endswith(" avg")])
        rows = t.loc[sorted(set(idx))]
        out.append(f"The table has {len(t)} rows" +
                   (f" ({int(t['rank'].notna().sum())} ranked economies" if ranked else "") +
                   f"; the statements below cover the first three, the last three"
                   + (" and the economies named in the question" if focus_codes else "") + ".")

    for _, r in rows.iterrows():
        key = row_key(r)
        pos = ""
        if ranked and not pd.isna(r.get("rank")):
            pos = f"rank {int(r['rank'])} of {int(t['rank'].notna().sum())}, "
        if multi_cycle:
            path = " → ".join(f"{c}: {level(r, f'estimate_{c}', f'se_{c}')}" for c in cycles)
            sentence = f"{head} {what} — {key}: {pos}{path}".replace("  ", " ")
            change, se_change = r.get("change"), r.get("se_change")
            if "change" in t.columns:
                if pd.isna(change):
                    sentence += (f"; change {cycles[-1]} minus {cycles[0]}: not computed "
                                 "(see notes)")
                else:
                    verdict = _sig(change, se_change)
                    sentence += (f"; change {cycles[-1]} minus {cycles[0]} = "
                                 f"{fmt(change)} (SE {fmt(se_change)})"
                                 + (f", {verdict}" if verdict else ""))
            out.append(sentence + ".")
        else:
            val = level(r, "estimate", "se")
            sentence = f"{head} {what} — {key}"
            if single_cycle:
                sentence += f", PISA {single_cycle}"
            sentence += f": {pos}{val}"
            if contrastive and not pd.isna(r.get("estimate")):
                verdict = _sig(r.get("estimate"), r.get("se"))
                if verdict:
                    sentence += f", {verdict}"
            out.append(sentence.replace("  ", " ") + ".")

    # pairwise differences between the economies the question names, and
    # between each of them and any benchmark-average row — the only way a
    # "significantly higher than" claim can be checked
    if (not multi_cycle and not contrastive and "CNT" in t.columns
            and "estimate" in t.columns and "measure" not in t.columns):
        all_codes = [str(c) for c in t["CNT"]]
        avgs = [c for c in all_codes if c.endswith(" avg")]
        codes = [c for c in focus_codes if c in set(all_codes)]
        if len(t) - len(avgs) <= 4 and not codes:
            codes = [c for c in all_codes if not c.endswith(" avg")]
        codes = codes[:4]
        sub = t.set_index(t["CNT"].astype(str))
        members = plan.get("_benchmark_members") or {}
        cycle_key = single_cycle or (next(iter(members)) if members else "")
        member_sets = members.get(cycle_key, {}) if isinstance(members, dict) else {}

        def pair(a, b):
            ea, eb = sub.loc[a, "estimate"], sub.loc[b, "estimate"]
            sa, sb = sub.loc[a, "se"], sub.loc[b, "se"]
            if any(pd.isna(x) for x in (ea, eb, sa, sb)):
                return
            diff = float(ea) - float(eb)
            if b.endswith(" avg"):
                n = len(member_sets.get(b, []) or [])
                if n and a in set(member_sets.get(b, [])):
                    # a member's own estimate is inside the average:
                    # var(a - avg) = SE_a^2 (1 - 2/N) + SE_avg^2
                    se = float(np.sqrt(float(sa) ** 2 * (1 - 2 / n) + float(sb) ** 2))
                    how = f"the economy's share of the {n}-member average accounted for"
                else:
                    se = float(np.sqrt(float(sa) ** 2 + float(sb) ** 2))
                    how = "independent samples"
            else:
                se = float(np.sqrt(float(sa) ** 2 + float(sb) ** 2))
                how = "independent samples"
            out.append(f"{_name(a, names)} minus {_name(b, names)}: {fmt(diff)} "
                       f"(SE {fmt(se)}, {how}), {_sig(diff, se)}.")

        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                pair(a, b)
            for b in avgs:
                pair(a, b)
    return out


# ---------- checking prose against the result ----------

# Thousands-grouped numbers first ("6,622"), then plain ones; a plain number
# may be followed by a comma ("SE 2.99, independent") but not by a digit group.
NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|(?<![\w.,])-?\d+(?:\.\d+)?(?!\w|\.\d|,\d{3})")
YEAR = re.compile(r"^(19|20)\d\d$")
SIG_SENTENCE = re.compile(r"[^.!?\n]*significan[^.!?\n]*", re.I)
NEGATED_SIG = re.compile(r"\b(not|no|isn'?t|wasn'?t|aren'?t|weren'?t|non-?|never)\s*(statistically\s+)?significan", re.I)

# Phrasings the app must never produce on its own. Each is allowed only when
# the app's own notes contain the same phrase (then it is the app's claim).
FORBIDDEN = [
    (re.compile(r"\b(not|never|wasn'?t|weren'?t) (collected|administered|asked)\b", re.I),
     "claims a variable was not collected / administered"),
    (re.compile(r"\b(did not|didn'?t|does not|doesn'?t|has not|hasn'?t|never) (participate|take part|taken part)\b", re.I),
     "claims an economy did not participate"),
    (re.compile(r"\b(projected|projections?|forecasts?|forecasted|anticipated results|expected results)\b", re.I),
     "describes results as projections"),
    (re.compile(r"\b(I am|I'm) (an? )?(AI|language model|text-based|chatbot)", re.I),
     "speaks as a language model"),
    (re.compile(r"(developed|built|made|created) by the OECD|OECD'?s? (own )?(tool|app|explorer)", re.I),
     "presents the app as an OECD product"),
]
CAUSAL = re.compile(r"\b(due to|because of|caused by|driven by|attributable to|as a result of|"
                    r"thanks to|owing to|is the result of|led to|resulted in)\b", re.I)


def _numbers_in(text: str) -> list[tuple[str, float, int]]:
    out = []
    for m in NUMBER.finditer(text or ""):
        raw = m.group(0)
        clean = raw.replace(",", "")
        try:
            v = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((raw, abs(v), decimals))
    return out


def _table_numbers(table: pd.DataFrame | None) -> list[float]:
    if table is None or table.empty:
        return []
    vals = []
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            vals.extend(abs(float(v)) for v in table[col].dropna().tolist())
        else:
            for v in table[col].dropna().astype(str):
                vals.extend(n for _, n, _ in _numbers_in(v))
    return vals


def _matches(value: float, decimals: int, pool) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - p) <= tol for p in pool)


def check_prose(text: str, table: pd.DataFrame | None, provenance: dict | None,
                plan: dict | None, question: str, mentioned, allowed_codes,
                language: str = "English") -> list[str]:
    """Problems found in `text`; an empty list means the prose is grounded.
    `mentioned(text) -> [codes]` finds economies named in a text; `allowed_codes`
    = economies the prose may name (result rows, question, notes)."""
    issues = []
    prov = provenance or {}
    plan = plan or {}
    notes_text = " ".join(prov.get("notes") or []) + " " + str(prov.get("method") or "")
    sample_text = " ".join(str(s.get("students") or "") + " " + str(s.get("weighted_students") or "")
                           for s in (prov.get("sample") or []))

    # 1. numbers
    pool = _table_numbers(table)
    pool += [n for _, n, _ in _numbers_in(notes_text + " " + sample_text + " " + question)]
    pool += [n for _, n, _ in _numbers_in(" ".join(str(v) for k, v in plan.items()
                                                    if k in ("explanation", "limitation_note",
                                                             "substitution_note")))]
    if table is not None and not table.empty:
        pool.append(float(len(table)))
        if "rank" in table.columns:
            pool.append(float(table["rank"].notna().sum()))
    # the app's own statements (pairwise differences and their SEs) count
    pool += [n for line in (prov.get("facts") or []) for _, n, _ in _numbers_in(line)]
    # method constants a summary may legitimately mention
    pool += [Z, 95.0, 100.0, 0.05, 10.0, 15.0, 80.0, 90.0]
    bad = []
    for raw, value, decimals in _numbers_in(re.sub(r"\b15-year", "", text or "")):
        if YEAR.match(raw) or (decimals == 0 and value <= 12):
            continue
        if not _matches(value, decimals, pool):
            bad.append(raw)
    if bad:
        issues.append("numbers not in the result: " + ", ".join(dict.fromkeys(bad)))

    # 2. economies (English only — names in other languages are not matched)
    if language.lower() in ("english", "en"):
        extra = [c for c in mentioned(text) if c not in set(allowed_codes)]
        if extra:
            issues.append("names economies not in the result: " + ", ".join(extra))

    # 3. significance claims
    verdicts = set()
    if table is not None and not table.empty:
        for est_col, se_col in (("change", "se_change"), ("estimate", "se")):
            if est_col == "estimate" and str(plan.get("template")) not in (
                    "gap", "quartile_gap", "percentile_spread", "correlation", "regression"):
                continue
            if est_col in table.columns and se_col in table.columns:
                for est, se in zip(table[est_col], table[se_col]):
                    v = _sig(est, se)
                    if v:
                        verdicts.add(v)
        # pairwise contrasts the app stated itself count as verdicts
        for line in prov.get("facts") or []:
            if " minus " in line and "statistically significant" in line:
                verdicts.add("not statistically significant" if "not statistically" in line
                             else "statistically significant")
    for m in SIG_SENTENCE.finditer(text or ""):
        sentence = m.group(0)
        claim = ("not statistically significant" if NEGATED_SIG.search(sentence)
                 else "statistically significant")
        if claim not in verdicts:
            issues.append(f"claims '{claim}' but no such verdict is in the result: "
                          f"\"{sentence.strip()[:120]}\"")
            break

    # 4. forbidden phrasings, unless the app's notes say the same
    for rx, why in FORBIDDEN:
        if rx.search(text or "") and not rx.search(notes_text):
            issues.append(why)
    # Shanghai may only appear in a sentence that also names B-S-J-Z
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        if re.search(r"\bShanghai\b", sentence, re.I) and "B-S-J-Z" not in sentence:
            issues.append("calls B-S-J-Z (China) Shanghai")
            break
    causal_context = bool(re.search(r"\b(why|factor|cause|reason|explain|behind|driver)", question, re.I)) \
        or (table is not None and "change" in getattr(table, "columns", []))
    if causal_context and CAUSAL.search(text or "") and not CAUSAL.search(notes_text):
        issues.append("attributes a cause (PISA cannot establish causes)")
    return issues
