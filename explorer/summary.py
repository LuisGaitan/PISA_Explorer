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

KNOWN_COLS = {"estimate", "se", "n_pv", "cycle", "rank", "change", "se_change", "n",
              "n_schools", "wcov", "category", "ci95_low", "ci95_high"}
SE_COL = re.compile(r"^se(_\d{4})?$")


def _pair_se(sa, sb) -> float:
    return float(np.sqrt(float(sa) ** 2 + float(sb) ** 2))


def fact_sentences(table: pd.DataFrame, plan: dict, provenance: dict,
                   names: dict, label_of, focus_codes=(), max_rows: int = 8,
                   category_label=None, low_coverage=None) -> list[str]:
    """Deterministic sentences stating what the table holds. `label_of`
    renders a measure expression ("PV{pv}MATH" -> "Mathematics score");
    `category_label(column, value)` renders a grouping value with its
    codebook label ("ST004D01T = 1 (Female)"); `low_coverage` maps a cycle to
    [(row key, % of the weighted population without a value)]."""
    if table is None or table.empty:
        return ["The analysis returned no rows."]
    t = table.reset_index(drop=True)
    template = str(plan.get("template") or "")
    if template == "raw_sql":
        # a direct query: state its rows as they are (no estimate/SE shape)
        out = [f"Direct SQL result with {len(t)} row(s) and columns {', '.join(map(str, t.columns))} "
               "(no weighting, plausible-value or replicate treatment unless the query applied it)."]
        for _, r in t.head(max_rows).iterrows():
            out.append("Row: " + "; ".join(
                f"{c} = {fmt(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}"
                for c, v in r.items()) + ".")
        return out
    cycles = sorted(m.group(1) for c in t.columns for m in [re.fullmatch(r"estimate_(\d{4})", c)] if m)
    multi_cycle = bool(cycles)
    single_cycle = str(t["cycle"].iloc[0]) if "cycle" in t.columns else \
        (sorted({str(c) for c in plan.get("cycles") or []}) or [""])[0]
    # grouping columns beyond the standard ones (a gender or immigrant
    # background column): every statement names the group WITH its label
    extra_cols = [c for c in t.columns
                  if c not in ("CNT", "measure", "contrast", "quarter", "percentile", "term",
                               "row_label", "row", "col_label", "col", "category")
                  and c not in KNOWN_COLS and not ESTIMATE_COL.match(c) and not SE_COL.match(c)
                  and not c.startswith("ci95")]
    key_cols = [c for c in ("CNT", "measure", "contrast", "quarter", "percentile",
                            "term", "row_label", "row", "col_label", "col", "category")
                if c in t.columns] + extra_cols
    ranked = "rank" in t.columns
    ascending = ranked and plan.get("sort_desc") is False
    n_ranked = int(t["rank"].notna().sum()) if ranked else 0
    out = []

    def group_label(c, v) -> str:
        if category_label is not None:
            try:
                return str(category_label(c, v))
            except Exception:  # noqa: BLE001 — a label is a nicety
                pass
        return f"{c} = {int(v) if isinstance(v, float) and v.is_integer() else v}"

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
            elif c in extra_cols:
                parts.append(group_label(c, v))
            else:
                parts.append(str(v))
        return ", ".join(parts) or "overall"

    by_cols = [c for c in (plan.get("by") or []) if isinstance(c, str) and c in t.columns]
    for c in ("quarter", "percentile", "term"):
        if c in t.columns and c not in by_cols:
            by_cols.append(c)

    def coverage_flag(r) -> str:
        if not low_coverage or not by_cols:
            return ""
        key = ", ".join(str(r[c]) for c in by_cols)
        hits = []
        for cyc, rows in (low_coverage or {}).items():
            for k, pct in rows:
                if k == key:
                    hits.append((cyc, pct))
        if not hits:
            return ""
        if multi_cycle:
            return " [" + "; ".join(f"PISA {c}: {p}% of the weighted population has no value on the "
                                    "variable" for c, p in sorted(hits)) + "]"
        return f" [{hits[0][1]}% of the weighted population has no value on the variable]"

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
        "crosstab": "Row percentage", "weighted_sd": "Standard deviation",
        "resilient_share": "Share of academically resilient students (%)",
        "between_school_share": "Between-school share of the variance (%)",
    }.get(template, "Mean")
    what = _measure_word(plan, label_of)
    if template == "weighted_proportion" and "category" in t.columns:
        what = f"of students with {t['category'].iloc[0]}"
    elif template in ("gap", "quartile_gap", "quartile_means", "percentiles",
                      "percentile_spread", "weighted_sd", "resilient_share",
                      "between_school_share") and plan.get("measure"):
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
        if "CNT" in t.columns:
            idx += list(t.index[t["CNT"].astype(str).str.endswith(" avg")])
        if not ranked:
            # an unsorted table: the extremes are not the first and last rows,
            # so the largest and smallest value of every group are stated
            # ("which has the highest share ... in the top quarter")
            val_col = "change" if multi_cycle and "change" in t.columns else (
                f"estimate_{cycles[-1]}" if multi_cycle else "estimate")
            group_cols = [c for c in ("measure", "quarter", "percentile", "term", "contrast") + tuple(extra_cols)
                          if c in t.columns]
            if val_col in t.columns:
                groups = t.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, t)]
                for _, g in groups:
                    vals = g[val_col].dropna()
                    if not vals.empty:
                        idx += [vals.idxmax(), vals.idxmin()]
        rows = t.loc[sorted(set(idx))]
        out.append(f"The table has {len(t)} rows" +
                   (f" ({n_ranked} ranked economies" if ranked else "") +
                   f"; the statements below cover the first three, the last three"
                   + (", the largest and smallest values" if not ranked else "")
                   + (" and the economies named in the question" if focus_codes else "") + ".")
    if ascending:
        out.append(f"Ranks count from the highest value (rank 1 = largest of the {n_ranked}); "
                   "the table is sorted smallest first, so the first row is the smallest value.")
    if multi_cycle and "change" in t.columns and "CNT" in t.columns and len(t) > max_rows:
        # the shape of a large trend table in one sentence: how many rose,
        # fell or did not change significantly, and the extremes — so a
        # "which countries got worse" answer never has to be a row dump
        econ_rows = t[~t["CNT"].astype(str).str.endswith(" avg") & t["change"].notna()]
        if not econ_rows.empty and "se_change" in t.columns:
            z = econ_rows["change"] / econ_rows["se_change"].replace(0, np.nan)
            up = econ_rows[z > Z].sort_values("change", ascending=False)
            down = econ_rows[z < -Z].sort_values("change")
            flat = int((z.abs() <= Z).sum())

            def few(rows):
                return ", ".join(f"{_name(r['CNT'], names)} {fmt(r['change'])} (SE {fmt(r['se_change'])})"
                                 for _, r in rows.head(5).iterrows())
            out.append(f"Of the {len(econ_rows)} economies with a change {cycles[-1]} minus {cycles[0]}, "
                       f"{len(down)} decreased significantly, {len(up)} increased significantly and "
                       f"{flat} showed no statistically significant change"
                       + (f"; largest declines: {few(down)}" if len(down) else "")
                       + (f"; largest increases: {few(up)}" if len(up) else "") + ".")

    for _, r in rows.iterrows():
        key = row_key(r) + coverage_flag(r)
        pos = ""
        if ranked and not pd.isna(r.get("rank")):
            pos = f"rank {int(r['rank'])} of {n_ranked}, "
            if ascending:
                pos = f"rank {int(r['rank'])} of {n_ranked} from the top ({n_ranked - int(r['rank']) + 1}th smallest), "
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
            is_intercept = str(r.get("term", "")).startswith("(intercept)")
            if contrastive and not pd.isna(r.get("estimate")) and not is_intercept:
                verdict = _sig(r.get("estimate"), r.get("se"))
                if verdict:
                    sentence += f", {verdict}"
            out.append(sentence.replace("  ", " ") + ".")

    # ---- statements the app must make itself: pairwise differences ----
    # between the economies the question names (or every pair when there are
    # few), between each of them and any benchmark-average row, the change in
    # a difference across cycles (= the difference of two changes; the link
    # error cancels), and which economies are statistically tied with a
    # named one in a ranking — the only way such claims can be checked.
    if "CNT" in t.columns and (("estimate" in t.columns) or multi_cycle):
        members = plan.get("_benchmark_members") or {}
        cycle_key = single_cycle or (cycles[-1] if multi_cycle else "")
        member_sets = members.get(cycle_key, {}) if isinstance(members, dict) else {}
        pair_ok = template in ("weighted_mean", "weighted_proportion", "gap", "quartile_gap",
                               "percentile_spread", "quartile_means", "percentiles")
        group_cols = [c for c in ["measure", "quarter", "percentile", "contrast"] + extra_cols
                      if c in t.columns]
        groups = t.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, t)]
        for gkey, tm in groups:
            if not pair_ok:
                break
            tm = tm[tm["CNT"].notna()]
            if tm.empty or tm["CNT"].astype(str).duplicated().any():
                continue
            gparts = []
            if group_cols:
                keyvals = gkey if isinstance(gkey, tuple) else (gkey,)
                for c, v in zip(group_cols, keyvals):
                    if pd.isna(v):
                        continue
                    gparts.append(group_label(c, v) if c in extra_cols else
                                  (f"quarter {int(v)}" if c == "quarter" else
                                   f"P{int(v)}" if c == "percentile" else str(v)))
            tag = f" ({'; '.join(gparts)})" if gparts else ""
            all_codes = [str(c) for c in tm["CNT"]]
            avgs = [c for c in all_codes if c.endswith(" avg")]
            econ = [c for c in all_codes if not c.endswith(" avg")]
            focus = [c for c in focus_codes if c in set(econ)]
            pair_ids = [x for x in (plan.get("_pair") or []) if x in set(econ)]
            codes = list(dict.fromkeys(pair_ids + focus))
            if len(econ) <= 4:
                codes = econ
            ranked_top = econ[:3] if ranked else []
            codes = list(dict.fromkeys(codes[:4] + (ranked_top if avgs else [])))
            sub = tm.set_index(tm["CNT"].astype(str))
            contrast_word = ("difference" if template in ("gap", "quartile_gap", "percentile_spread")
                             else "estimate")

            def pair(a, b, est="estimate", se="se", cyc=""):
                ea, eb = sub.loc[a, est], sub.loc[b, est]
                sa, sb = sub.loc[a, se], sub.loc[b, se]
                if any(pd.isna(x) for x in (ea, eb, sa, sb)):
                    return None
                diff = float(ea) - float(eb)
                if b.endswith(" avg"):
                    n = len(member_sets.get(b, []) or [])
                    if n and a in set(member_sets.get(b, [])):
                        # a member's own estimate is inside the average:
                        # var(a - avg) = SE_a^2 (1 - 2/N) + SE_avg^2
                        s = float(np.sqrt(float(sa) ** 2 * (1 - 2 / n) + float(sb) ** 2))
                        how = f"the economy's share of the {n}-member average accounted for"
                    else:
                        s = _pair_se(sa, sb)
                        how = "independent samples"
                else:
                    s = _pair_se(sa, sb)
                    how = "independent samples"
                head_txt = (f"{contrast_word.capitalize()} for {_name(a, names)} minus for {_name(b, names)}"
                            if contrast_word == "difference" else
                            f"{_name(a, names)} minus {_name(b, names)}")
                out.append(f"{head_txt}{tag}{cyc}: {fmt(diff)} (SE {fmt(s)}, {how}), {_sig(diff, s)}.")
                return diff, s

            if not multi_cycle:
                for i, a in enumerate(codes):
                    for b in codes[i + 1:]:
                        if a in focus or b in focus or len(econ) <= 4 or {a, b} <= set(pair_ids):
                            pair(a, b)
                    for b in avgs:
                        pair(a, b)
                # statistically tied economies and the rank range of a named one
                if len(econ) >= 5 and "se" in tm.columns:
                    for a in (pair_ids or focus)[:2]:
                        ea, sa = sub.loc[a, "estimate"], sub.loc[a, "se"]
                        if pd.isna(ea) or pd.isna(sa):
                            continue
                        tied, above, below = [], 0, 0
                        for b in econ:
                            if b == a:
                                continue
                            eb, sb = sub.loc[b, "estimate"], sub.loc[b, "se"]
                            if pd.isna(eb) or pd.isna(sb):
                                continue
                            d = float(eb) - float(ea)
                            if abs(d) <= Z * _pair_se(sa, sb):
                                tied.append(b)
                            elif d > 0:
                                above += 1
                            else:
                                below += 1
                        tied_txt = (", ".join(_name(c, names) for c in tied[:25])
                                    + (f" and {len(tied) - 25} more" if len(tied) > 25 else "")) if tied else "none"
                        sentence = (f"Economies whose {contrast_word} is not statistically different from "
                                    f"{_name(a, names)}{tag} (independent samples, 1.96 SE): {tied_txt}")
                        if ranked and not pd.isna(sub.loc[a].get("rank")):
                            lo, hi = above + 1, above + 1 + len(tied)
                            sentence += (f"; {above} economies are significantly higher and {below} "
                                         f"significantly lower, so its position could be anywhere from "
                                         f"rank {lo} to rank {hi}")
                        out.append(sentence + ".")
            else:
                first, last = cycles[0], cycles[-1]
                for i, a in enumerate(codes):
                    for b in codes[i + 1:]:
                        if not (a in focus or b in focus or len(econ) <= 4 or {a, b} <= set(pair_ids)):
                            continue
                        d_last = pair(a, b, f"estimate_{last}", f"se_{last}", f", PISA {last}")
                        d_first = pair(a, b, f"estimate_{first}", f"se_{first}", f", PISA {first}")
                        if d_last and d_first:
                            change = d_last[0] - d_first[0]
                            s = float(np.sqrt(d_last[1] ** 2 + d_first[1] ** 2))
                            out.append(
                                f"Change {last} minus {first} in ({_name(a, names)} minus {_name(b, names)}){tag} "
                                f"— equal to the change for {_name(a, names)} minus the change for {_name(b, names)}: "
                                f"{fmt(change)} (SE {fmt(s)}, sampling error of the four estimates; the "
                                f"link error cancels), {_sig(change, s)}.")
                    for b in avgs:
                        pair(a, b, f"estimate_{last}", f"se_{last}", f", PISA {last}")
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
    (re.compile(r"\b(partial (data|table|view|results?)|table is partial|(provided|available|shown) "
                r"(data|table|rows) (is|are|only|does)|does not include (data for )?all|"
                r"only (includes|contains|shows|covers) (data for )?(some|a subset|part)|"
                r"a complete list would require|based on the (available|provided|partial) (data|table)|"
                r"(provided |available )?(results|data|table|output) (do|does) not (contain|include|show)"
                r"( any)? (data|rows|results|values))\b", re.I),
     "describes the result as partial or incomplete (the full table is in the result; use the verified statements)"),
    (re.compile(r"\b(cannot|can'?t|unable to|not able to) (directly )?(compute|calculate|perform|test|determine) "
                r"(the |a |statistical )?(difference|significance|standard error|comparison)\b", re.I),
     "claims the app cannot compute a difference or its significance (the verified statements contain them)"),
]
# Word pairs that name opposite groups: a number the verified statements
# attach to one side must not be attributed to the other in the prose
# ("the female mean" reported as boys'; a share who DISAGREE reported as
# agreeing).
OPPOSITES = [
    (re.compile(r"\b(male|males|boys?|men|masculin\w*|niños|chicos|garçons)\b", re.I),
     re.compile(r"\b(female|females|girls?|women|femenin\w*|niñas|chicas|filles)\b", re.I), "male", "female"),
    (re.compile(r"\bdisagree\w*|\bdesacuerdo\b", re.I), re.compile(r"\bagree\w*|\bacuerdo\b", re.I),
     "disagree", "agree"),
    (re.compile(r"\bprivate\b|\bprivad[ao]s?\b", re.I), re.compile(r"\bpublic\b|\bpúblic[ao]s?\b", re.I),
     "private", "public"),
    (re.compile(r"\brural(es)?\b", re.I), re.compile(r"\burban[ao]?s?\b", re.I), "rural", "urban"),
    (re.compile(r"\b(immigrants?|foreign-born|first-generation|second-generation|inmigrantes?)\b", re.I),
     re.compile(r"\b(native|natives|non-immigrants?|native-born|nativos?)\b", re.I), "immigrant", "native"),
]
CAUSAL = re.compile(r"\b(due to|because of|caused by|driven by|attributable to|as a result of|"
                    r"thanks to|owing to|is the result of|led to|resulted in)\b", re.I)
# "-15.27 points lower": a signed number and a direction word say the same
# thing twice and read as a double negative
SIGNED_DIRECTION = re.compile(
    r"(?<![\w.])[-−–]\s?\d+(?:[.,]\d+)?\s*(?:points?|pp|percentage points?|puntos?|points de|Punkte|"
    r"pontos?|ポイント|%)?\s*(?:\([^)]*\)\s*)?(lower|higher|less|more|fewer|greater|below|above|"
    r"menos|más|menor|mayor|inférieur|supérieur|niedriger|höher|abaixo|acima)\b", re.I)


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
                           + " " + str(s.get("schools") or "")
                           for s in (prov.get("sample") or []))

    # 1. numbers — non-English drafts write thousands as "6 770" or "6.770"
    if language.lower() not in ("english", "en"):
        text = re.sub(r"(?<!\d)(\d{1,3})[   ](\d{3})(?!\d)", r"\1\2", text or "")
        text = re.sub(r"(?<![\d.,])(\d{1,3})\.(\d{3})(?![\d.])", r"\1\2", text)
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
        contrastive = str(plan.get("template")) in ("gap", "quartile_gap", "percentile_spread",
                                                    "correlation", "regression")
        pairs = [("change", "se_change")]
        if contrastive:
            # a contrast's level in each cycle carries its own verdict too
            pairs += [("estimate", "se")] + [(c, "se" + c[len("estimate"):]) for c in table.columns
                                            if re.fullmatch(r"estimate_\d{4}", str(c))]
        for est_col, se_col in pairs:
            if est_col in table.columns and se_col in table.columns:
                for est, se in zip(table[est_col], table[se_col]):
                    v = _sig(est, se)
                    if v:
                        verdicts.add(v)
        # pairwise contrasts and tie statements the app stated itself count as verdicts
        for line in prov.get("facts") or []:
            if " minus " in line and "statistically significant" in line:
                verdicts.add("not statistically significant" if "not statistically" in line
                             else "statistically significant")
            if "not statistically different from" in line:
                verdicts.add("not statistically significant")
                if re.search(r"\b[1-9]\d* economies are significantly higher|\b[1-9]\d* significantly lower", line):
                    verdicts.add("statistically significant")
    for m in SIG_SENTENCE.finditer(text or ""):
        sentence = m.group(0)
        claim = ("not statistically significant" if NEGATED_SIG.search(sentence)
                 else "statistically significant")
        if claim not in verdicts:
            issues.append(f"claims '{claim}' but no such verdict is in the result: "
                          f"\"{sentence.strip()[:120]}\"")
            break

    # 3b. a number attributed to the opposite group of the one the app stated
    facts = prov.get("facts") or []
    for a_rx, b_rx, a_word, b_word in OPPOSITES:
        sides: list[tuple[float, int, str]] = []
        for line in facts:
            if a_word == "disagree":
                # an item stem ("Agree:", "Agree/disagree: Your intelligence…")
                # names no side; only the decoded codes do
                line = re.sub(r"\bagree\s*/\s*disagree\s*:|\bagree\s*:|to what extent do you agree or disagree",
                              " ", line, flags=re.I)
            sa, sb = bool(a_rx.search(line)), bool(b_rx.search(line))
            if sa == sb:
                continue
            for _, n, d in _numbers_in(line):
                if d == 0 and n <= 12:
                    continue
                sides.append((n, d, a_word if sa else b_word))
        if not sides:
            continue
        flagged = False
        for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
            sa, sb = bool(a_rx.search(sentence)), bool(b_rx.search(sentence))
            if sa == sb:
                continue
            side = a_word if sa else b_word
            for raw, n, d in _numbers_in(sentence):
                if YEAR.match(raw) or (d == 0 and n <= 12):
                    continue
                stated = {s for (fn, fd, s) in sides if abs(fn - n) <= 0.5 * 10 ** (-min(d, fd)) + 1e-9}
                if stated and side not in stated:
                    issues.append(f"attributes {raw} to the {side} group but the verified statement "
                                  f"gives it for the {', '.join(sorted(stated))} group: "
                                  f"\"{sentence.strip()[:120]}\"")
                    flagged = True
                    break
            if flagged:
                break

    # 3c. a standard error rounded away to 0.0 (the facts keep two decimals)
    if re.search(r"\(\s*(SE|EE|ET|SE\s*=)\s*0(?:[.,]0+)?\s*\)", text or "", re.I) \
            and not any(re.search(r"\(SE 0\.00?\)", line) for line in facts):
        issues.append("rounds a standard error to 0.0 (copy the SE as the verified statement gives it)")
    # 3d. a negative number paired with a direction word ("-15.3 points lower")
    if SIGNED_DIRECTION.search(text or ""):
        issues.append("pairs a negative number with 'lower/higher' (write the absolute value with the "
                      "direction word: '15.3 points lower')")

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
