"""The closed plan grammar: what the planner may say, and how the app turns
it into an executable plan.

Until method v6 the planner wrote SQL fragments straight into the plan
("where": "CNT IN ('KAZ','TJK') AND ESCS_Q = 4", "measure": "CASE WHEN …",
"by": ["CNT", "group_col"]). Every class of wrong number found by the red
teams that was not a missing OECD convention came from that freedom:
invented economy codes, invented columns, a CASE in `by`, a pooled mean
across economies, a dummy that coded non-respondents as the reference
group. Here the planner fills a FORM instead — a statistic from a fixed
list, measures as small objects (a domain, a variable, a level share, a
response-code share), economies as codes, filters as (variable, op,
values), contrasts as code sets, predictors as variables with optional
dummy codes — and the app compiles the SQL. Anything the form cannot say
cannot be planned, so it cannot be computed wrongly; it is stated instead.

compile_plan() returns the legacy plan dict that Agent.execute() runs, so
every repair, guard, note and golden test downstream is unchanged. The
grammar plan itself is kept under plan["_grammar"] for the provenance.
"""

import json
import re

from . import catalog, link_errors, regions

STATISTICS = {
    "mean": "weighted_mean",
    "share": "weighted_mean",              # a share is the mean of a 100/0 measure
    "gap": "gap",
    "quartile_means": "quartile_means",
    "quartile_gap": "quartile_gap",
    "correlation": "correlation",
    "percentiles": "percentiles",
    "percentile_spread": "percentile_spread",
    "crosstab": "crosstab",
    "regression": "regression",
    "sd": "weighted_sd",
    "resilient_share": "resilient_share",
    "between_school_share": "between_school_share",
    "raw_sql": "raw_sql",
}

# plausible-value families the data hold, by suffix
PV_DOMAINS = {"MATH", "READ", "SCIE", "CMPS", "CPPK", "CMOD", "CPRO", "SEPS", "SEDE",
              "SEID", "SENV", "GLCM", "FLIT", "CRTH_NC"}
DOMAIN_WORDS = {"mathematics": "MATH", "math": "MATH", "maths": "MATH", "reading": "READ",
                "science": "SCIE", "creative thinking": "CRTH_NC", "global competence": "GLCM",
                "financial literacy": "FLIT"}
FILTER_OPS = {"=", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null", "not_null"}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CODE = re.compile(r"^[A-Za-z]{3}$")


class GrammarError(ValueError):
    """The planner's form cannot be compiled; the message names the field."""


def _is_variable(name: str) -> bool:
    return bool(IDENT.match(name)) and (name == "CNT" or not catalog.describe(name).empty)


def _num(v) -> str:
    """A numeric literal as SQL text (a code list value)."""
    if isinstance(v, bool):
        raise GrammarError("a value must be a number, not true/false")
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) and not float(v).is_integer() else str(int(v))
    s = str(v).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return s
    raise GrammarError(f"a value must be a number, not {v!r}")


def _domain(spec) -> str:
    d = str(spec or "").strip()
    d = DOMAIN_WORDS.get(d.lower(), d.upper())
    d = re.sub(r"^PV(\{pv\}|\d{1,2})", "", d)
    if d not in PV_DOMAINS:
        raise GrammarError(f"unknown score domain {spec!r}; use MATH, READ, SCIE, CMPS, CPPK, CMOD, "
                           "CPRO, SEPS, SEDE, SEID, SENV, GLCM, FLIT or CRTH_NC")
    return d


def _level_bound(domain: str, level) -> float:
    lv = str(level).strip().lower().replace("level", "").strip()
    for name, bound in link_errors.LEVELS.get(domain, []):
        if name == lv:
            return bound
    try:
        value = float(lv)                                   # an explicit cut point
    except ValueError:
        raise GrammarError(f"unknown proficiency level {level!r} for {domain}") from None
    if value < 100:
        raise GrammarError(f"unknown proficiency level {level!r} for {domain} (levels are 1c, 1b, "
                           "1a, 2, 3, 4, 5, 6, or a score point cut such as 500)")
    return value


def compile_measure(m) -> str:
    """One measure object (or shortcut string) -> a SQL expression."""
    if m is None:
        raise GrammarError("a measure is missing")
    if isinstance(m, str):
        s = m.strip()
        if not s:
            raise GrammarError("a measure is missing")
        if s.lower() in DOMAIN_WORDS or s.upper() in PV_DOMAINS or re.match(r"^PV(\{pv\}|\d{1,2})", s):
            return "PV{pv}" + _domain(s)
        if _is_variable(s):
            return s
        raise GrammarError(f"{s!r} is not a variable in the catalog and not a score domain")
    if not isinstance(m, dict):
        raise GrammarError(f"a measure must be an object, not {type(m).__name__}")
    keys = set(m)
    if "score" in keys:
        expr = "PV{pv}" + _domain(m["score"])
        if m.get("divide_by"):
            expr = f"{expr} / {_num(m['divide_by'])}"
        return expr
    if "variable" in keys and not (keys & {"codes", "op", "value"}):
        v = str(m["variable"]).strip()
        if not _is_variable(v):
            raise GrammarError(f"{v!r} is not a variable in the catalog")
        return v
    if "level_share" in keys or ("level" in keys and ("domain" in keys or "domains" in keys)):
        spec = m.get("level_share") if "level_share" in keys else m
        side = str(spec.get("side") or spec.get("op") or "below").strip().lower().replace(" ", "_")
        if side in ("at_or_above", ">=", "at least", "at_least", "above_or_at", "reaching"):
            op = ">="
        elif side in ("below", "<", "under"):
            op = "<"
        elif side in ("above", ">"):
            op = ">"
        elif side in ("at_or_below", "<=", "at most", "at_most"):
            op = "<="
        else:
            raise GrammarError(f"unknown side {side!r} in a level share (use below or at_or_above)")
        domains = spec.get("domains") or ([spec["domain"]] if spec.get("domain") else [])
        if not domains:
            raise GrammarError("a level share needs a domain")
        conds = []
        for d in domains:
            dom = _domain(d)
            conds.append(f"PV{{pv}}{dom} {op} {_level_bound(dom, spec.get('level'))}")
        joiner = " OR " if str(spec.get("combine") or "all").lower() in ("any", "or") else " AND "
        cond = joiner.join(conds)
        return f"CASE WHEN {cond} THEN 100.0 ELSE 0.0 END"
    if "code_share" in keys or ("variable" in keys and "codes" in keys):
        spec = m.get("code_share") if "code_share" in keys else m
        v = str(spec.get("variable") or "").strip()
        if not _is_variable(v):
            raise GrammarError(f"{v!r} is not a variable in the catalog (code share)")
        codes = spec.get("codes")
        if not isinstance(codes, list) or not codes:
            raise GrammarError(f"a code share of {v} needs a non-empty list of codes")
        lst = ", ".join(_num(c) for c in codes)
        valid = spec.get("valid")
        if isinstance(valid, list) and valid:
            not_in = ", ".join(_num(c) for c in valid if c not in codes)
            tail = f"WHEN {v} IN ({not_in}) THEN 0.0" if not_in else f"WHEN {v} IS NOT NULL THEN 0.0"
        else:
            tail = f"WHEN {v} IS NOT NULL THEN 0.0"
        return f"CASE WHEN {v} IN ({lst}) THEN 100.0 {tail} END"
    if "threshold_share" in keys or ("variable" in keys and "op" in keys):
        spec = m.get("threshold_share") if "threshold_share" in keys else m
        v = str(spec.get("variable") or "").strip()
        if not _is_variable(v):
            raise GrammarError(f"{v!r} is not a variable in the catalog (threshold share)")
        op = str(spec.get("op") or "").strip()
        if op not in ("<", "<=", ">", ">=", "=", "!="):
            raise GrammarError(f"unknown comparison {op!r} in a threshold share")
        return f"CASE WHEN {v} {op} {_num(spec.get('value'))} THEN 100.0 WHEN {v} IS NOT NULL THEN 0.0 END"
    if "count_share" in keys:
        # % of students for whom exactly / at least N of several items take
        # one of the codes ("repeated a grade only once" = exactly one of the
        # three ST127 items answered 'once')
        spec = m["count_share"]
        vars_ = [str(v).strip() for v in (spec.get("variables") or [])]
        if len(vars_) < 2 or not all(_is_variable(v) for v in vars_):
            raise GrammarError("a count share needs two or more catalog variables")
        codes = spec.get("codes")
        if not isinstance(codes, list) or not codes:
            raise GrammarError("a count share needs the codes that count")
        op = str(spec.get("op") or "=").strip()
        if op not in ("=", ">=", "<=", ">", "<"):
            raise GrammarError(f"unknown comparison {op!r} in a count share")
        lst = ", ".join(_num(c) for c in codes)
        total = " + ".join(f"CASE WHEN {v} IN ({lst}) THEN 1 ELSE 0 END" for v in vars_)
        return (f"CASE WHEN ({total}) {op} {_num(spec.get('count', 1))} THEN 100.0 "
                f"WHEN COALESCE({', '.join(vars_)}) IS NOT NULL THEN 0.0 END")
    if "school_mean_of" in keys:
        v = str(m["school_mean_of"]).strip()
        if not _is_variable(v):
            raise GrammarError(f"{v!r} is not a variable in the catalog (school mean)")
        return f"AVG({v}) OVER (PARTITION BY CNTSCHID)"
    raise GrammarError(f"unknown measure form {json.dumps(m)[:80]}; use {{\"score\": …}}, "
                       "{\"variable\": …}, {\"level_share\": …}, {\"code_share\": …} or "
                       "{\"threshold_share\": …}")


def compile_filters(filters, invented: dict, present: set) -> tuple[str | None, list[str], list[str]]:
    """Filters -> (SQL where, economy codes named, notes). CNT filters become
    economies; STRATUM filters are dropped (the app places a named stratum
    beside the economy itself)."""
    if filters is None:
        return None, [], []
    if isinstance(filters, dict):
        filters = [filters]
    if not isinstance(filters, list):
        raise GrammarError("filters must be a list of {variable, op, values}")
    clauses, economies, notes = [], [], []
    for f in filters:
        if not isinstance(f, dict):
            raise GrammarError(f"a filter must be an object, not {f!r}")
        v = str(f.get("variable") or "").strip()
        op = str(f.get("op") or "=").strip().lower()
        if op in ("==", "eq"):
            op = "="
        if op in ("<>", "ne"):
            op = "!="
        if op in ("not in", "notin"):
            op = "not_in"
        values = f.get("values")
        if values is None and "value" in f:
            values = [f["value"]]
        if op not in FILTER_OPS:
            raise GrammarError(f"unknown filter operator {op!r} (use =, !=, <, <=, >, >=, in, not_in)")
        if v.upper() == "CNT":
            codes = [str(c).upper() for c in (values or []) if CODE.match(str(c))]
            codes = [invented.get(c, c) if c not in present else c for c in codes]
            if op in ("=", "in"):
                economies.extend(codes)
            elif codes:
                clauses.append("CNT NOT IN (" + ", ".join(f"'{c}'" for c in codes) + ")")
            continue
        if v.upper() == "STRATUM":
            notes.append("stratum filter left to the app")
            continue
        if not _is_variable(v):
            raise GrammarError(f"filter variable {v!r} is not in the catalog")
        if op == "is_null":
            clauses.append(f"{v} IS NULL")
        elif op == "not_null":
            clauses.append(f"{v} IS NOT NULL")
        elif op in ("in", "not_in"):
            if not isinstance(values, list) or not values:
                raise GrammarError(f"filter on {v} needs a list of values")
            lst = ", ".join(_num(x) for x in values)
            clauses.append(f"{v} {'NOT IN' if op == 'not_in' else 'IN'} ({lst})")
        else:
            if not isinstance(values, list) or len(values) != 1:
                raise GrammarError(f"filter {v} {op} needs exactly one value")
            clauses.append(f"{v} {op} {_num(values[0])}")
    return (" AND ".join(clauses) or None), list(dict.fromkeys(economies)), notes


def compile_contrast(c) -> dict:
    """{"variable", "minuend": [codes], "subtrahend": [codes], "label"} ->
    gap fields. Single codes on both sides use the column directly; sets
    become a NULL-safe CASE (students in neither set stay out)."""
    if not isinstance(c, dict):
        raise GrammarError("a gap needs a contrast object {variable, minuend, subtrahend, label}")
    v = str(c.get("variable") or "").strip()
    a, b = c.get("minuend"), c.get("subtrahend")
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        raise GrammarError("a contrast needs a minuend and a subtrahend (lists of codes)")
    label = str(c.get("label") or "").strip() or None
    if v.upper() == "CNT":
        codes = [str(x).upper() for x in a + b]
        if len(a) != 1 or len(b) != 1 or not all(CODE.match(x) for x in codes):
            raise GrammarError("an economy-vs-economy contrast takes one economy code on each side")
        return {"group_col": "CNT", "minuend": codes[0], "subtrahend": codes[1], "group_label": label}
    if not _is_variable(v):
        raise GrammarError(f"contrast variable {v!r} is not in the catalog")
    if len(a) == 1 and len(b) == 1:
        return {"group_col": v, "minuend": float(_num(a[0])), "subtrahend": float(_num(b[0])),
                "group_label": label}
    la, lb = ", ".join(_num(x) for x in a), ", ".join(_num(x) for x in b)
    return {"group_col": f"CASE WHEN {v} IN ({la}) THEN 1 WHEN {v} IN ({lb}) THEN 0 END",
            "minuend": 1, "subtrahend": 0, "group_label": label or f"{v} in ({la}) minus {v} in ({lb})"}


def compile_predictors(preds) -> tuple[list[str], list[str]]:
    if not isinstance(preds, list) or not preds:
        raise GrammarError("a regression needs a list of predictors")
    exprs, names = [], []
    preds = [{"variable": p} if isinstance(p, str) else p for p in preds]
    # every code a variable's dummies name: a sibling category is 0 on this
    # dummy, not missing (two dummies of IMMIG with reference [1] each would
    # otherwise leave only natives with complete rows)
    codes_by_var: dict[str, set] = {}
    for p in preds:
        if isinstance(p, dict) and (p.get("codes") or p.get("dummy_codes")):
            c = p.get("codes") or p.get("dummy_codes")
            codes_by_var.setdefault(str(p.get("variable")), set()).update(
                _num(x) for x in (c if isinstance(c, list) else [c]))
    for p in preds:
        if not isinstance(p, dict):
            raise GrammarError(f"a predictor must be an object, not {p!r}")
        if "school_mean_of" in p:
            v = str(p["school_mean_of"]).strip()
            if not _is_variable(v):
                raise GrammarError(f"{v!r} is not a variable in the catalog")
            exprs.append(f"AVG({v}) OVER (PARTITION BY CNTSCHID)")
            names.append(str(p.get("label") or f"school average of {v}"))
            continue
        v = str(p.get("variable") or "").strip()
        if not _is_variable(v):
            raise GrammarError(f"predictor {v!r} is not in the catalog")
        codes = p.get("codes") or p.get("dummy_codes")
        if codes:
            codes = codes if isinstance(codes, list) else [codes]
            ref = p.get("reference") or p.get("reference_codes") or []
            ref = ref if isinstance(ref, list) else [ref]
            lst = ", ".join(_num(x) for x in codes)
            zeros = sorted({_num(x) for x in ref} | (codes_by_var.get(v, set()) - {_num(x) for x in codes}),
                           key=lambda s: float(s))
            tail = (f"WHEN {v} IN ({', '.join(zeros)}) THEN 0" if zeros
                    else f"WHEN {v} IS NOT NULL THEN 0")
            exprs.append(f"CASE WHEN {v} IN ({lst}) THEN 1 {tail} END")
            names.append(str(p.get("label") or f"{v} in ({lst}) (1) vs other (0)"))
        else:
            exprs.append(v)
            names.append(str(p.get("label") or v))
    return exprs, names


def compile_benchmarks(bench) -> tuple[bool, list]:
    if bench is None:
        return False, []
    if isinstance(bench, (str, dict)):
        bench = [bench]
    if not isinstance(bench, list):
        raise GrammarError("benchmarks must be a list")
    oecd, groups = False, []
    for b in bench:
        if isinstance(b, dict):
            members = [str(c).upper() for c in (b.get("members") or []) if CODE.match(str(c))]
            if len(members) < 2:
                raise GrammarError("a named group needs at least two member codes")
            groups.append({"label": str(b.get("label") or "Group")[:28], "members": members})
        elif isinstance(b, str):
            s = b.strip()
            if s.lower() == "oecd":
                oecd = True
            elif s.lower() in ("all participants", "world", "all economies", "all countries"):
                groups.append("All participants")
            elif regions.canonical(s):
                groups.append(regions.canonical(s))
            else:
                raise GrammarError(f"{s!r} is not a benchmark group this app knows (OECD, "
                                   "All participants, or a region from the list)")
    return oecd, groups


def _by(cols) -> list[str]:
    if cols is None:
        return ["CNT"]
    if isinstance(cols, str):
        cols = [cols]
    out = []
    for c in cols:
        c = str(c).strip()
        if c.upper() == "CNT":
            out.append("CNT")
        elif c.lower() in ("cycle", "year"):
            continue
        elif _is_variable(c):
            out.append(c)
        else:
            raise GrammarError(f"grouping variable {c!r} is not in the catalog")
    return list(dict.fromkeys(out))


def compile_plan(g: dict, invented: dict | None = None, present: set | None = None) -> dict:
    """A grammar plan -> the legacy plan Agent.execute() runs."""
    if not isinstance(g, dict):
        raise GrammarError("the plan must be a JSON object")
    invented = invented or {}
    present = present or set()
    plan: dict = {"_grammar": json.loads(json.dumps(g, default=str))}
    for k in ("action", "clarify", "explanation", "substitution_note", "limitation_note",
              "instrument", "regions", "sql"):
        if g.get(k) is not None:
            plan[k] = g[k]
    if g.get("action") == "clarify":
        plan["action"] = "clarify"
        return plan
    stat = str(g.get("statistic") or g.get("template") or "").strip().lower()
    if stat not in STATISTICS:
        raise GrammarError(f"unknown statistic {stat!r}; choose one of {', '.join(STATISTICS)}")
    template = STATISTICS[stat]
    plan["template"] = template
    plan["action"] = "analyze"
    cycles = g.get("cycles") or ["2025"]
    cycles = [str(c) for c in (cycles if isinstance(cycles, list) else [cycles])]
    bad = [c for c in cycles if c not in ("2018", "2022", "2025")]
    if bad:
        raise GrammarError(f"unknown cycle(s) {bad}; the data hold 2018, 2022 and 2025")
    plan["cycles"] = sorted(set(cycles))
    if template == "raw_sql":
        if not g.get("sql"):
            raise GrammarError("raw_sql needs sql")
        plan["by"] = []
        return plan

    # measures
    ms = g.get("measures")
    if ms is None and g.get("measure") is not None:
        ms = [g["measure"]]
    if isinstance(ms, (str, dict)):
        ms = [ms]
    exprs = [compile_measure(m) for m in (ms or [])] if template not in ("correlation", "crosstab") else []
    if template in ("weighted_mean",) and len(exprs) > 1:
        plan["measures"] = exprs
        plan["measure"] = None
    elif exprs:
        plan["measure"] = exprs[0]
        if len(exprs) > 1:
            plan["limitation_note"] = ((plan.get("limitation_note") or "") +
                                       f" Only the first measure was computed for this statistic; also asked: "
                                       f"{len(exprs) - 1} more.").strip()
    elif template not in ("correlation", "crosstab"):
        raise GrammarError(f"{stat} needs a measure")
    if stat == "share" and exprs and not any(re.match(r"^\s*CASE\b", e, re.I) for e in exprs):
        raise GrammarError("a share needs a level_share, code_share or threshold_share measure")

    # filters and economies
    where, econ_from_filters, notes = compile_filters(g.get("filters"), invented, present)
    econ = g.get("economies") or []
    econ = econ if isinstance(econ, list) else [econ]
    codes = []
    for c in econ:
        c = str(c).strip().upper()
        if not CODE.match(c):
            raise GrammarError(f"{c!r} is not a three-letter economy code")
        codes.append(invented.get(c, c) if c not in present else c)
    codes = list(dict.fromkeys(codes + econ_from_filters))
    if codes:
        clause = f"CNT = '{codes[0]}'" if len(codes) == 1 else \
            "CNT IN (" + ", ".join(f"'{c}'" for c in codes) + ")"
        where = f"{clause} AND {where}" if where else clause
    plan["where"] = where
    if notes:
        plan["_grammar_notes"] = notes

    # grouping, contrasts and the rest
    plan["by"] = _by(g.get("by"))
    if template in ("gap",):
        plan.update(compile_contrast(g.get("contrast")))
    if template in ("quartile_means", "quartile_gap", "resilient_share"):
        qv = str(g.get("quart_variable") or "ESCS").strip()
        if not _is_variable(qv):
            raise GrammarError(f"quart_variable {qv!r} is not in the catalog")
        plan["quart_variable"] = qv
    if template in ("correlation", "regression", "weighted_mean", "gap") and g.get("quart_variable") \
            and template not in ("quartile_means",):
        qv = str(g.get("quart_variable")).strip()
        if not _is_variable(qv):
            raise GrammarError(f"quart_variable {qv!r} is not in the catalog")
        plan["quart_variable"] = qv
    if template == "correlation":
        plan["x"] = compile_measure(g.get("x"))
        plan["y"] = compile_measure(g.get("y"))
    if template == "regression":
        plan["predictors"], plan["predictor_names"] = compile_predictors(g.get("predictors"))
    if template == "percentiles" and g.get("ps"):
        plan["ps"] = [int(p) for p in g["ps"]]
    if template == "percentile_spread":
        plan["upper"] = int(g.get("upper") or 90)
        plan["lower"] = int(g.get("lower") or 10)
    if template == "resilient_share" and g.get("level") is not None:
        dom = link_errors.domain_of(plan["measure"]) or "MATH"
        plan["level"] = _level_bound(dom, g["level"])
    if template == "crosstab":
        for k in ("row_var", "col_var"):
            v = str(g.get(k) or "").strip()
            if not _is_variable(v):
                raise GrammarError(f"{k} {v!r} is not in the catalog")
            plan[k] = v
        for k in ("valid_rows", "valid_cols"):
            if isinstance(g.get(k), list):
                plan[k] = [float(_num(x)) for x in g[k]]
    oecd, groups = compile_benchmarks(g.get("benchmarks"))
    if oecd:
        plan["include_oecd_average"] = True
    if groups:
        plan["include_average_of"] = groups
    sort = g.get("sort")
    if isinstance(sort, dict) and sort.get("by"):
        plan["sort_by"] = "change" if str(sort["by"]).lower() == "change" else "estimate"
        plan["sort_desc"] = bool(sort.get("desc", True))
        if sort.get("top_n"):
            plan["top_n"] = int(sort["top_n"])
    elif isinstance(sort, str) and sort:
        plan["sort_by"] = "change" if sort.lower() == "change" else "estimate"
        plan["sort_desc"] = True
    if g.get("top_n") and not plan.get("top_n"):
        plan["top_n"] = int(g["top_n"])

    # per-cycle overrides: the same form, compiled field by field
    ov_in = g.get("cycle_overrides")
    if isinstance(ov_in, dict) and ov_in:
        out = {}
        for cyc, spec in ov_in.items():
            if not isinstance(spec, dict) or not spec:
                continue
            o: dict = {}
            if spec.get("measure") is not None or spec.get("measures") is not None:
                mm = spec.get("measures") if spec.get("measures") is not None else [spec["measure"]]
                mm = mm if isinstance(mm, list) else [mm]
                ex = [compile_measure(m) for m in mm]
                if len(ex) > 1 and template == "weighted_mean":
                    o["measures"] = ex
                else:
                    o["measure"] = ex[0]
            if spec.get("contrast") is not None:
                o.update({k: v for k, v in compile_contrast(spec["contrast"]).items() if v is not None})
            if spec.get("by") is not None:
                o["by"] = _by(spec["by"])
            if spec.get("filters") is not None or spec.get("economies") is not None:
                w, ec, _ = compile_filters(spec.get("filters"), invented, present)
                ec = list(dict.fromkeys([str(c).upper() for c in (spec.get("economies") or [])] + ec))
                if ec:
                    cl = "CNT IN (" + ", ".join(f"'{c}'" for c in ec) + ")"
                    w = f"{cl} AND {w}" if w else cl
                if w:
                    o["where"] = w
            if spec.get("predictors") is not None:
                o["predictors"], o["predictor_names"] = compile_predictors(spec["predictors"])
            for k in ("x", "y"):
                if spec.get(k) is not None:
                    o[k] = compile_measure(spec[k])
            if spec.get("quart_variable"):
                o["quart_variable"] = str(spec["quart_variable"])
            if spec.get("instrument"):
                o["instrument"] = str(spec["instrument"])
            if o:
                out[str(cyc)] = o
        if out:
            plan["cycle_overrides"] = out
    return plan


# ---------- the planner prompt ----------

SCHEMA = """
Reply with ONLY this JSON object (a FORM: every field is a choice from a
menu or a code from the cards — never SQL, never a CASE expression, never a
column you invent):
{
 "action": "analyze" | "clarify",
 "clarify": str|null   (plain language for a non-technical reader; variables
            by label with the code in parentheses),
 "statistic": one of
     "mean"                 weighted mean of a score or a variable,
     "share"                % of students meeting a condition (a
                            proficiency level, a response code, a threshold),
     "gap"                  mean difference between two groups (contrast),
     "quartile_means"       mean per weighted quarter of quart_variable,
     "quartile_gap"         top minus bottom quarter of quart_variable,
     "correlation"          weighted correlation of x and y,
     "percentiles"          weighted percentiles (ps),
     "percentile_spread"    P<upper> minus P<lower> (dispersion, P90-P10),
     "crosstab"             row percentages of col_var within row_var,
     "regression"           weighted least squares of the measure on predictors,
     "sd"                   weighted standard deviation,
     "resilient_share"      % of academically resilient students (bottom
                            quarter of quart_variable, top quarter of the score),
     "between_school_share" between-school share of the variance (ICC, %),
     "raw_sql"              one read-only aggregate SELECT (last resort),
 "cycles": chronological list from ["2018","2022","2025"] (["2025"] when no
            cycle is named; all three for "over time" / "trend" / "since 2018"),
 "instrument": "stu_qqq" (default) | "stu_sch" (a school variable with student
            outcomes) | "stu_crt" (creative thinking) | "flt_qqq" (financial
            literacy) | "sch_qqq" | "tch_qqq",
 "measures": [measure, ...]   (one or more; several for "math, reading and
            ESCS"; each measure is ONE of:
              {"score": "MATH"|"READ"|"SCIE"|"CMPS"|"CPPK"|"CMOD"|"CPRO"|
                        "SEPS"|"SEDE"|"SEID"|"SENV"|"GLCM"|"FLIT"|"CRTH_NC"}
              {"variable": "ESCS"}                    a variable from the cards
              {"level_share": {"domain": "MATH", "side": "below"|"at_or_above",
                               "level": "2"}}          % below Level 2 in math
                 ("domains": ["MATH","READ","SCIE"] with "combine": "all"|"any"
                  for a joint condition; levels 1c 1b 1a 2 3 4 5 6)
              {"code_share": {"variable": "ST263Q02JA", "codes": [1, 2]}}
                 % of valid respondents whose answer is one of the codes
              {"threshold_share": {"variable": "REPEAT", "op": ">=", "value": 1}}
              {"count_share": {"variables": ["ST127Q01TA", "ST127Q02TA",
                               "ST127Q03TA"], "codes": [2], "op": "=", "count": 1}}
                 % for whom exactly (=) / at least (>=) N of the listed items
                 take one of the codes — "repeated a grade only once" = exactly
                 one of the three grade-repetition items answered "once" (2)
              {"school_mean_of": "ESCS"}              the student's school average
            ),
 "economies": ["EST", "POL"]|null   (CNT codes of the economies named; null =
            every economy — right for rankings; NEVER list a region's members
            here, use "regions"),
 "regions": ["Latin America and the Caribbean"]|null   (names from the REGIONS
            list; the app expands them per cycle),
 "filters": [{"variable": "IMMIG", "op": "in", "values": [2, 3]}, ...]|null
            (ops: =, !=, <, <=, >, >=, in, not_in, is_null, not_null; values
            are numbers; NEVER filter on STRATUM — the app handles named
            regions, cities and school networks itself),
 "by": ["CNT"] (default) or ["CNT", "ST004D01T"] — grouping variables from
            the cards; never a plan field name, never an expression,
 "contrast": {"variable": "ST004D01T", "minuend": [2], "subtrahend": [1],
              "label": "male minus female"}   (gap only; the two sides are
            LISTS of codes of ONE variable — immigrant = IMMIG [2, 3] vs
            native [1]; rural = SC001Q01TA [1, 2] vs city [4, 5, 6], with a
            label that names what is left out; variable "CNT" with one economy
            code on each side for economy A minus economy B),
 "quart_variable": "ESCS"   (quartile_means / quartile_gap / resilient_share;
            also allowed with correlation or regression to run them WITHIN
            each quarter),
 "x": measure, "y": measure   (correlation),
 "predictors": [{"variable": "ESCS", "label": "ESCS (socio-economic index)"},
                {"variable": "ST004D01T", "codes": [2], "reference": [1],
                 "label": "male (1) vs female (0)"}, ...]   (regression, up
            to 6; a categorical variable is given as codes (=1) and reference
            codes (=0) — students in neither are left out; the label states
            what a positive coefficient means),
 "level": "3"|null   (resilient_share only: the older "at or above Level N"
            definition instead of the top performance quarter),
 "ps": [10, 50, 90]|null, "upper": 90|null, "lower": 10|null,
 "row_var": str|null, "col_var": str|null, "valid_rows": [..]|null,
 "valid_cols": [..]|null   (crosstab),
 "benchmarks": ["OECD", "European Union", "All participants",
                {"label": "Mercosur", "members": ["ARG","BRA","PRY","URY"]}]|null
            (average rows appended per cycle: the unweighted mean of the
            members' estimates, the OECD convention; a benchmark named earlier
            in the conversation stays in every later plan),
 "sort": {"by": "estimate"|"change", "desc": true, "top_n": 10|null}|null
            (rankings / top-N; the app numbers the rows),
 "cycle_overrides": {"2025": {same fields, only those that differ}}|null
            (a variable that differs by cycle — gender: contrast
            {"variable": "MALE", "minuend": [1], "subtrahend": [0]} in 2025;
            by ["CNT", "MALE"] in 2025),
 "sql": str|null   (raw_sql only),
 "substitution_note": str|null, "limitation_note": str|null,
 "explanation": one sentence of what will be computed
}
"""

RULES = """
Reply with EXACTLY ONE JSON object — never an array, never several plans.
SEVERAL MEASURES asked at once ("math, reading and ESCS for X", "how did X
do" with no domain => the three scores, science last) => statistic "mean"
with every one of them in "measures"; whatever cannot be computed in the
same plan is named in limitation_note — never dropped silently.
A COMPOUND question (a computable part plus "why" / "which factors") =>
analyze the computable part and fill limitation_note (PISA is a repeated
cross-section: no causes) — never clarify because one part is out of reach.
COVERAGE: a card line "NOT collected for X" means the variable is missing
for X; analyze the economies that have it and list the others in
limitation_note; use another cycle's direct measure of the same construct
when the named cycle has none (say so in the explanation); clarify only when
no card measures the construct for the economy in any cycle. Never claim a
variable "was not collected" unless its card says so.
Several explanatory variables against ONE outcome => "regression" with
every one of them as predictors — never a question asking which one first.
A gap, quartile_gap, correlation or regression asked for SEVERAL subjects =>
run it for the first subject named (science when none) and list the others
in limitation_note.
"X and the top N" / "X compared with the best countries" => ONE ranking
(economies null, sort by estimate): the app locates X's row and rank.
"Which countries are best / rank all" => economies null, sort by estimate.
A SHARE ("% below Level 2", "share who disagree", "how many repeated a
grade") => statistic "share" with a level_share, code_share or
threshold_share measure; "top performers" = at_or_above Level 5; "low
achievers" / "Level 1 or below" = below Level 2. A condition over
SEVERAL items ("only once", "at least two of") => count_share — never
clarify that no single variable holds it. Percentages of a
questionnaire answer => code_share with the codes the cards give (a share who
DISAGREE uses the disagree codes). A share or mean WITHIN ESCS quarters
("% below Level 2 among the top ESCS quarter") => quartile_means with the
share as the measure — never a clarify about filtering by quarter.
"High vs low X" for a continuous X => quartile_gap on X (never invent
thresholds). Distribution / spread / inequality => percentiles or
percentile_spread ("P90-P10" => percentile_spread, upper 90, lower 10).
Two categorical variables against each other => crosstab. "Controlling for"
/ "after accounting for" => regression. "Standard deviation" => sd.
"Resilient students" => resilient_share (quart_variable ESCS, score
measure). "ICC" / "between-school variance" => between_school_share.
"Correlation within ESCS quarters" / "regression by ESCS quarter" =>
correlation or regression with quart_variable "ESCS".
GENDER: contrast ST004D01T minuend [2] (male) subtrahend [1] (female) in
2018/2022; in 2025 use MALE minuend [1] subtrahend [0] in cycle_overrides
(the app keeps the direction aligned). Public vs private: SC013Q01TA
minuend [2] (private) subtrahend [1] (public), instrument stu_sch.
Immigrant vs native: IMMIG minuend [1] (native) subtrahend [2, 3], label
"non-immigrant minus immigrant" (or the reverse, labelled).
DIFFERENCES BETWEEN ECONOMIES ("gap in A minus gap in B", "difference between
A and B and did it change", "which countries are statistically tied with A")
are never a reason to clarify: plan the statistic for BOTH (or all)
economies in one table (economies [A, B], by ["CNT"]; a gap stays a gap); the
app states every pairwise difference, the change in a difference and the
ties itself. Never write that the system "cannot compute" such a difference.
"The average of A, B and C" as ONE number => economies [A, B, C] plus a
benchmark {"label": …, "members": [A, B, C]} — never a pooled mean. "World
average" => benchmark "All participants" (the OECD average is a different
benchmark; never substitute one for the other).
"Which policy works best" / "effect of a policy" => the within-cycle
association (regression or correlation) with limitation_note that PISA
cannot evaluate policies causally — never a cross-cycle regression.
Sample sizes come with every answer automatically: never plan raw_sql to
count students. A period longer than the loaded cycles ("since 2012") =>
2018-2025 with a note in the explanation. When the question names no
achievement domain, use science (the 2025 major domain) and say so.
Questionnaire INDICES (WLE scales, ESCS) across cycles: plan the cycles side
by side; the app decides per index whether the change is comparable.
A stratum, region, city or school network inside an economy (Scotland,
Dubai, Nazarbayev Intellectual Schools, DKI Jakarta): plan the WHOLE economy
(economies [code]); the app adds the labelled stratum row itself.
"""


def planner_system(knowledge: str) -> str:
    """The grammar planner's system prompt: the domain knowledge shared with
    the legacy planner, then the form and the rules."""
    return knowledge.rstrip() + "\n" + SCHEMA + RULES
