"""High-level analysis templates built on the estimator.

These are the query templates the future LLM layer fills in rather than
derives: each returns a tidy long-format DataFrame with `estimate` and `se`
columns, ready for tables and charts alike.

Measures use `{pv}` for the plausible-value slot, e.g. "PV{pv}MATH".
"""

import numpy as np
import pandas as pd

from .estimator import (
    ALL_WEIGHTS,
    GROUP_STATS,
    _pv_list,
    combine,
    contrast,
    fetch_frame,
    group_stats,
    replicate_estimates,
    replicates_from_frame,
)


def weighted_mean(con, table, measure, by=(), where=None) -> pd.DataFrame:
    """Weighted mean of a measure (PV-aware), with BRR standard errors."""
    reps = replicate_estimates(con, table, measure, by=tuple(by), where=where)
    return combine(reps, by=tuple(by))


def weighted_proportion(con, table, variable, value, by=(), where=None,
                        valid_values=None) -> pd.DataFrame:
    """Percentage of (valid) respondents with `variable` = `value`.

    `valid_values`: iterable restricting the denominator to valid response
    codes (PISA questionnaire variables use high codes like 95/97/98/99 for
    missing categories — pass the valid codes to exclude them).
    """
    if valid_values is not None:
        codes = ", ".join(str(v) for v in valid_values)
        expr = (f"CASE WHEN {variable} = {value} THEN 100.0 "
                f"WHEN {variable} IN ({codes}) THEN 0.0 END")
    else:
        # missing codes are NULL in this database: the denominator is the
        # non-missing responses, as in the OECD's recode + [aw] convention
        expr = (f"CASE WHEN {variable} = {value} THEN 100.0 "
                f"WHEN {variable} IS NOT NULL THEN 0.0 END")
    reps = replicate_estimates(con, table, expr, by=tuple(by), where=where)
    return combine(reps, by=tuple(by))


def gap(con, table, measure, group_col, minuend, subtrahend, by=(),
        where=None, group_expr: str | None = None,
        group_label: str | None = None) -> pd.DataFrame:
    """Difference in the weighted mean between two groups (e.g. gender gap),
    with the statistically correct SE (replicate-wise differencing).

    `group_expr`: a SQL expression defining the two groups when they are not
    a single column's codes — e.g. non-immigrant (IMMIG = 1) vs immigrant
    (IMMIG in 2, 3): "CASE WHEN IMMIG = 1 THEN 1 WHEN IMMIG IN (2, 3) THEN 0
    END" with minuend=1, subtrahend=0. `group_col` is then only a label."""
    if group_expr:
        from .estimator import _pv_list, fetch_frame, replicates_from_frame
        exprs = _pv_list(measure)
        cols = [f"m_{i}" for i in range(1, len(exprs) + 1)]
        df = fetch_frame(con, table, exprs, by=by, where=where,
                         extra={"_grp": group_expr})
        reps = replicates_from_frame(df, cols, by=("_grp", *by))
        out = contrast(reps, "_grp", minuend, subtrahend, by=tuple(by))
        out.insert(0, "contrast", f"{group_label or group_col}: {minuend} - {subtrahend}")
        return out
    reps = replicate_estimates(
        con, table, measure, by=(group_col, *by), where=where
    )
    out = contrast(reps, group_col, minuend, subtrahend, by=tuple(by))
    out.insert(0, "contrast", f"{group_col}: {minuend} - {subtrahend}")
    return out


def _assign_weighted_quarters(df: pd.DataFrame, by=()) -> pd.DataFrame:
    """Add a `quarter` column (1=bottom .. 4=top) from the `qv` column, using
    W_FSTUWT-weighted quartiles computed within each `by` group — the OECD
    convention (e.g. ESCS quarters are weighted within each economy)."""
    qv_all = df["qv"].to_numpy(dtype=float)
    w_all = df["W_FSTUWT"].to_numpy(dtype=float)
    quarters = np.empty(len(df), dtype=np.int64)

    def assign(pos: np.ndarray) -> None:
        qv, w = qv_all[pos], w_all[pos]
        order = np.argsort(qv, kind="stable")
        frac = np.empty(len(qv))
        cw = np.cumsum(w[order])
        frac[order] = (cw - 0.5 * w[order]) / cw[-1]   # midpoint convention
        quarters[pos] = 1 + np.searchsorted([0.25, 0.5, 0.75], frac)

    if by:
        for pos in df.groupby(list(by), dropna=False, observed=True).indices.values():
            assign(np.asarray(pos))
    else:
        assign(np.arange(len(df)))
    df["quarter"] = quarters
    return df


def _quartile_replicates(con, table, measure, quart_variable, by=(), where=None):
    exprs = _pv_list(measure)
    df = fetch_frame(con, table, exprs, by=tuple(by), where=where,
                     extra={"qv": quart_variable})
    # Weighted coverage of the quartile variable per group BEFORE dropping
    # students without it (USA 2025: 43% of the weighted population has no
    # ESCS) — the quarters describe only the students who have a value.
    has_qv = df["qv"].notna().to_numpy()
    if by:
        w_all = df.groupby(list(by), dropna=False, observed=True)["W_FSTUWT"].sum()
        w_obs = df[has_qv].groupby(list(by), dropna=False, observed=True)["W_FSTUWT"].sum()
        coverage = (w_obs / w_all).fillna(0.0)
    else:
        total = float(df["W_FSTUWT"].sum())
        coverage = float(df.loc[has_qv, "W_FSTUWT"].sum() / total) if total else 1.0
    df = df[has_qv].reset_index(drop=True)
    # Quarters are defined within each ECONOMY (the OECD's escs_q is cut per
    # country), then results are broken down by any further grouping — a
    # "bottom quarter" must mean the same ESCS range for boys and girls.
    df = _assign_weighted_quarters(df, by=("CNT",) if "CNT" in by else ())
    cols = [f"m_{i}" for i in range(1, len(exprs) + 1)]
    reps = replicates_from_frame(df, cols, by=(*by, "quarter"))
    if not reps.empty:
        if by:
            keys = pd.MultiIndex.from_frame(reps[list(by)]) if len(by) > 1 else reps[by[0]]
            reps["wcov"] = coverage.reindex(keys).to_numpy()
        else:
            reps["wcov"] = coverage
    return reps


def quartile_means(con, table, measure, quart_variable, by=(),
                   where=None) -> pd.DataFrame:
    """Weighted mean of `measure` within each weighted quarter (1=bottom,
    4=top) of `quart_variable`, quartiles computed within each `by` group."""
    reps = _quartile_replicates(con, table, measure, quart_variable, by, where)
    return combine(reps, by=(*by, "quarter"))


def quartile_gap(con, table, measure, quart_variable, by=(),
                 where=None) -> pd.DataFrame:
    """Top-quarter minus bottom-quarter difference in `measure` (e.g. the
    ESCS equity gradient), SE via replicate-wise differencing."""
    reps = _quartile_replicates(con, table, measure, quart_variable, by, where)
    out = contrast(reps, "quarter", 4, 1, by=tuple(by))
    out.insert(0, "contrast", f"{quart_variable}: top quarter - bottom quarter")
    return out


def correlation(con, table, x, y, by=(), where=None) -> pd.DataFrame:
    """Weighted Pearson correlation of two variables, PV-aware (write {pv} in
    either expression; PV_i of one pairs with PV_i of the other), SE via BRR +
    Rubin's rules — unlike a raw-SQL corr(), this is a population estimate."""
    xs, ys = _pv_list(x), _pv_list(y)
    n_pv = max(len(xs), len(ys))
    if len(xs) not in (1, n_pv) or len(ys) not in (1, n_pv):
        raise ValueError("PV-measure lengths do not align")
    xs = xs * n_pv if len(xs) == 1 else xs
    ys = ys * n_pv if len(ys) == 1 else ys

    extra = {}
    for i, (ex, ey) in enumerate(zip(xs, ys), start=1):
        extra[f"x_{i}"] = ex
        extra[f"y_{i}"] = ey
    df = fetch_frame(con, table, [], by=tuple(by), where=where, extra=extra)

    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]
    frames = []
    n_w = len(ALL_WEIGHTS)
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)
        rs = np.empty((n_pv, n_w))
        stats = None
        for i in range(1, n_pv + 1):
            xv = g[f"x_{i}"].to_numpy(dtype=float)
            yv = g[f"y_{i}"].to_numpy(dtype=float)
            mask = ~(np.isnan(xv) | np.isnan(yv))
            if stats is None:
                stats = group_stats(g, mask)
            wm, xm, ym = weights[mask], xv[mask], yv[mask]
            # A group with no complete pairs (construct not administered)
            # yields NaN by design — suppress the expected 0/0 warnings.
            with np.errstate(invalid="ignore", divide="ignore"):
                total = wm.sum(axis=0)
                mx, my = (wm.T @ xm) / total, (wm.T @ ym) / total
                cov = (wm.T @ (xm * ym)) / total - mx * my
                vx = (wm.T @ (xm * xm)) / total - mx * mx
                vy = (wm.T @ (ym * ym)) / total - my * my
                rs[i - 1] = cov / np.sqrt(vx * vy)
        frame = pd.DataFrame({
            "pv": np.repeat(np.arange(1, n_pv + 1), n_w),
            "rep": np.tile(np.arange(n_w), n_pv),
            "value": rs.ravel(), **stats,
        })
        for col, val in zip(by, key):
            frame[col] = val
        frames.append(frame)
    if not frames:
        return combine(pd.DataFrame(columns=list(by) + ['pv', 'rep', 'value'] + GROUP_STATS), by=())
    reps = pd.concat(frames, ignore_index=True)
    return combine(reps, by=tuple(by))


def _percentile_replicates(con, table, measure, by=(), where=None,
                           ps=(10, 25, 50, 75, 90)) -> pd.DataFrame:
    """Weighted empirical percentiles (first value whose cumulative weight
    reaches p) per group, PV, and replicate weight."""
    exprs = _pv_list(measure)
    cols = [f"m_{i}" for i in range(1, len(exprs) + 1)]
    df = fetch_frame(con, table, exprs, by=tuple(by), where=where)
    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]

    n_w = len(ALL_WEIGHTS)
    frames = []
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)
        stats = None
        for pv_i, col in enumerate(cols, start=1):
            v = g[col].to_numpy(dtype=float)
            mask = ~np.isnan(v)
            if stats is None:
                stats = group_stats(g, mask)
            vm, wm = v[mask], weights[mask]
            order = np.argsort(vm, kind="stable")
            vs, cum = vm[order], np.cumsum(wm[order], axis=0)   # (n,), (n, 81)
            totals = cum[-1]
            for p in ps:
                idx = np.clip((cum < totals * (p / 100)).sum(axis=0), 0, len(vs) - 1)
                frame = pd.DataFrame({
                    "percentile": p, "pv": pv_i,
                    "rep": np.arange(n_w), "value": vs[idx], **stats,
                })
                for c, val in zip(by, key):
                    frame[c] = val
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(by) + ['percentile', 'pv', 'rep', 'value'] + GROUP_STATS)
    return pd.concat(frames, ignore_index=True)


def percentiles(con, table, measure, by=(), where=None,
                ps=(10, 25, 50, 75, 90)) -> pd.DataFrame:
    """Weighted percentiles of a measure per group, with BRR SEs."""
    reps = _percentile_replicates(con, table, measure, by, where, ps)
    return combine(reps, by=(*by, "percentile"))


def percentile_spread(con, table, measure, by=(), where=None,
                      upper=90, lower=10) -> pd.DataFrame:
    """P<upper> minus P<lower> — the within-group dispersion / inequality
    measure (P90-P10 by default), SE via replicate-wise differencing."""
    reps = _percentile_replicates(con, table, measure, by, where, (lower, upper))
    out = contrast(reps, "percentile", upper, lower, by=tuple(by))
    out.insert(0, "contrast", f"P{upper} - P{lower}")
    return out


MAX_CROSSTAB_CATEGORIES = 12


def crosstab(con, table, row_var, col_var, by=(), where=None,
             valid_rows=None, valid_cols=None) -> pd.DataFrame:
    """Weighted two-way table: within each category of `row_var`, the
    percentage of (valid) respondents in each category of `col_var` (row
    percentages sum to ~100). Each cell carries its own BRR SE."""
    def observed(var, valid):
        if valid:
            return [v for v in valid]
        vals = [r[0] for r in con.sql(
            f"SELECT DISTINCT {var} FROM {table} WHERE {var} IS NOT NULL"
            + (f" AND ({where})" if where else "") + f" ORDER BY {var}").fetchall()]
        if len(vals) > MAX_CROSSTAB_CATEGORIES:
            raise ValueError(
                f"{var} has {len(vals)} categories (max {MAX_CROSSTAB_CATEGORIES}) "
                "— pass the valid response codes explicitly")
        return vals

    row_cats = observed(row_var, valid_rows)
    col_cats = observed(col_var, valid_cols)
    row_list = ", ".join(str(v) for v in row_cats)
    where_rows = f"{row_var} IN ({row_list})"
    full_where = f"({where}) AND {where_rows}" if where else where_rows

    parts = []
    for c in col_cats:
        res = weighted_proportion(con, table, col_var, c, by=(*by, row_var),
                                  where=full_where, valid_values=col_cats)
        res["col"] = c
        parts.append(res)
    out = pd.concat(parts, ignore_index=True).rename(columns={row_var: "row"})
    return out[list(by) + ["row", "col", "estimate", "se", "n_pv"] + [c for c in GROUP_STATS if c in out.columns]]


MAX_REGRESSION_PREDICTORS = 6


def regression(con, table, y, xs, by=(), where=None,
               names=None) -> pd.DataFrame:
    """Weighted least-squares regression of `y` on predictors `xs`
    (SQL expressions; encode categorical contrasts as 0/1 CASE dummies).
    PV-aware in y; coefficients averaged over PVs, SEs via BRR + Rubin.
    Listwise deletion of rows with any missing value. `names` (parallel to
    `xs`) supplies readable term labels — without them a CASE dummy's raw SQL
    becomes the term name, which readers (and summarizers) misinterpret."""
    if len(xs) > MAX_REGRESSION_PREDICTORS:
        raise ValueError(f"at most {MAX_REGRESSION_PREDICTORS} predictors")
    if any("{pv}" in x for x in xs):
        raise ValueError("{pv} is only supported in y")
    if names is not None and len(names) != len(xs):
        raise ValueError("predictor_names must match predictors in length")
    ys_list = _pv_list(y)
    extra = {f"y_{i}": e for i, e in enumerate(ys_list, start=1)}
    extra.update({f"x_{j}": e for j, e in enumerate(xs, start=1)})
    df = fetch_frame(con, table, [], by=tuple(by), where=where, extra=extra)

    terms = ["(intercept)"] + list(names if names is not None else xs)
    n_w = len(ALL_WEIGHTS)
    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]
    frames = []
    skipped: list[tuple[str, str]] = []
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)
        x_mat = np.column_stack(
            [np.ones(len(g))] + [g[f"x_{j}"].to_numpy(dtype=float)
                                 for j in range(1, len(xs) + 1)])
        stats = None
        for pv_i in range(1, len(ys_list) + 1):
            yv = g[f"y_{pv_i}"].to_numpy(dtype=float)
            mask = ~np.isnan(yv) & ~np.isnan(x_mat).any(axis=1)
            if stats is None:
                stats = group_stats(g, mask)
            x_use, y_use, w_use = x_mat[mask], yv[mask], weights[mask]
            if mask.sum() <= len(terms):
                # name a predictor (or the outcome) with no values at all in
                # this group — "not administered", not "insufficient data"
                empty = [terms[j + 1] for j in range(x_mat.shape[1] - 1)
                         if np.isnan(x_mat[:, j + 1]).all()]
                if np.isnan(yv).all():
                    problem = "the outcome has no values in this group (not administered)"
                elif empty:
                    problem = f"{', '.join(empty)} has no values in this group (not administered there)"
                else:
                    problem = "fewer complete observations than predictors"
            else:
                # A predictor that is constant in this group (a dummy for a
                # category nobody is in, a variable with one value) or a linear
                # combination of the others makes the normal equations
                # singular. Name the offender instead of "Singular matrix".
                problem = None
                if np.linalg.matrix_rank(x_use) < x_use.shape[1]:
                    spread = x_use[:, 1:].std(axis=0)
                    flat = [terms[j + 1] for j, s in enumerate(spread) if s == 0]
                    problem = (f"{', '.join(flat)} has a single value" if flat else
                               "two or more predictors are linearly dependent (e.g. dummies "
                               "for every category of one variable)")
            if problem:
                # one economy that cannot be estimated must not abort the
                # other 80: skip it and report why
                skipped.append((", ".join(str(k) for k in key) if by else "this group", problem))
                break
            # 81 weighted normal-equation solves in one einsum each
            xtwx = np.einsum("nw,ni,nj->wij", w_use, x_use, x_use, optimize=True)
            xtwy = np.einsum("nw,ni,n->wi", w_use, x_use, y_use, optimize=True)
            betas = np.linalg.solve(xtwx, xtwy)          # (81, k)
            for t_i, term in enumerate(terms):
                frame = pd.DataFrame({
                    "term": term, "pv": pv_i,
                    "rep": np.arange(n_w), "value": betas[:, t_i], **stats,
                })
                for c, val in zip(by, key):
                    frame[c] = val
                frames.append(frame)
    if not frames:
        if skipped:
            raise ValueError("the regression cannot be estimated: " +
                             "; ".join(f"{who}: {why}" for who, why in skipped[:3]) +
                             ". Drop that predictor or leave one category out.")
        return combine(pd.DataFrame(columns=list(by) + ['term', 'pv', 'rep', 'value'] + GROUP_STATS), by=())
    reps = pd.concat(frames, ignore_index=True)
    out = combine(reps, by=(*by, "term"))
    out.attrs["skipped"] = skipped
    return out


def trend(results, result_2022: pd.DataFrame | None = None,
          link_error: float | None = None,
          by=()) -> pd.DataFrame:
    """Cross-cycle table for already-combined results with matching groups.

    `results` maps cycle -> result frame in chronological order, e.g.
    {"2018": r18, "2022": r22, "2025": r25} (the legacy two-frame call
    trend(r18, r22, by=...) still works). The output carries
    estimate_<cycle> / se_<cycle> for every cycle plus `change` and
    `se_change` = LAST cycle minus FIRST cycle. Groups missing from a cycle
    (e.g. an economy that first joined in 2025) keep their row with NULLs
    for that cycle, so the table says what is missing instead of hiding it.

    Cycles are independent samples, so var(change) = var_first + var_last,
    plus the square of `link_error` when one is given: a float for every
    row (the OECD's published link error for a mean score), or a DataFrame
    with the key columns and a `link_error` column for row-specific values
    (a proficiency-share link error derived per economy). The caller decides
    when a link error applies: it does for levels of the score scale, it
    cancels for differences between groups measured in the same cycle.
    """
    if isinstance(results, pd.DataFrame):
        results = {"2018": results, "2022": result_2022}
    cycles = list(results)
    if len(cycles) < 2:
        raise ValueError("trend needs at least two cycles")
    keys = list(by)
    merged = None
    for cycle in cycles:
        part = results[cycle].rename(
            columns={"estimate": f"estimate_{cycle}", "se": f"se_{cycle}"})
        part = part[[c for c in part.columns if c not in ("n_pv",) + tuple(GROUP_STATS)]]
        if merged is None:
            merged = part
        elif keys:
            merged = merged.merge(part, on=keys, how="outer")
        else:
            merged = (merged.assign(_k=1).merge(part.assign(_k=1), on="_k")
                      .drop(columns="_k"))
    first, last = cycles[0], cycles[-1]
    merged["change"] = merged[f"estimate_{last}"] - merged[f"estimate_{first}"]
    # Independent samples: var_first + var_last (+ the OECD link error for
    # the cycle pair and domain when it is known — see explorer/link_errors.py)
    if isinstance(link_error, pd.DataFrame):
        le_keys = [c for c in link_error.columns if c != "link_error"]
        merged = merged.merge(link_error, on=le_keys, how="left") if le_keys else \
            merged.assign(link_error=float(link_error["link_error"].iloc[0]))
        le_sq = merged.pop("link_error").fillna(0.0) ** 2
    else:
        le_sq = float(link_error or 0.0) ** 2
    merged["se_change"] = np.sqrt(merged[f"se_{first}"] ** 2
                                  + merged[f"se_{last}"] ** 2
                                  + le_sq)
    cols = keys + [c for cycle in cycles
                   for c in (f"estimate_{cycle}", f"se_{cycle}")] \
        + ["change", "se_change"]
    return merged[cols]
