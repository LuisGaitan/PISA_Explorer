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
    _pv_list,
    combine,
    contrast,
    fetch_frame,
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
        expr = f"CASE WHEN {variable} = {value} THEN 100.0 ELSE 0.0 END"
    reps = replicate_estimates(con, table, expr, by=tuple(by), where=where)
    return combine(reps, by=tuple(by))


def gap(con, table, measure, group_col, minuend, subtrahend, by=(),
        where=None) -> pd.DataFrame:
    """Difference in the weighted mean between two groups (e.g. gender gap),
    with the statistically correct SE (replicate-wise differencing)."""
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
    clause = f"({quart_variable}) IS NOT NULL"
    where = f"({where}) AND {clause}" if where else clause
    df = fetch_frame(con, table, exprs, by=tuple(by), where=where,
                     extra={"qv": quart_variable})
    df = _assign_weighted_quarters(df, by=tuple(by))
    cols = [f"m_{i}" for i in range(1, len(exprs) + 1)]
    return replicates_from_frame(df, cols, by=(*by, "quarter"))


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
        for i in range(1, n_pv + 1):
            xv = g[f"x_{i}"].to_numpy(dtype=float)
            yv = g[f"y_{i}"].to_numpy(dtype=float)
            mask = ~(np.isnan(xv) | np.isnan(yv))
            wm, xm, ym = weights[mask], xv[mask], yv[mask]
            total = wm.sum(axis=0)
            mx, my = (wm.T @ xm) / total, (wm.T @ ym) / total
            cov = (wm.T @ (xm * ym)) / total - mx * my
            vx = (wm.T @ (xm * xm)) / total - mx * mx
            vy = (wm.T @ (ym * ym)) / total - my * my
            rs[i - 1] = cov / np.sqrt(vx * vy)
        frame = pd.DataFrame({
            "pv": np.repeat(np.arange(1, n_pv + 1), n_w),
            "rep": np.tile(np.arange(n_w), n_pv),
            "value": rs.ravel(),
        })
        for col, val in zip(by, key):
            frame[col] = val
        frames.append(frame)
    reps = pd.concat(frames, ignore_index=True)
    return combine(reps, by=tuple(by))


def trend(result_2018: pd.DataFrame, result_2022: pd.DataFrame,
          by=()) -> pd.DataFrame:
    """2022 minus 2018 for two already-combined results with matching groups.

    Cycles are independent samples, so var(diff) = var18 + var22. NOTE: this
    excludes the OECD link error (the extra uncertainty from scale equating
    across cycles); comparisons against published trend SEs will be slightly
    smaller. Add the link-error term when absolute trend inference matters.
    """
    keys = list(by)
    merged = result_2018.merge(result_2022, on=keys, suffixes=("_2018", "_2022")) \
        if keys else result_2018.assign(_k=1).merge(
            result_2022.assign(_k=1), on="_k", suffixes=("_2018", "_2022")
        ).drop(columns="_k")
    merged["change"] = merged.estimate_2022 - merged.estimate_2018
    merged["se_change"] = np.sqrt(merged.se_2018**2 + merged.se_2022**2)
    cols = keys + ["estimate_2018", "se_2018", "estimate_2022", "se_2022",
                   "change", "se_change"]
    return merged[cols]
