"""Survey-correct estimation engine for PISA.

Implements the OECD-prescribed methodology exactly:
  - population statistics weighted by the final student weight W_FSTUWT
  - achievement statistics computed once per plausible value (PV1..PV10)
    and averaged (Rubin's rules for the imputation variance)
  - sampling variance by Fay's Balanced Repeated Replication with k = 0.5
    over the 80 replicate weights W_FSTURWT1..80:
        U = (1 / (G * k^2)) * sum_g (T_g - T)^2   with G=80, k=0.5  ->  U = sum/20
  - total variance = mean sampling variance over PVs
                     + (1 + 1/M) * between-PV variance   (M = number of PVs)

The engine computes every statistic under all 81 weights x all PVs in a single
DuckDB scan (one wide aggregate query), then combines in pandas. A measure is
any SQL expression; write `{pv}` where the plausible-value number belongs
(e.g. "PV{pv}MATH"); expressions without `{pv}` are treated as a single
implicate (M = 1, no imputation variance).
"""

import numpy as np
import pandas as pd

N_REPLICATES = 80
FAY_K = 0.5
VAR_FACTOR = 1.0 / (N_REPLICATES * FAY_K**2)  # = 1/20
N_PV = 10

FULL_WEIGHT = "W_FSTUWT"
REPLICATE_WEIGHTS = [f"W_FSTURWT{i}" for i in range(1, N_REPLICATES + 1)]
ALL_WEIGHTS = [FULL_WEIGHT] + REPLICATE_WEIGHTS  # index 0 = full sample


def _pv_list(measure: str) -> list[str]:
    if "{pv}" in measure:
        return [measure.format(pv=i) for i in range(1, N_PV + 1)]
    return [measure]


def replicate_estimates(
    con,
    table: str,
    measure: str,
    by: tuple[str, ...] = (),
    where: str | None = None,
) -> pd.DataFrame:
    """Weighted mean of `measure` for every (group, pv, weight) combination.

    Returns a long DataFrame: [*by, pv (1-based), rep (0 = full weight,
    1..80 = replicates), value]. Rows where the measure is NULL are excluded
    from both numerator and denominator (weighted mean of observed values).

    DuckDB projects just the needed columns (measure expressions evaluated in
    SQL); the 10 PV x 81 weight estimate grid is then one masked matrix
    product per group in numpy — hundreds of times faster than expressing
    the 810 aggregates in SQL.
    """
    exprs = _pv_list(measure)
    measure_cols = [f"m_{i}" for i in range(1, len(exprs) + 1)]
    df = fetch_frame(con, table, exprs, by=by, where=where)
    return replicates_from_frame(df, measure_cols, by=by)


def fetch_frame(
    con, table: str, exprs: list[str], by=(), where: str | None = None,
    extra: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Project just what the estimator needs: group columns, each measure
    expression as m_1..m_k, optional named extra expressions, all 81 weights."""
    select = (
        list(by)
        + [f"({e}) AS m_{i}" for i, e in enumerate(exprs, start=1)]
        + [f"({e}) AS {name}" for name, e in (extra or {}).items()]
        + ALL_WEIGHTS
    )
    sql = f"SELECT {', '.join(select)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return con.sql(sql).df()


def replicates_from_frame(
    df: pd.DataFrame, measure_cols: list[str], by=()
) -> pd.DataFrame:
    """The estimation core, on an already-fetched frame (which may carry
    derived columns, e.g. a weighted-quartile assignment)."""
    n_pv, n_w = len(measure_cols), len(ALL_WEIGHTS)
    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]

    key_rows, blocks, counts = [], [], []
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)      # (n, 81)
        estimates = np.empty((n_pv, n_w))
        n_obs = 0
        for i, col in enumerate(measure_cols):
            m = g[col].to_numpy(dtype=float)                # (n,)
            mask = ~np.isnan(m)
            n_obs = max(n_obs, int(mask.sum()))
            wm = weights[mask]
            # A group with no observed values yields NaN by design (e.g. a
            # question not administered there) — suppress the 0/0 warnings.
            with np.errstate(invalid="ignore", divide="ignore"):
                estimates[i] = (wm.T @ m[mask]) / wm.sum(axis=0)
        key_rows.append(key)
        blocks.append(estimates)
        counts.append(n_obs)

    pv_idx = np.repeat(np.arange(1, n_pv + 1), n_w)
    rep_idx = np.tile(np.arange(n_w), n_pv)
    frames = []
    for key, est, n_obs in zip(key_rows, blocks, counts):
        frame = pd.DataFrame({"pv": pv_idx, "rep": rep_idx, "value": est.ravel(), "n": n_obs})
        for col, val in zip(by, key):
            frame[col] = val
        frames.append(frame)
    if not frames:      # no rows matched (e.g. an economy absent from this cycle)
        return pd.DataFrame(columns=list(by) + ["pv", "rep", "value", "n"])
    out = pd.concat(frames, ignore_index=True)
    return out[list(by) + ["pv", "rep", "value", "n"]]


def combine(replicates: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Rubin + Fay-BRR combination of a replicate frame (possibly of derived
    statistics). Returns [*by, estimate, se, n_pv] plus `n` (students with an
    observed value) when the replicate frame carries it — the basis of the
    OECD's minimum-size reporting rule."""
    group_cols = list(by) if by else []
    has_n = "n" in replicates.columns
    if replicates.empty:
        return pd.DataFrame(columns=group_cols + ["estimate", "se", "n_pv"] + (["n"] if has_n else []))

    def _one(group: pd.DataFrame) -> pd.Series:
        main = group[group.rep == 0].set_index("pv").value  # T_v per PV
        reps = group[group.rep > 0]
        # sampling variance per PV, then averaged
        u = (
            reps.merge(main.rename("t"), left_on="pv", right_index=True)
            .assign(sq=lambda d: (d.value - d.t) ** 2)
            .groupby("pv")
            .sq.sum()
            * VAR_FACTOR
        )
        m = len(main)
        b = main.var(ddof=1) if m > 1 else 0.0  # between-PV (imputation) variance
        total_var = u.mean() + (1 + 1 / m) * b if m > 1 else u.mean()
        estimate = main.mean()
        # no observed values (item not administered): no estimate, no SE —
        # never "blank (SE 0.0)"
        se = float(np.sqrt(total_var)) if not np.isnan(estimate) else float("nan")
        out = {"estimate": estimate, "se": se, "n_pv": m}
        if has_n:
            out["n"] = int(group["n"].min())
        return pd.Series(out)

    if group_cols:
        out = (
            replicates.groupby(group_cols, dropna=False, observed=True)
            .apply(_one, include_groups=False)
            .reset_index()
        )
    else:
        out = _one(replicates).to_frame().T
    out["n_pv"] = out["n_pv"].astype(int)
    if has_n:
        out["n"] = out["n"].astype(int)
    return out


def contrast(
    replicates: pd.DataFrame,
    group_col: str,
    minuend,
    subtrahend,
    by: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Replicate-wise difference (minuend group - subtrahend group), combined
    with full PV+BRR error propagation — the correct SE for a gap, because the
    two groups share the same sample and replicate structure."""
    a = replicates[replicates[group_col] == minuend]
    b = replicates[replicates[group_col] == subtrahend]
    keys = list(by) + ["pv", "rep"]
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    merged["value"] = merged.value_a - merged.value_b
    cols = keys + ["value"]
    if "n_a" in merged.columns:
        merged["n"] = merged[["n_a", "n_b"]].min(axis=1)   # the smaller group bounds the rule
        cols.append("n")
    return combine(merged[cols], by=by)
