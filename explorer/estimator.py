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
    select = (
        list(by)
        + [f"({e}) AS m_{i}" for i, e in enumerate(exprs, start=1)]
        + ALL_WEIGHTS
    )
    sql = f"SELECT {', '.join(select)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    df = con.sql(sql).df()

    n_pv, n_w = len(exprs), len(ALL_WEIGHTS)
    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]

    key_rows, blocks = [], []
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)      # (n, 81)
        estimates = np.empty((n_pv, n_w))
        for i, col in enumerate(measure_cols):
            m = g[col].to_numpy(dtype=float)                # (n,)
            mask = ~np.isnan(m)
            wm = weights[mask]
            estimates[i] = (wm.T @ m[mask]) / wm.sum(axis=0)
        key_rows.append(key)
        blocks.append(estimates)

    pv_idx = np.repeat(np.arange(1, n_pv + 1), n_w)
    rep_idx = np.tile(np.arange(n_w), n_pv)
    frames = []
    for key, est in zip(key_rows, blocks):
        frame = pd.DataFrame({"pv": pv_idx, "rep": rep_idx, "value": est.ravel()})
        for col, val in zip(by, key):
            frame[col] = val
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out[list(by) + ["pv", "rep", "value"]]


def combine(replicates: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Rubin + Fay-BRR combination of a replicate frame (possibly of derived
    statistics). Returns [*by, estimate, se, n_pv]."""
    group_cols = list(by) if by else []

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
        return pd.Series(
            {"estimate": main.mean(), "se": float(np.sqrt(total_var)), "n_pv": m}
        )

    if group_cols:
        out = (
            replicates.groupby(group_cols, dropna=False, observed=True)
            .apply(_one, include_groups=False)
            .reset_index()
        )
    else:
        out = _one(replicates).to_frame().T
    out["n_pv"] = out["n_pv"].astype(int)
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
    return combine(merged[keys + ["value"]], by=by)
