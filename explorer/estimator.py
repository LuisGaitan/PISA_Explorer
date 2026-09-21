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
SCHOOL_ID = "CNTSCHID"
# Per-group descriptors that travel with every replicate frame and survive
# combine(): n = students with an observed value, n_schools = distinct
# schools among them, wcov = share of the group's weighted population with an
# observed value (1.0 when nothing is missing).
GROUP_STATS = ["n", "n_schools", "wcov"]


def group_stats(g: pd.DataFrame, mask: np.ndarray) -> dict:
    """n, n_schools and weighted coverage of the observed rows of a group."""
    w = g[FULL_WEIGHT].to_numpy(dtype=float)
    total = float(w.sum())
    out = {"n": int(mask.sum()),
           "wcov": float(w[mask].sum() / total) if total > 0 else 1.0}
    out["n_schools"] = -1              # unknown: the school floor is skipped
    if SCHOOL_ID in g.columns and mask.any():
        ids = g[SCHOOL_ID].to_numpy()[mask]
        known = ids[~pd.isna(ids)]
        # a file with no school identifier at all for this group (five 2025
        # economies) says nothing about how many schools were sampled
        if known.size:
            out["n_schools"] = int(pd.unique(known).size)
    return out


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
    tail = f" FROM {table}" + (f" WHERE {where}" if where else "")
    # The school identifier rides along (never returned to users) so the
    # OECD's 5-school reporting floor can be applied; tables without it
    # (some cognitive/timing files) simply skip that floor.
    if SCHOOL_ID not in by:
        try:
            return con.sql(f"SELECT {', '.join(select + [SCHOOL_ID])}{tail}").df()
        except Exception:  # noqa: BLE001 — no school id in this table
            pass
    return con.sql(f"SELECT {', '.join(select)}{tail}").df()


def replicates_from_frame(
    df: pd.DataFrame, measure_cols: list[str], by=()
) -> pd.DataFrame:
    """The estimation core, on an already-fetched frame (which may carry
    derived columns, e.g. a weighted-quartile assignment)."""
    n_pv, n_w = len(measure_cols), len(ALL_WEIGHTS)
    groups = df.groupby(list(by), dropna=False, observed=True) if by else [((), df)]

    key_rows, blocks, stats = [], [], []
    for key, g in groups:
        if not isinstance(key, tuple):
            key = (key,)
        weights = g[ALL_WEIGHTS].to_numpy(dtype=float)      # (n, 81)
        estimates = np.empty((n_pv, n_w))
        first_mask = None
        for i, col in enumerate(measure_cols):
            m = g[col].to_numpy(dtype=float)                # (n,)
            mask = ~np.isnan(m)
            if first_mask is None:
                first_mask = mask
            wm = weights[mask]
            # A group with no observed values yields NaN by design (e.g. a
            # question not administered there) — suppress the 0/0 warnings.
            with np.errstate(invalid="ignore", divide="ignore"):
                estimates[i] = (wm.T @ m[mask]) / wm.sum(axis=0)
        key_rows.append(key)
        blocks.append(estimates)
        stats.append(group_stats(g, first_mask))

    pv_idx = np.repeat(np.arange(1, n_pv + 1), n_w)
    rep_idx = np.tile(np.arange(n_w), n_pv)
    frames = []
    for key, est, st in zip(key_rows, blocks, stats):
        frame = pd.DataFrame({"pv": pv_idx, "rep": rep_idx, "value": est.ravel(), **st})
        for col, val in zip(by, key):
            frame[col] = val
        frames.append(frame)
    if not frames:      # no rows matched (e.g. an economy absent from this cycle)
        return pd.DataFrame(columns=list(by) + ["pv", "rep", "value"] + GROUP_STATS)
    out = pd.concat(frames, ignore_index=True)
    return out[list(by) + ["pv", "rep", "value"] + GROUP_STATS]


def combine(replicates: pd.DataFrame, by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Rubin + Fay-BRR combination of a replicate frame (possibly of derived
    statistics). Returns [*by, estimate, se, n_pv] plus `n` (students with an
    observed value) when the replicate frame carries it — the basis of the
    OECD's minimum-size reporting rule."""
    group_cols = list(by) if by else []
    stat_cols = [c for c in GROUP_STATS if c in replicates.columns]
    has_n = bool(stat_cols)
    if replicates.empty:
        return pd.DataFrame(columns=group_cols + ["estimate", "se", "n_pv"] + stat_cols)

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
        for c in stat_cols:                    # the smaller side bounds a rule
            out[c] = float(group[c].min())
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
    for c in ("n", "n_schools"):
        if c in out.columns:
            out[c] = out[c].astype(int)
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
    # outer: a group that is absent (no private schools sampled) gives a
    # blank difference for that row instead of dropping the row silently
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), how="outer")
    merged["value"] = merged.value_a - merged.value_b
    cols = keys + ["value"]
    for c in GROUP_STATS:
        if f"{c}_a" in merged.columns:
            merged[c] = merged[[f"{c}_a", f"{c}_b"]].min(axis=1)   # the smaller group bounds the rule
            cols.append(c)
    return combine(merged[cols], by=by)
