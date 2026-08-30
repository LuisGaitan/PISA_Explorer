"""High-level analysis templates built on the estimator.

These are the query templates the future LLM layer fills in rather than
derives: each returns a tidy long-format DataFrame with `estimate` and `se`
columns, ready for tables and charts alike.

Measures use `{pv}` for the plausible-value slot, e.g. "PV{pv}MATH".
"""

import numpy as np
import pandas as pd

from .estimator import combine, contrast, replicate_estimates


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
