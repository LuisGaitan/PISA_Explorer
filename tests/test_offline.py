"""Checks that need neither the OECD data nor an API key: the cross-cycle
trend arithmetic, the Fay-BRR/Rubin combination on a synthetic replicate
frame, the institution-name normalizer, and the SQL guards."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PISA_EVENTS_BACKEND", "jsonl")

from explorer.analysis import trend                     # noqa: E402
from explorer.estimator import ALL_WEIGHTS, combine     # noqa: E402


def _result(cnt, est, se):
    return pd.DataFrame({"CNT": cnt, "estimate": est, "se": se, "n_pv": 10})


def test_trend_three_cycles_change_is_last_minus_first():
    r18 = _result(["FIN", "USA"], [520.0, 505.0], [2.3, 3.6])
    r22 = _result(["FIN", "USA", "KHM"], [490.0, 504.0, 380.0], [2.3, 4.3, 3.0])
    r25 = _result(["FIN", "KHM"], [480.0, 390.0], [2.0, 3.1])
    out = trend({"2018": r18, "2022": r22, "2025": r25}, by=("CNT",)).set_index("CNT")
    assert list(out.columns) == ["estimate_2018", "se_2018", "estimate_2022", "se_2022",
                                 "estimate_2025", "se_2025", "change", "se_change"]
    assert out.loc["FIN", "change"] == pytest.approx(-40.0)
    assert out.loc["FIN", "se_change"] == pytest.approx(np.sqrt(2.3**2 + 2.0**2))
    # an economy missing from a cycle keeps its row (outer join), with NaN change
    assert np.isnan(out.loc["KHM", "estimate_2018"])
    assert np.isnan(out.loc["KHM", "change"])
    assert np.isnan(out.loc["USA", "estimate_2025"])


def test_trend_legacy_two_frame_call_still_works():
    r18 = _result(["FIN"], [520.0], [2.0])
    r22 = _result(["FIN"], [490.0], [2.0])
    out = trend(r18, r22, by=("CNT",))
    assert out.change.iloc[0] == pytest.approx(-30.0)
    assert "estimate_2025" not in out.columns


def test_combine_rubin_and_fay_brr_variance():
    """One group, 2 PVs, 80 replicates: sampling variance = sum(sq)/20,
    imputation variance = var(PV means), total = mean(U) + (1 + 1/M) * B."""
    rows = []
    rng = np.random.default_rng(0)
    for pv, main in ((1, 500.0), (2, 502.0)):
        rows.append({"pv": pv, "rep": 0, "value": main})
        for rep in range(1, len(ALL_WEIGHTS)):
            rows.append({"pv": pv, "rep": rep, "value": main + rng.normal(0, 4)})
    reps = pd.DataFrame(rows)
    out = combine(reps)
    assert out.estimate.iloc[0] == pytest.approx(501.0)
    u = []
    for pv, main in ((1, 500.0), (2, 502.0)):
        r = reps[(reps.pv == pv) & (reps.rep > 0)].value
        u.append(((r - main) ** 2).sum() / 20)
    b = np.var([500.0, 502.0], ddof=1)
    assert out.se.iloc[0] == pytest.approx(np.sqrt(np.mean(u) + 1.5 * b))
    assert int(out.n_pv.iloc[0]) == 2


def test_clean_institution_normalizes_and_rejects():
    from explorer.app import clean_institution
    assert clean_institution("  Universit%C3%A4t   Wien ") == "Universität Wien"
    assert clean_institution("Penn GSE") == "Penn GSE"
    assert clean_institution("X") is None
    assert clean_institution("") is None
    assert len(clean_institution("A" * 200)) == 80


def test_sql_guards_reject_writes_and_multiple_statements():
    from explorer.agent import Agent
    with pytest.raises(ValueError):
        Agent._check_fragment("CNT = 'USA'; DROP TABLE x")
    with pytest.raises(ValueError):
        Agent._check_identifier("CNT; DROP")
    Agent._check_fragment("CNT IN ('USA','FIN')")   # fine
    Agent._check_identifier("ST004D01T")             # fine
