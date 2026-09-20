"""Build and method identifiers stamped on every answer's provenance.

A number an institution cites has to be reproducible later, after further
fixes have changed how the app plans or summarizes. Two stamps make that
possible:

  build    which deployed code produced the answer — the image tag on Cloud
           Run (PISA_BUILD, set from the build's _TAG), else the git commit
           of a local checkout, else "dev".
  method   the version of the statistical method itself. It changes ONLY
           when a computed number can change for the same plan (a new
           estimator rule, link errors loaded, a comparability rule). Prompt
           and prose fixes do not bump it. History below.
"""

import os
import subprocess
from pathlib import Path

METHOD_VERSION = "4"
# 1  weighted mean / proportion / gap / quartiles / correlation / percentiles /
#    crosstab / regression: W_FSTUWT, 10 PVs (Rubin), Fay-BRR k=0.5 x 80.
# 2  cross-cycle change SEs include the OECD link errors for mean scores
#    (explorer/link_errors.py); empty cycles dropped, not crashed on.
# 3  within-cycle-standardized indices (WLE scales, ESCS) get no cross-cycle
#    change; group-average rows (OECD / region / ad-hoc) = unweighted mean of
#    member estimates with SE = sqrt(sum SE^2) / N over ALL members.
# 4  estimates resting on fewer than 30 students are suppressed (OECD
#    reporting minimum); shares written as CASE ... ELSE 0 keep NULL as NULL
#    and 1/0 fractions are rescaled to percentages; Viet Nam's separately
#    released 2018 plausible values are joined into stu_qqq_2018.


def _git_short_hash() -> str | None:
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — git absent or not on PATH
        return None


BUILD = os.environ.get("PISA_BUILD") or _git_short_hash() or "dev"


def stamp() -> str:
    """One line for the provenance card: 'build v21, method v3'."""
    return f"build {BUILD}, method v{METHOD_VERSION}"
