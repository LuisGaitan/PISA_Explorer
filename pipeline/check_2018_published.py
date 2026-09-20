"""Reproduce the PISA 2018 Results (Volume I) Table I.1 snapshot — the
published mean reading, mathematics and science score of every economy —
from the converted data (StatLink EDU-2019-4228-EN-T001.XLSX, downloaded
from the OECD; put it in the PISA 2018 raw folder).

Together with check_2025_published.py this pins the 2018 cycle to the OECD's
own figures: the whole chain (conversion, DuckDB views, PV x Fay-BRR
estimator, and — since Viet Nam's separately released plausible values were
joined in — the stu_qqq_2018 view) must reproduce each mean within tolerance.

Usage: python pipeline/check_2018_published.py [--tolerance 0.05]
Exit code 0 when every published mean is matched.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from explorer.analysis import weighted_mean            # noqa: E402
from explorer.db import connect                        # noqa: E402
from pipeline.check_2025_published import NAME_TO_CNT  # noqa: E402
from pipeline.sources import DATA_2018                 # noqa: E402

TABLE = DATA_2018 / "EDU-2019-4228-EN-T001.XLSX"
# 2018-only names in the snapshot table
NAMES_2018 = {**NAME_TO_CNT, "Russia": "RUS", "Baku (Azerbaijan)": "QAZ", "Ukraine": "UKR",
              "Belarus": "BLR", "Bosnia and Herzegovina": "BIH", "Moldova": "MDA",
              "North Macedonia": "MKD", "Czech Republic": "CZE", "Turkey": "TUR",
              "Slovak Republic": "SVK", "Korea": "KOR", "Viet Nam": "VNM",
              "Brunei Darussalam": "BRN", "Panama": "PAN", "Jordan": "JOR",
              "Malta": "MLT", "Montenegro": "MNE", "Serbia": "SRB", "Albania": "ALB"}
COLUMNS = {"READ": 1, "MATH": 3, "SCIE": 5}       # mean columns in Table I.1


def load_published() -> pd.DataFrame:
    raw = pd.read_excel(TABLE, sheet_name="Table I.1", header=None)
    rows = []
    for _, r in raw.iterrows():
        name = str(r[0]).strip()
        if name in ("nan", "OECD average") or name not in NAMES_2018:
            continue
        for dom, col in COLUMNS.items():
            val = r[col]
            if isinstance(val, (int, float)) and pd.notna(val):
                rows.append({"CNT": NAMES_2018[name], "name": name, "domain": dom,
                             "published": float(val)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    if not TABLE.exists():
        print(f"published table not found: {TABLE}")
        return 1
    published = load_published()
    con = connect(read_only=True)
    computed = {}
    for dom in COLUMNS:
        res = weighted_mean(con, "stu_qqq_2018", f"PV{{pv}}{dom}", by=("CNT",))
        computed[dom] = res.set_index("CNT")["estimate"]
    bad = 0
    for r in published.itertuples():
        have = computed[r.domain].get(r.CNT)
        if have is None or pd.isna(have):
            print(f"MISSING {r.name} {r.domain}: published {r.published:.2f}, computed none")
            bad += 1
            continue
        diff = abs(float(have) - r.published)
        if diff > args.tolerance:
            print(f"FAIL    {r.name} {r.domain}: published {r.published:.3f}, computed {float(have):.3f}")
            bad += 1
    n = len(published)
    print(f"{n - bad} of {n} published 2018 means reproduced within {args.tolerance} points"
          f" ({published.CNT.nunique()} economies)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
