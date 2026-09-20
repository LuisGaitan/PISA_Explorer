"""Viet Nam's PISA 2018 plausible values, released by the OECD separately.

The main 2018 student file (CY07MSU) holds Viet Nam's 5,377 students with
their weights, replicate weights and questionnaire answers but NO plausible
values: the OECD withheld them at the December 2019 release. It later
published them in a separate file (cy07_vnm_stu_qqq.sas7bdat: identifiers
and PV1-10 for MATH, READ, SCIE only, no weights). Every CNTSTUID in it
matches a row of the main file, so the correct load is a join: the main
file's rows keep their weights and the PVs are filled from this file.

This script converts that file to a small Parquet; pipeline/build_db.py then
defines the stu_qqq_2018 view with COALESCE(main PV, Viet Nam PV) when the
Parquet exists (and exactly as before when it does not).

    python pipeline/load_vnm_2018.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import DATA_2018, PARQUET_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load_vnm_2018")

SOURCE = DATA_2018 / "VNM" / "cy07_vnm_stu_qqq.sas7bdat"
TARGET = PARQUET_DIR / "pisa2018" / "vnm_pv.parquet"
EXPECTED_ROWS = 5_377
PV_COLUMNS = [f"PV{i}{d}" for i in range(1, 11) for d in ("MATH", "READ", "SCIE")]


def main() -> int:
    if not SOURCE.exists():
        log.error(f"source missing: {SOURCE}")
        return 1
    df, _meta = pyreadstat.read_sas7bdat(str(SOURCE), encoding="LATIN1")
    missing = [c for c in ["CNT", "CNTSTUID"] + PV_COLUMNS if c not in df.columns]
    if missing:
        log.error(f"columns missing from the Viet Nam file: {missing}")
        return 1
    df = df[["CNT", "CNTSTUID"] + PV_COLUMNS].copy()
    if len(df) != EXPECTED_ROWS or set(df.CNT) != {"VNM"}:
        log.error(f"unexpected content: {len(df)} rows, CNT {sorted(set(df.CNT))}")
        return 1
    if df.CNTSTUID.duplicated().any():
        log.error("duplicate CNTSTUID in the Viet Nam file")
        return 1
    df["CNTSTUID"] = df["CNTSTUID"].astype("float64")   # same type as the main file
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    tmp = TARGET.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(TARGET)
    log.info(f"{TARGET}: {len(df):,} rows, {len(df.columns)} columns "
             f"({TARGET.stat().st_size / 1024:.0f} KB). Now run pipeline/build_db.py "
             "and pipeline/build_coverage.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
