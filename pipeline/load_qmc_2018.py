"""Moscow City (CNT 'QMC'), an additional PISA 2018 participant the OECD
released as a separate set of files (CY07MSU_QMC_*), not in the main
international database. This converts its student and school questionnaire
files to Parquet; pipeline/build_db.py then appends them to the
stu_qqq_2018 / sch_qqq_2018 views (UNION ALL BY NAME) when they exist.

    python pipeline/load_qmc_2018.py

Moscow City is then the 81st 2018 economy: it appears in rankings and
filters like Moscow Region (QMR) and Tatarstan (QRT), which are in the main
file, and is excluded from OECD averages (OECD flag 0).
"""

import logging
import sys
from pathlib import Path

import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import DATA_2018, PARQUET_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load_qmc_2018")

FILES = {
    "stu_qqq": (DATA_2018 / "QMC" / "cy07msu_qmc_stu_qqq.sas7bdat", 5_768),
    "sch_qqq": (DATA_2018 / "QMC" / "cy07msu_qmc_sch_qqq.sas7bdat", None),
}


def main() -> int:
    for instrument, (source, expected) in FILES.items():
        if not source.exists():
            log.error(f"source missing: {source}")
            return 1
        df, _meta = pyreadstat.read_sas7bdat(str(source), encoding="LATIN1")
        if set(df.CNT) != {"QMC"} or (expected and len(df) != expected):
            log.error(f"{instrument}: unexpected content ({len(df)} rows, CNT {sorted(set(df.CNT))})")
            return 1
        target = PARQUET_DIR / "pisa2018" / f"qmc_{instrument}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(target)
        log.info(f"{target.name}: {len(df):,} rows, {len(df.columns)} columns, "
                 f"{target.stat().st_size / 1e6:.1f} MB")
    log.info("now run pipeline/build_db.py and pipeline/build_coverage.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
