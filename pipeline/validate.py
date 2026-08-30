"""Validate the converted Parquet files against official PISA figures.

Checks, per file:
  - Parquet row count == official public-use count (where known)
  - Parquet row count == row count recorded in the SAS source header
  - Student questionnaires: all key columns present (CNT, IDs, PV1-PV10 for
    MATH/READ/SCIE, W_FSTUWT, the 80 replicate weights, ESCS, STRATUM) and
    exactly 80 distinct economies

Exit code 0 only if every check passes.

Usage: python pipeline/validate.py [--only 2018 stu_qqq ...]
"""

import argparse
import sys

import duckdb
import pyarrow.parquet as pq
import pyreadstat

from sources import (
    EXPECTED_ECONOMIES_STU_QQQ,
    STU_QQQ_KEY_COLUMNS,
    get_sources,
)


def validate_source(source, check_header: bool) -> list[str]:
    """Return a list of failure strings; empty list means the file passed."""
    problems = []
    if not source.parquet_path.exists():
        return [f"parquet missing: {source.parquet_path}"]

    meta = pq.read_metadata(source.parquet_path)
    rows = meta.num_rows
    columns = set(meta.schema.names)

    if source.expected_rows is not None and rows != source.expected_rows:
        problems.append(f"rows {rows:,} != official {source.expected_rows:,}")

    if check_header:
        _, sas_meta = pyreadstat.read_sas7bdat(str(source.sas_path), metadataonly=True)
        if sas_meta.number_rows is not None and rows != sas_meta.number_rows:
            problems.append(f"rows {rows:,} != SAS header {sas_meta.number_rows:,}")

    if source.instrument == "stu_qqq":
        missing = [c for c in STU_QQQ_KEY_COLUMNS if c not in columns]
        if missing:
            problems.append(f"missing key columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        n_economies = duckdb.sql(
            f"SELECT count(DISTINCT CNT) FROM read_parquet('{source.parquet_path}')"
        ).fetchone()[0]
        if n_economies != EXPECTED_ECONOMIES_STU_QQQ:
            problems.append(f"{n_economies} economies != {EXPECTED_ECONOMIES_STU_QQQ}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--skip-header-check", action="store_true",
                        help="skip re-reading SAS headers (faster)")
    args = parser.parse_args()

    failed = 0
    for source in get_sources(args.only):
        problems = validate_source(source, check_header=not args.skip_header_check)
        if problems:
            failed += 1
            print(f"FAIL {source.name}")
            for p in problems:
                print(f"       - {p}")
        else:
            rows = pq.read_metadata(source.parquet_path).num_rows
            ncols = pq.read_metadata(source.parquet_path).num_columns
            print(f"OK   {source.name:<14} {rows:>9,} rows  {ncols:>5,} cols")

    print("-" * 50)
    print("ALL CHECKS PASSED" if failed == 0 else f"{failed} FILE(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
