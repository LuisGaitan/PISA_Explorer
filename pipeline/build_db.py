"""Build data/pisa.duckdb over the converted Parquet files.

Each converted instrument becomes a VIEW named `<instrument>_<cycle>`
(e.g. stu_qqq_2018, stu_cog_2022) over its Parquet file — zero duplication,
instantly rebuilt by rerunning this script. The small ESCS trend file is
materialized as a real table `escs_trend`.

Note: 2018, 2022 and 2025 keep their own per-cycle tables on purpose. The
cycles' questionnaires overlap but are not identical (variables were added,
dropped, and renamed between cycles), so cross-cycle comparability is decided
per variable at query time, not forced at the schema level.

Usage: python pipeline/build_db.py
"""

import logging
import sys

import duckdb

from sources import DB_PATH, ESCS_TREND_CSV, SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_db")


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    built, missing = [], []
    for source in SOURCES:
        if not source.parquet_path.exists():
            missing.append(source.name)
            continue
        parquet = str(source.parquet_path).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {source.name} AS "
            f"SELECT * FROM read_parquet('{parquet}')"
        )
        built.append(source.name)

    catalog_dir = DB_PATH.parent / "catalog"
    for table, filename in (("catalog_variables", "variables.parquet"),
                            ("catalog_comparability", "comparability.parquet")):
        path = catalog_dir / filename
        if path.exists():
            p = str(path).replace("'", "''")
            con.execute(f"CREATE OR REPLACE TABLE {table} AS "
                        f"SELECT * FROM read_parquet('{p}')")
            log.info(f"{table}: registered")

    if ESCS_TREND_CSV.exists():
        csv = str(ESCS_TREND_CSV).replace("'", "''")
        con.execute("DROP TABLE IF EXISTS escs_trend")
        con.execute(
            f"CREATE TABLE escs_trend AS SELECT * FROM read_csv_auto('{csv}')"
        )
        n = con.execute("SELECT count(*) FROM escs_trend").fetchone()[0]
        log.info(f"escs_trend: materialized {n:,} rows")
    else:
        log.warning(f"escs_trend: CSV not found at {ESCS_TREND_CSV}")

    con.close()

    log.info(f"{DB_PATH}: {len(built)} views created")
    for name in built:
        log.info(f"  view {name}")
    for name in missing:
        log.info(f"  (skipped {name} — no parquet yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
