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
        vnm = source.parquet_path.parent / "vnm_pv.parquet"
        if source.name == "stu_qqq_2018" and vnm.exists():
            # Viet Nam's 2018 plausible values were released separately
            # (pipeline/load_vnm_2018.py); fill them into the main file's
            # rows, which keep their weights. Other economies are untouched.
            pvs = [f"PV{i}{d}" for i in range(1, 11) for d in ("MATH", "READ", "SCIE")]
            replace = ", ".join(f'COALESCE(s."{c}", v."{c}") AS "{c}"' for c in pvs)
            vnm_path = str(vnm).replace("'", "''")
            con.execute(
                f"CREATE OR REPLACE VIEW {source.name} AS "
                f"SELECT s.* REPLACE ({replace}) FROM read_parquet('{parquet}') s "
                f"LEFT JOIN read_parquet('{vnm_path}') v "
                f"ON s.CNT = v.CNT AND s.CNTSTUID = v.CNTSTUID"
            )
            log.info(f"{source.name}: Viet Nam 2018 plausible values joined in")
        else:
            con.execute(
                f"CREATE OR REPLACE VIEW {source.name} AS "
                f"SELECT * FROM read_parquet('{parquet}')"
            )
        built.append(source.name)

    # stu_sch_<cycle>: every student row joined to its school's questionnaire
    # (one row per school; CNT + CNTSCHID is unique). LEFT JOIN keeps students
    # whose school has no questionnaire row (school variables NULL there).
    # Student weights stay valid: this is the OECD's own setup for "public vs
    # private" style comparisons of student outcomes by school characteristics.
    for cycle in sorted({s.cycle for s in SOURCES}):
        stu, sch = f"stu_qqq_{cycle}", f"sch_qqq_{cycle}"
        if stu not in built or sch not in built:
            continue
        cols = lambda t: [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position", [t]).fetchall()]
        stu_cols = set(cols(stu))
        sch_only = [c for c in cols(sch) if c not in stu_cols]
        sch_select = ", ".join(f'h."{c}"' for c in sch_only)
        con.execute(
            f"CREATE OR REPLACE VIEW stu_sch_{cycle} AS "
            f"SELECT s.*, {sch_select} FROM {stu} s LEFT JOIN {sch} h "
            f"ON s.CNT = h.CNT AND s.CNTSCHID = h.CNTSCHID"
        )
        built.append(f"stu_sch_{cycle}")
        log.info(f"stu_sch_{cycle}: joined view, {len(sch_only)} school columns")

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
