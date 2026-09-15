"""Per-economy coverage of every questionnaire variable, per cycle.

PISA's optional questionnaires (well-being, ICT familiarity, financial
literacy, parent, teacher, "una hora", Learning in the Digital World …) are
administered by a SUBSET of economies, and so are many national options. In
the public-use files the columns exist for everyone but are entirely missing
for economies that did not administer them — the well-being index EXPWB, for
example, has values for 15 of the 80 economies in 2022 and none for the
United States. The variable catalog cannot see that; this table can.

Output (rebuildable, small):
  data/catalog/coverage.parquet   one row per (variable, table):
      variable, table_name, cycle, instrument,
      n_economies      economies in the table
      n_with_data      economies with at least one non-missing value
      with_data        space-joined codes with data   (only when partial)
      missing          space-joined codes without data (only when partial)

The explorer reads it via explorer.catalog.coverage(); a missing file simply
disables coverage checks. Re-run after any data rebuild:
    python pipeline/build_coverage.py          (~2-4 min: one pass per table)
"""

import logging
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import CYCLES, DATA_DIR, DB_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_coverage")

CATALOG_DIR = DATA_DIR / "catalog"
INSTRUMENTS = ("stu_qqq", "sch_qqq")
CHUNK = 120          # columns per GROUP BY pass (bounds memory on wide tables)


def table_coverage(con, table: str) -> list[dict]:
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position", [table]).fetchall()]
    if "CNT" not in cols:
        log.warning(f"{table}: no CNT column, skipped")
        return []
    economies = sorted(r[0] for r in con.execute(
        f"SELECT DISTINCT CNT FROM {table}").fetchall())
    instrument, cycle = table.rsplit("_", 1)
    counts: dict[str, pd.Series] = {}
    targets = [c for c in cols if c != "CNT"]
    for i in range(0, len(targets), CHUNK):
        chunk = targets[i:i + CHUNK]
        aggs = ", ".join(f'count("{c}") AS "{c}"' for c in chunk)
        df = con.execute(f'SELECT CNT, {aggs} FROM {table} GROUP BY CNT').df()
        df = df.set_index("CNT")
        for c in chunk:
            counts[c] = df[c]
    rows = []
    for var, series in counts.items():
        with_data = sorted(series.index[series > 0])
        partial = len(with_data) < len(economies)
        missing = sorted(set(economies) - set(with_data)) if partial else []
        rows.append({
            "variable": var, "table_name": table, "cycle": cycle,
            "instrument": instrument,
            "n_economies": len(economies), "n_with_data": len(with_data),
            "with_data": " ".join(with_data) if partial else None,
            "missing": " ".join(missing) if partial else None,
        })
    n_partial = sum(1 for r in rows if r["with_data"] is not None)
    log.info(f"{table}: {len(rows):,} variables, {n_partial:,} partially "
             f"administered, {len(economies)} economies")
    return rows


def main() -> int:
    if not DB_PATH.exists():
        log.error(f"{DB_PATH} not found — run pipeline/build_db.py first")
        return 1
    con = duckdb.connect(str(DB_PATH), read_only=True)
    started = time.time()
    rows: list[dict] = []
    for cycle in CYCLES:
        for instrument in INSTRUMENTS:
            table = f"{instrument}_{cycle}"
            try:
                con.execute(f"SELECT 1 FROM {table} LIMIT 0")
            except duckdb.Error:
                log.warning(f"{table}: not in the database, skipped")
                continue
            rows.extend(table_coverage(con, table))
    out = pd.DataFrame(rows)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = CATALOG_DIR / "coverage.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    log.info(f"{path.name}: {len(out):,} rows, {path.stat().st_size / 1024:.0f} KB, "
             f"{time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
