"""Shared paths and DuckDB connection for the explorer layer."""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "pisa.duckdb"
CATALOG_DIR = DATA_DIR / "catalog"


import os

# DuckDB's buffer pool defaults to ~80% of the machine's RAM. In a Cloud Run
# container that budget is shared with the ~550 MB pandas frame an
# all-economies query materializes (plus its Arrow copy), which is what blew a
# 2 GiB instance past its limit under load. Bound the pool explicitly; override
# with PISA_DUCKDB_MEMORY (e.g. "4GB" on a big workstation).
DUCKDB_MEMORY = os.environ.get("PISA_DUCKDB_MEMORY", "1GB")
DUCKDB_THREADS = int(os.environ.get("PISA_DUCKDB_THREADS", "2"))


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run pipeline/convert.py then pipeline/build_db.py"
        )
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    return con
