"""Shared paths and DuckDB connection for the explorer layer."""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "pisa.duckdb"
CATALOG_DIR = DATA_DIR / "catalog"


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found — run pipeline/convert.py then pipeline/build_db.py"
        )
    return duckdb.connect(str(DB_PATH), read_only=read_only)
