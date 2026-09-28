"""Shared pytest configuration.

The catalog parquet files and the DuckDB live under data/ and are not in
git, so CI has neither. A test that reaches them by accident (a grammar
compile that looks a variable up, an agent helper that reads a codebook
label) used to fail every push with FileNotFoundError. Here such a failure
becomes a skip on a machine without the data — the test still runs, and
still fails on a real problem, wherever the data exist. A test that CAN run
without the data should stub the catalog instead (see tests/test_grammar.py),
so it keeps running on CI."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    try:
        outcome.get_result()
    except FileNotFoundError as e:
        missing = str(getattr(e, "filename", "") or e)
        if str(DATA_DIR) in missing or "/data/" in missing.replace("\\", "/"):
            pytest.skip(f"needs the local data files (not in CI): {missing}")
        raise
