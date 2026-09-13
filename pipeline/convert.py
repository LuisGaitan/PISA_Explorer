"""Raw OECD file -> Parquet converter for the PISA 2018/2022/2025 databases
(SAS7BDAT for 2018/2022, SPSS .sav for 2025).

Design rules (each one fixes a defect found in the previous pipeline):
  1. ATOMIC WRITES: every Parquet file is written to `<name>.parquet.tmp` and
     renamed onto the final name only after the row count is verified. A crash
     can never leave a plausible-looking partial file.
  2. SELF-VALIDATING: the SAS file header records its own row count
     (meta.number_rows); the written Parquet must match it exactly, and must
     match the official public-use count in sources.py where one is known.
  3. STREAMING: files are read in chunks sized to the column count, so even
     the 15 GB cognitive files never come close to exhausting RAM. The chunk
     loop is explicit (row_offset/row_limit) rather than pyreadstat's
     read_file_in_chunks helper, which drops the encoding override that the
     2025 questionnaire-timing file needs.
  4. METADATA CAPTURED: column labels and value labels are dumped to JSON at
     conversion time — the raw material for the future variable catalog.

Usage:
  python pipeline/convert.py                  # convert everything not yet done
  python pipeline/convert.py --only 2025      # one cycle
  python pipeline/convert.py --only stu_qqq_2018 sch_qqq_2022
  python pipeline/convert.py --force          # reconvert even if output exists
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from sources import METADATA_DIR, Source, get_sources, read_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("convert")

# Target ~1.6 GB of raw float64 cells per chunk; clamp to a sane row range.
CHUNK_CELL_BUDGET = 200_000_000


def choose_chunk_size(n_columns: int) -> int:
    return max(10_000, min(100_000, CHUNK_CELL_BUDGET // max(n_columns, 1)))


def build_schema(df: pd.DataFrame, variable_types: dict[str, str]) -> pa.Schema:
    """Fixed Arrow schema from readstat variable types, so every chunk is cast
    identically (readstat: numeric -> 'double', character -> 'string')."""
    fields = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            fields.append(pa.field(col, pa.timestamp("us")))
        elif variable_types.get(col) == "double":
            fields.append(pa.field(col, pa.float64()))
        else:
            fields.append(pa.field(col, pa.string()))
    return pa.schema(fields)


def to_arrow(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    for field in schema:
        if field.type == pa.string():
            col = df[field.name]
            if col.dtype != object or col.isna().any():
                df[field.name] = col.astype(object).where(col.notna(), None)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


def read_chunks(source: Source, chunk_size: int):
    """Yield DataFrame chunks of the raw file in order (any format)."""
    reader = pyreadstat.read_sav if source.fmt == "sav" else pyreadstat.read_sas7bdat
    kwargs = {"encoding": source.encoding} if source.encoding else {}
    offset = 0
    while True:
        df, _ = reader(str(source.path), row_offset=offset, row_limit=chunk_size,
                       **kwargs)
        if len(df) == 0:
            return
        yield df
        if len(df) < chunk_size:
            return
        offset += len(df)


def dump_metadata(source: Source, meta, n_rows: int, n_cols: int) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": source.name,
        "cycle": source.cycle,
        "instrument": source.instrument,
        "source_file": str(source.path),
        "format": source.fmt,
        "rows": n_rows,
        "columns": n_cols,
        "column_labels": meta.column_names_to_labels,
        "variable_value_labels": meta.variable_value_labels,
        "variable_types": meta.readstat_variable_types,
    }
    with open(source.metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def convert_source(source: Source) -> dict:
    if not source.path.exists():
        raise FileNotFoundError(f"{source.name}: source missing: {source.path}")

    # Header-only read: authoritative row count + labels, costs seconds.
    meta = read_metadata(source)
    header_rows = meta.number_rows
    n_cols = len(meta.column_names)
    chunk_size = choose_chunk_size(n_cols)
    size_gb = source.path.stat().st_size / 1024**3

    expected = source.expected_rows
    if expected is not None and header_rows is not None and expected != header_rows:
        raise RuntimeError(
            f"{source.name}: file header says {header_rows:,} rows but official "
            f"count is {expected:,} — investigate before converting"
        )
    target_rows = expected if expected is not None else header_rows

    log.info(
        f"{source.name}: {size_gb:.2f} GB, {n_cols:,} columns, "
        f"{'?' if target_rows is None else format(target_rows, ',')} rows expected, "
        f"chunk={chunk_size:,}"
    )

    source.parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = source.parquet_path.with_name(source.parquet_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    total_rows = 0
    started = time.time()
    try:
        for df_chunk in read_chunks(source, chunk_size):
            if writer is None:
                schema = build_schema(df_chunk, meta.readstat_variable_types)
                writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
            writer.write_table(to_arrow(df_chunk, schema))
            total_rows += len(df_chunk)
            log.info(f"{source.name}:   {total_rows:,} rows written")
    finally:
        if writer is not None:
            writer.close()

    if target_rows is not None and total_rows != target_rows:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source.name}: wrote {total_rows:,} rows, expected {target_rows:,} "
            f"— temp file deleted, nothing published"
        )

    os.replace(tmp_path, source.parquet_path)
    dump_metadata(source, meta, total_rows, n_cols)

    out_gb = source.parquet_path.stat().st_size / 1024**3
    elapsed = time.time() - started
    log.info(
        f"{source.name}: DONE {total_rows:,} rows -> {out_gb:.2f} GB parquet "
        f"in {elapsed/60:.1f} min"
    )
    return {"name": source.name, "rows": total_rows, "columns": n_cols,
            "parquet_gb": round(out_gb, 2), "seconds": round(elapsed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="names/instruments/cycles to convert")
    parser.add_argument("--force", action="store_true", help="reconvert existing outputs")
    args = parser.parse_args()

    sources = get_sources(args.only)
    if not sources:
        log.error(f"no sources match {args.only!r}")
        return 2

    results, failures = [], []
    for source in sources:
        if source.parquet_path.exists() and not args.force:
            log.info(f"{source.name}: already converted, skipping (--force to redo)")
            continue
        try:
            results.append(convert_source(source))
        except Exception:
            log.exception(f"{source.name}: FAILED")
            failures.append(source.name)

    log.info("=" * 60)
    for r in results:
        log.info(f"  OK  {r['name']:<14} {r['rows']:>9,} rows  {r['parquet_gb']:>6.2f} GB")
    for name in failures:
        log.info(f"  FAIL {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
