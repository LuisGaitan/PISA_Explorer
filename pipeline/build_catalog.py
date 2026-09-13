"""Build the compact variable catalog that the AI layer retrieves from.

Sources, per instrument:
  - column labels + types: data/metadata/*.json (captured at conversion time)
  - value labels 2025:     embedded in the SPSS source, already captured in the
                           metadata JSON at conversion time
  - value labels 2022:     the SPSS .SAV files (embedded labels, metadata-only read)
  - value labels 2018:     .FORMAT.SAS (variable -> format name) combined with
                           the .SAS7BCAT catalogs (format name -> {value: label});
                           the catalogs need latin1 (they choke pyreadstat's
                           default encoding detection)

Outputs (all rebuildable):
  data/catalog/variables.parquet       one row per (variable, table):
      variable, table_name, cycle, instrument, label, var_type,
      value_labels (JSON string or NULL), n_value_labels
  data/catalog/comparability.parquet   one row per (instrument, variable) for
      instruments present in at least two cycles: in_2018 / in_2022 / in_2025,
      label_<cycle>, n_cycles, label_changed (any two present labels differ)
  ...and both registered as tables in data/pisa.duckdb
      (catalog_variables, catalog_comparability).

Usage: python pipeline/build_catalog.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyreadstat

from sources import CYCLES, DATA_2022, DATA_DIR, DB_PATH, SOURCES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_catalog")

CATALOG_DIR = DATA_DIR / "catalog"
# A format statement is `format` followed by one or more `VAR FMTNAME.` pairs,
# terminated by `;` — some files put one pair per statement, others hundreds.
FORMAT_BLOCK = re.compile(r"\bformat\b(.*?);", re.IGNORECASE | re.DOTALL)
FORMAT_PAIR = re.compile(r"(\w+)\s+(\$?\w+)\.")


def parse_format_sas(text: str) -> dict[str, str]:
    pairs = {}
    for block in FORMAT_BLOCK.findall(text):
        for var, fmt in FORMAT_PAIR.findall(block):
            pairs[var] = fmt.upper()
    return pairs


def clean_key(value) -> str:
    """Serialize a value-label key: 1.0 -> '1', 'MDA' -> 'MDA'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def value_labels_2022(instrument: str) -> dict[str, dict]:
    """Value labels from the matching SPSS .SAV, metadata-only."""
    target = f"CY08MSP_{instrument.upper()}.SAV"
    matches = [p for p in DATA_2022.rglob("*.SAV") if p.name.upper() == target]
    if not matches:
        log.warning(f"{instrument}_2022: no SAV found ({target})")
        return {}
    _, meta = pyreadstat.read_sav(str(matches[0]), metadataonly=True)
    return meta.variable_value_labels or {}


def value_labels_2018(sas_path: Path, instrument: str) -> dict[str, dict]:
    """Value labels via FORMAT.SAS (var->format) + SAS7BCAT (format->labels)."""
    folder = sas_path.parent
    instr = instrument.upper()

    fmt_sas = [p for p in folder.glob("*.SAS") if instr in p.name.upper()
               and "FORMAT" in p.name.upper()]
    catalogs = [p for p in folder.glob("*.SAS7BCAT") if instr in p.name.upper()]
    if not fmt_sas or not catalogs:
        log.warning(f"{instrument}_2018: FORMAT.SAS or SAS7BCAT missing in {folder}")
        return {}

    var_to_format = parse_format_sas(fmt_sas[0].read_text(encoding="latin1"))
    _, cat_meta = pyreadstat.read_sas7bcat(str(catalogs[0]), encoding="LATIN1")
    format_sets = {name.upper(): labels
                   for name, labels in (cat_meta.value_labels or {}).items()}

    out = {}
    for var, fmt in var_to_format.items():
        labels = format_sets.get(fmt) or format_sets.get("$" + fmt)
        if labels:
            out[var] = labels
    return out


def main() -> int:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for source in SOURCES:
        meta_json = source.metadata_path
        if not meta_json.exists():
            log.warning(f"{source.name}: metadata JSON missing, skipping")
            continue
        info = json.loads(meta_json.read_text(encoding="utf-8"))
        column_labels = info["column_labels"] or {}
        var_types = info["variable_types"] or {}

        if info.get("variable_value_labels"):
            vvl = info["variable_value_labels"]        # SPSS sources (2025)
        elif source.cycle == "2022":
            vvl = value_labels_2022(source.instrument)
        else:
            vvl = value_labels_2018(source.path, source.instrument)

        n_labeled = 0
        for var, label in column_labels.items():
            labels = vvl.get(var)
            if labels:
                labels = {clean_key(k): str(v) for k, v in labels.items()}
                n_labeled += 1
            rows.append({
                "variable": var,
                "table_name": source.name,
                "cycle": source.cycle,
                "instrument": source.instrument,
                "label": label or "",
                "var_type": var_types.get(var, "double"),
                "value_labels": json.dumps(labels, ensure_ascii=False) if labels else None,
                "n_value_labels": len(labels) if labels else 0,
            })
        log.info(f"{source.name}: {len(column_labels):,} variables, "
                 f"{n_labeled:,} with value labels")

    variables = pd.DataFrame(rows)
    variables.to_parquet(CATALOG_DIR / "variables.parquet", index=False)
    log.info(f"variables.parquet: {len(variables):,} rows")

    # Cross-cycle comparability for instruments present in 2+ cycles
    per_cycle = variables.groupby("instrument")["cycle"].nunique()
    shared = per_cycle[per_cycle >= 2].index
    comp_rows = []
    for instrument in shared:
        sub = variables[variables.instrument == instrument]
        labels = {c: sub[sub.cycle == c].set_index("variable").label
                  for c in CYCLES}
        all_vars = sorted(set().union(*(set(v.index) for v in labels.values())))
        for var in all_vars:
            present = [c for c in CYCLES if var in labels[c].index]
            row = {"instrument": instrument, "variable": var}
            for c in CYCLES:
                row[f"in_{c}"] = c in present
            for c in CYCLES:
                row[f"label_{c}"] = labels[c][var] if c in present else None
            norm = {labels[c][var].strip().lower() for c in present}
            row["n_cycles"] = len(present)
            row["label_changed"] = len(norm) > 1
            comp_rows.append(row)
    comparability = pd.DataFrame(comp_rows)
    comparability.to_parquet(CATALOG_DIR / "comparability.parquet", index=False)
    n_multi = int((comparability.n_cycles >= 2).sum())
    n_all = int((comparability.n_cycles == len(CYCLES)).sum())
    log.info(f"comparability.parquet: {len(comparability):,} variables across "
             f"{len(shared)} shared instruments ({n_multi:,} in 2+ cycles, "
             f"{n_all:,} in all {len(CYCLES)})")

    con = duckdb.connect(str(DB_PATH))
    for table, path in [("catalog_variables", CATALOG_DIR / "variables.parquet"),
                        ("catalog_comparability", CATALOG_DIR / "comparability.parquet")]:
        p = str(path).replace("'", "''")
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{p}')")
    con.close()
    log.info("registered catalog_variables + catalog_comparability in pisa.duckdb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
