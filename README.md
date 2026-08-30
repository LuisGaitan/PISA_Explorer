# PISA Explorer

Foundation for a conversational PISA analysis tool built on the **official OECD
PISA 2018 (CY07MSU) and 2022 (CY08MSP) public-use databases** — both cycles,
all 80 economies, with the survey weights, plausible values, and replicate
weights that make population statistics methodologically correct.

Full background, audit findings, and the agreed architecture live in
[PISA_PROJECT_HANDOFF.md](PISA_PROJECT_HANDOFF.md).

## Layout

```
pipeline/
  sources.py    registry of raw SAS files + official row counts
  convert.py    SAS7BDAT -> Parquet (chunked, atomic writes, self-validating)
  build_db.py   builds data/pisa.duckdb (views over Parquet + escs_trend table)
  validate.py   checks row counts, key columns, economy counts
data/           (gitignored) parquet/, metadata/, pisa.duckdb — fully rebuildable
```

Raw data stays where it is, read-only, and is never committed:

- 2018: `C:\Users\Luis\Desktop\DataMining\PISA_Data2018`
- 2022: `C:\Users\Luis\Desktop\DataMining\PISA_Data2022`

## Rebuild from scratch

```
pip install -r requirements.txt
python pipeline/convert.py       # ~43 GB SAS -> ~3.7 GB Parquet, ~25 min total
python pipeline/build_db.py
python pipeline/validate.py
```

`convert.py` is idempotent (skips finished files; `--force` to redo,
`--only 2022` / `--only stu_qqq_2018` to filter). Every Parquet file is written
to a temp name and renamed only after its row count matches both the SAS header
and the official figure — a crash cannot leave a plausible-looking partial file
(the defect that silently truncated the old CSV pipeline).

## Database

`data/pisa.duckdb` exposes one view per instrument per cycle:

| View | Content |
|---|---|
| `stu_qqq_2018` / `stu_qqq_2022` | Student questionnaire (PV1–PV10 MATH/READ/SCIE, `W_FSTUWT`, `W_FSTURWT1–80`, ESCS) |
| `sch_qqq_*`, `tch_qqq_*` | School and teacher questionnaires |
| `stu_cog_*`, `stu_tim_*`, `stu_ttm_2018` | Cognitive item responses and timing |
| `flt_qqq_*`, `flt_cog_*`, `flt_tim_*`, `flt_ttm_2018` | Financial literacy |
| `crt_cog_2022` | Creative thinking (2022 only) |
| `escs_trend` | OECD comparable-ESCS trend table |

The two cycles deliberately keep separate tables: questionnaires overlap but
are not identical across cycles (variables added/dropped/renamed), so
cross-cycle comparability is a per-variable decision made at query time.

## Methodology rules (for everything built on top)

- Population statistics always use the final student weight `W_FSTUWT`.
- Point estimates for achievement average over all 10 plausible values.
- Standard errors: Fay's BRR (k = 0.5) over `W_FSTURWT1`–`W_FSTURWT80`,
  combined with between-PV imputation variance.
- Codes stay in the data; labels come from `data/metadata/*.json`
  (column + value labels captured at conversion time) and are applied at
  display time.
