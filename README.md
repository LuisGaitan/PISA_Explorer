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
  sources.py       registry of raw SAS files + official row counts
  convert.py       SAS7BDAT -> Parquet (chunked, atomic writes, self-validating)
  build_db.py      builds data/pisa.duckdb (views over Parquet + escs_trend table)
  build_catalog.py variable catalog + value labels + cross-cycle comparability map
  validate.py      checks row counts, key columns, economy counts
explorer/
  catalog.py       keyword retrieval over the 23k-variable catalog (search/describe/comparability)
  estimator.py     PV x Fay-BRR replicate engine (the survey-methodology core)
  analysis.py      templates: weighted_mean, weighted_proportion, gap, trend,
                   quartile_means, quartile_gap (weighted within-group quartiles),
                   correlation (weighted, PV-aware, BRR SE)
  demo.py          end-to-end demo:  python -m explorer.demo
  llm.py           Gemini API client (key from GEMINI_API_KEY env var or .env file)
  agent.py         question -> retrieval -> plan -> validated execution -> provenance
  chat.py          terminal chat:  python -m explorer.chat ["question"]
  app.py           local web app:  python -m explorer.app  ->  http://127.0.0.1:8765
  events.py        usage analytics store (Firestore on Cloud Run, JSONL locally)
  static/index.html  the web UI (gate, chat, charts with 95% CI, tables, provenance, feedback, CSV export)
  static/charts.js   shared SVG chart library (bars/league, dumbbell, diverging, heatmap)
  static/admin.html  usage dashboard at /admin (PISA_ADMIN_CODE)
data/              (gitignored) parquet/, metadata/, catalog/, pisa.duckdb — fully rebuildable
```

Raw data stays where it is, read-only, and is never committed:

- 2018: `C:\Users\Luis\Desktop\DataMining\PISA_Data2018`
- 2022: `C:\Users\Luis\Desktop\DataMining\PISA_Data2022`

## Rebuild from scratch

```
pip install -r requirements.txt
python pipeline/convert.py       # ~43 GB SAS -> ~3.7 GB Parquet, ~25 min total
python pipeline/build_db.py
python pipeline/build_catalog.py
python pipeline/validate.py
python -m explorer.demo          # end-to-end check (~1 s)
```

## Chat with the data

Put your Gemini key in a `.env` file at the repo root (gitignored):

```
GEMINI_API_KEY=your-key-here
```

Then:

```
python -m explorer.chat "How did reading scores change in Finland between 2018 and 2022?"
python -m explorer.chat            # interactive; /export file.csv saves the last table
```

Every answer prints its provenance: source tables, the variables used with
their codebook labels, the filter, the number of students behind each number
(and the population they represent), and the exact statistical method. The
LLM only fills in validated analysis templates — retrieval feeds it ~40
catalog cards, never a schema dump; raw SQL (the escape hatch) is read-only,
SELECT-only, and always displayed. Variable substitutions are always stated,
never silent.

## Web app

```
python -m explorer.app           # opens on http://127.0.0.1:8765 (local only)
```

Same engine as the CLI, plus charts: country comparisons render as bars with
95%-confidence whiskers, 2018→2022 questions as dumbbells (2018 → 2022 dots),
gaps as diverging bars around zero, and crosstabs as heatmaps. Every chart has
hover tooltips (value, SE, CI, significance), sits above its full data table
(Export CSV button), and carries the same provenance card. Light/dark follows
the OS. `?demo=1` renders sample charts offline without spending API calls.
Conversations have memory (follow-ups and clarification answers work), and
each browser session keeps its own history.

### Analysis templates

All survey-correct (`W_FSTUWT`, 10 PVs via Rubin's rules, Fay-BRR SEs over the
80 replicate weights); the LLM fills parameters, never derives the statistics:

| Template | Question shape |
|---|---|
| `weighted_mean` | averages, shares above/below proficiency cutoffs |
| `weighted_proportion` | % in a category, missing codes excluded |
| `gap` | group differences (e.g. gender gap), replicate-wise SE |
| `quartile_means` / `quartile_gap` | means per weighted quarter of a continuous index; top-vs-bottom equity gaps (ESCS gradient) |
| `correlation` | weighted Pearson r, PV-aware |
| `percentiles` / `percentile_spread` | weighted P10…P90; P90−P10 dispersion |
| `crosstab` | weighted two-way row % with per-cell SEs, value-labeled |
| `regression` | weighted least squares, readable term names, "controlling for" questions |
| `raw_sql` | read-only SELECT escape hatch, clearly flagged as unweighted |

Cross-cycle (2018 vs 2022) versions of all of these run automatically when the
question compares cycles.

## Deployment

The public-sharing setup (Cloud Run, access code, per-session and global rate
limits, self-contained Docker image) is in [DEPLOY.md](DEPLOY.md). Environment
knobs: `PISA_ACCESS_CODE`, `PISA_RATE_LIMIT` (default 20/h per session),
`PISA_GLOBAL_RATE` (default 200/h total), `PORT`, `BIND_HOST`.

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

## Catalog and statistics layer

`catalog_variables` (23,307 rows, one per variable per table, ~95% with value
labels) and `catalog_comparability` (cross-cycle availability of 16,233
variables in the 8 shared instruments) live both as DuckDB tables and as
Parquet under `data/catalog/`. `explorer.catalog.search()` is the retrieval
function the AI layer will call — only the ~15 relevant variables ever reach
a prompt. 2018 value labels are recovered by combining each instrument's
`.FORMAT.SAS` (variable -> format name) with its `.SAS7BCAT` catalog
(format -> labels, latin1); 2022 labels come from the SPSS files.

## Methodology rules (implemented in `explorer/estimator.py`)

- Population statistics always use the final student weight `W_FSTUWT`.
- Point estimates for achievement average over all 10 plausible values.
- Standard errors: Fay's BRR (k = 0.5) over `W_FSTURWT1`–`W_FSTURWT80`,
  combined with between-PV imputation variance (Rubin's rules).
- Group gaps are differenced replicate-wise (correct covariance handling);
  cross-cycle trends add variances (independent samples; OECD link error not
  yet included — flagged in `analysis.trend`).
- **Validated:** country means AND standard errors reproduce the published
  PISA 2018/2022 figures (e.g. USA 2022 math 465 SE 4.0, FIN 484 SE 1.9,
  KOR 527 SE 3.9).
- Codes stay in the data; labels come from the catalog and are applied at
  display time.
