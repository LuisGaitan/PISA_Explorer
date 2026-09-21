# PISA Explorer

A conversational analysis tool for the **official OECD PISA 2018 (CY07MSU),
2022 (CY08MSP) and 2025 (CY09MS) public-use databases** — three cycles, all
economies (80 in 2018 and 2022, 90 in 2025), with the survey weights,
plausible values, and replicate weights that make population statistics
methodologically correct. Ask a question in plain language; get a
survey-weighted answer with standard errors, a chart, the data table, and the
full provenance of every number.

**Live site:** https://pisa-explorer-106435871926.us-central1.run.app (no
account needed — visitors name their institution or organization once).

**For researchers:** the statistics engine (`explorer/estimator.py`,
`explorer/analysis.py`) and the data pipeline have no language-model
dependency. You can rebuild the database from the OECD files, reproduce the
Technical Report validation, and call every analysis template from Python
without an API key; only the chat layer needs one.

Supported by the Penn GSE Learning Analytics and Artificial Intelligence
program. MIT licensed — see [LICENSE](LICENSE); contributions welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md); citation in [CITATION.cff](CITATION.cff).

## Layout

```
pipeline/
  sources.py       registry of raw SAS/SPSS files + official row counts
  convert.py       SAS7BDAT / SAV -> Parquet (chunked, atomic writes, self-validating)
  build_db.py      builds data/pisa.duckdb (views over Parquet + escs_trend table)
  build_catalog.py variable catalog + value labels + cross-cycle comparability map
  validate.py      checks row counts, key columns, economy counts
  check_2025_published.py  reproduces Technical Report Annex 14 (sample sizes,
                   weighted populations, SEs of mean scores) for all 90 economies
explorer/
  catalog.py       keyword retrieval over the 29k-variable catalog (search/describe/comparability)
  estimator.py     PV x Fay-BRR replicate engine (the survey-methodology core)
  analysis.py      templates: weighted_mean, weighted_proportion, gap, trend (any 2-3 cycles),
                   quartile_means, quartile_gap (weighted within-group quartiles),
                   correlation (weighted, PV-aware, BRR SE)
  demo.py          end-to-end demo:  python -m explorer.demo
  llm.py           Gemini API client (key from GEMINI_API_KEY env var or .env file)
  agent.py         question -> retrieval -> plan -> validated execution -> provenance
  chat.py          terminal chat:  python -m explorer.chat ["question"]
  app.py           local web app:  python -m explorer.app  ->  http://127.0.0.1:8765
  events.py        usage analytics store (Firestore on Cloud Run, JSONL locally)
  static/index.html  the web UI (institution gate, chat, charts with 95% CI, tables, provenance, feedback, CSV export)
  static/charts.js   shared SVG chart library (bars/league, dumbbell, diverging, heatmap)
  static/admin.html  usage dashboard at /admin (PISA_ADMIN_CODE)
data/              (gitignored) parquet/, metadata/, catalog/, pisa.duckdb — fully rebuildable
```

## Raw data

The repository contains **no PISA data**. Download the public-use files from
the OECD PISA data pages (https://www.oecd.org/en/about/programmes/pisa/pisa-data.html):
the 2018 and 2022 SAS releases and the 2025 SPSS release, which is the only
format the OECD published for 2025 (the 2025 page also has the codebook,
compendia and Technical Report annexes used for validation). Keep them
outside the repository under one folder:

```
<PISA_RAW_ROOT>/PISA_Data2018/   STU/, SCH/, TCH/, COG/, TIM/, TTM/, FLT/       (SAS)
<PISA_RAW_ROOT>/PISA_Data2022/   STU_QQQ_SAS/, SCH_QQQ_SAS/, ..., CRT_SAS/       (SAS)
<PISA_RAW_ROOT>/PISA_Data2025/   CY09_MS_*.sav, PISA2025_Codebook.xlsx, Excel Files/
```

and point `PISA_RAW_ROOT` at that folder (environment variable, or a line in
the gitignored `.env`; per-cycle overrides `PISA_RAW_2018/2022/2025`). The raw
files are read-only sources and are never modified. The OECD data are free to
use under the [OECD terms and conditions](https://www.oecd.org/en/about/terms-conditions.html);
everything the pipeline produces is derived from those public files.

**Hardware:** the three cycles are ~88 GB of raw files and convert to ~5 GB
of Parquet in about 35 minutes; 16 GB of RAM is enough for the pipeline, and
an all-economies three-cycle query peaks around 2.4 GB.

## Rebuild from scratch

```
pip install -r requirements.txt
pytest tests/                    # offline checks, no data or key needed
python pipeline/convert.py       # ~88 GB raw -> ~5.0 GB Parquet, ~35 min total (2025 alone: ~9 min)
python pipeline/build_db.py
python pipeline/build_catalog.py
python pipeline/validate.py
python pipeline/check_2025_published.py   # 2025 SEs/sample sizes vs the Technical Report
python pipeline/load_vnm_2018.py          # Viet Nam's separately released 2018 PVs (optional file)
python pipeline/build_db.py               # again, so the 2018 view joins them in
python pipeline/build_coverage.py
python pipeline/check_2018_published.py   # 2018 means vs PISA 2018 Results Vol. I Table I.1
python -m explorer.demo          # end-to-end check (~2 s)
```

Viet Nam 2018: the main CY07MSU file has Viet Nam's students without
plausible values; the OECD released them later in a separate file
(`VNM/cy07_vnm_stu_qqq.sas7bdat`). When that file is present the 2018 view
carries them, and every answer that uses them states the OECD's caution
that Viet Nam's 2018 data "did not meet the PISA technical standards but
were accepted as largely comparable".

## Data source and terms

The data are the OECD's PISA public-use files, used under the OECD's terms
of use for the PUF. Any use of results obtained with this software must
acknowledge the source as: **"Programme for International Student Assessment
(PISA) Organisation for Economic Co-operation and Development (OECD),
Paris"**. PISA Explorer is not an OECD product and is not affiliated with
the OECD; the [OECD disclaimers](https://www.oecd.org/en/about/terms-conditions/oecd-disclaimers)
apply. The app returns aggregates only: estimates based on fewer than 30
students are suppressed (the OECD's own reporting rule), the raw-SQL path
accepts aggregate queries only, and no student- or school-level record is
ever returned or distributed. The repository contains no PISA data.

## How answers are kept exact

Every number comes from code, never from the model; the model plans an
analysis and phrases the result. Four mechanisms keep the seams honest:

- **Golden regression sets.** `tests/golden/plans.json` holds analysis plans
  and the numbers the engine must reproduce (several verified against OECD
  publications); `pytest tests/test_golden_plans.py` fails on any drift and
  needs no API key. `tests/golden/questions.json` holds real user questions
  with the route, template, variables, numbers and forbidden phrases each
  answer must satisfy; `python scripts/golden_live.py` runs them through the
  model before a deploy. `python scripts/replay_events.py` re-asks recorded
  production questions and diffs the answers.
- **App-authored statements and a prose check.** For every result the app
  writes the facts itself (`explorer/summary.py`: levels, changes, ranks,
  significance verdicts, pairwise differences) and the model may only phrase
  them. Its draft is checked mechanically — every number must exist in the
  result, every economy named must be in it, significance claims must match
  the table, and phrasings such as "not collected", "did not participate" or
  "projected" are rejected unless the app's own notes say so. A failed draft
  is regenerated once with the problems named; if it fails again the app's
  statements are the answer. The statements are shown under "Verified
  statements" in the provenance card.
- **Standard variables per construct.** `explorer/standards.py` fixes which
  variable answers "gender", "public vs private", "socio-economic status",
  "immigrant background", "bullying", "AI use" and so on, per cycle, with the
  reason; the planner sees it first and a plan that picked a look-alike is
  switched to the standard and told so in the provenance.
- **Plan-shape repairs and stratum rows (method v6).** Before a plan runs,
  the app repairs the shapes the planner gets wrong in ways that would
  compute the wrong statistic: a benchmark average or a multi-economy filter
  always runs per economy (never a pooled student mean across economies);
  named ad-hoc groups get one average row each; "economy A minus economy B"
  becomes two rows whose difference, and its change across cycles, the app
  states itself; a per-cycle `by` override applies (MALE recoded to
  ST004D01T). A named sampling stratum (a region, a school network) is a
  labelled row **next to** the economy's row — `IDN/DKI Jakarta`,
  `KAZ excl. Intellectual schools` — never the economy's row; "versus the
  rest of the country" is a gap. "Level 1 or below" is read as below Level 2.
  Questionnaire indices whose OECD-average is not 0 in a later cycle (trend
  scales such as BULLIED, BELONG, ANXMAT) get a cross-cycle change with a
  sampling SE and a note; the rest stay blank. Creative thinking (2022, via
  the `stu_crt_2022` joined view), global competence (2018) and financial
  literacy (2018/2022) are standard measures.
- **A closed plan grammar.** The planner no longer writes SQL. It fills a
  form (`explorer/grammar.py`): a statistic from a fixed list, measures as
  small objects (a score domain, a catalog variable, a proficiency-level
  share, a response-code share, a threshold share), economies as codes,
  filters as (variable, operator, values), contrasts as sets of codes of
  one variable, predictors as variables with optional dummy and reference
  codes, benchmarks as named groups. The app compiles the form to SQL, so
  an invented column, an invented economy code, a pooled mean across
  economies or a dummy that counts non-respondents as the reference group
  can no longer be planned; a form that does not compile gets one corrected
  retry and then an honest answer naming the field. `PISA_PLANNER=legacy`
  restores the old free-text planner for comparison. One model per role can
  be set (`PISA_ROUTER_MODEL`, `PISA_PLANNER_MODEL`, `PISA_SUMMARY_MODEL`,
  `PISA_TRANSLATE_MODEL`); `scripts/planner_compare.py` runs the live golden
  set per planner model and reports pass rate, latency and tokens.
- **More statistics.** Weighted standard deviation; the OECD's academically
  resilient share (bottom ESCS quarter, top performance quarter within the
  economy, per plausible value, or "at or above Level N"); the
  between-school share of variance (one-way ICC, %); correlations and
  regressions within ESCS quarters. A gap answer also states each side's own
  mean. "Is that good?" reruns the last question with the OECD average
  beside it; "which members entered the average?" is answered from the last
  result's provenance; "why does your figure differ from Table X?" recomputes
  the figure and lists the reasons a published table can differ.
- **Statements the model cannot make up.** The app also states, for the
  economies a question names, which other economies are not statistically
  different (and the rank range that implies), gap-in-gap differences, and
  the change in a difference across cycles (the link error cancels). The
  prose check rejects a draft that attributes a number to the opposite
  group of the one the statement gives it (a female mean called boys', a
  share who disagree called agreeing), that rounds a standard error to 0.0,
  that calls the result "partial", or that says the app "cannot compute" a
  difference it computed.
- **Reproducibility.** The router and planner run at temperature 0. Every
  provenance card carries `build <image tag>, method v<N>` (`explorer/version.py`);
  the method version changes only when a computed number can change. Every
  deterministic guard that fires is recorded on the event (`guards`), and
  the admin dashboard shows guard hit rates, rejected model drafts and the
  build that answered.

## Chat with the data

Put your Gemini key in a `.env` file at the repo root (gitignored):

```
GEMINI_API_KEY=your-key-here
```

Then:

```
python -m explorer.chat "How did reading scores change in Finland between 2018 and 2025?"
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
95%-confidence whiskers, cross-cycle questions as dumbbells (one dot per cycle,
2018 → 2022 → 2025), gaps as diverging bars around zero, and crosstabs as
heatmaps. Every chart has
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

Cross-cycle versions of all of these run automatically when the question
compares cycles: the template runs per cycle (any two or all three of 2018,
2022, 2025), the table shows every cycle side by side, and `change` is the
last cycle minus the first. A question that names no cycle means 2025.

Regions are fixed in code, not improvised by the model: "Latin America",
"Asia", "the EU", "Nordic countries", "Sub-Saharan Africa" and the other groups
in `explorer/regions.py` expand to the exact economies present in each cycle,
and the provenance lists them (and which members are absent from a cycle).
Questions that also ask *why* ("what factors were behind it") get the
computable part answered plus an explicit note that PISA, a repeated
cross-section, cannot establish causes — with a pointer to the association
templates.

Where a variable differs by cycle the plan carries `cycle_overrides` — a
per-cycle replacement of plan fields that is always stated in the provenance.
The standing example is gender: `ST004D01T` (1 = female, 2 = male) in 2018
and 2022, but in 2025 fourteen economies release only the derived `MALE` flag
(1 = male, 0 = female/other), so 2025 gender analyses use `MALE`.

## Deployment

The public-sharing setup (Cloud Run, institution gate, per-session and
global rate limits, self-contained Docker image) is in [DEPLOY.md](DEPLOY.md).
There is no password: visitors enter the name of their institution or
organization once (remembered by the browser), and it is recorded with every
question for the admin dashboard. Environment knobs: `PISA_ADMIN_CODE`,
`PISA_RATE_LIMIT` (default 20/h per session), `PISA_GLOBAL_RATE` (default
200/h total), `PORT`, `BIND_HOST`.

`convert.py` is idempotent (skips finished files; `--force` to redo,
`--only 2025` / `--only stu_qqq_2018` to filter). Every Parquet file is written
to a temp name and renamed only after its row count matches both the raw file's
header and the official figure — a crash cannot leave a plausible-looking
partial file (the defect that silently truncated the old CSV pipeline). The
2025 questionnaire-timing file declares UTF-8 but contains Latin-1 bytes;
`sources.py` carries the per-file encoding override.

## Database

`data/pisa.duckdb` exposes one view per instrument per cycle:

| View | Content |
|---|---|
| `stu_qqq_2018` / `_2022` / `_2025` | Student questionnaire (PV1–PV10 MATH/READ/SCIE, `W_FSTUWT`, `W_FSTURWT1–80`, ESCS). 2025 adds science subscales (SEPS/SEDE/SEID/SENV), the Learning-in-the-Digital-World PVs (CMPS/CPPK/CMOD/CPRO) and the ICT and parent questionnaire items |
| `sch_qqq_*`, `tch_qqq_*` | School and teacher questionnaires (all three cycles) |
| `stu_cog_*` | Cognitive item responses (all three cycles) |
| `stu_tim_*` | Questionnaire timing (all three cycles) |
| `stu_ttm_2018` / `stu_ttm_2025` | Cognitive item process data (time, actions) |
| `flt_qqq_*`, `flt_cog_*`, `flt_tim_*`, `flt_ttm_2018` | Financial literacy (2018, 2022) |
| `crt_cog_2022` | Creative thinking (2022 only) |
| `ldw_cog_2025` | Learning in the Digital World item responses and process data (2025 only; the LDW *scores* are the CMPS/CPPK/CMOD/CPRO PVs in `stu_qqq_2025`) |
| `escs_trend` | OECD comparable-ESCS trend table |

The cycles deliberately keep separate tables: questionnaires overlap but are
not identical across cycles (variables added/dropped/renamed), so cross-cycle
comparability is a per-variable decision made at query time. The 2025 files
are mapped onto the instrument names by content (`CY09_MS_STU_TT_PUF` is the
questionnaire-timing file → `stu_tim_2025`; `CY09_MS_COG_PROCESS` is the
cognitive process file → `stu_ttm_2025`; `CY09_MS_LDW` → `ldw_cog_2025`).
Not loaded: the LDW self-regulated-learning file (not downloaded) and the
Foreign Language Assessment files (OECD release expected in 2027).

## Catalog and statistics layer

`catalog_variables` (29,337 rows, one per variable per table, ~95% with
value labels) and `catalog_comparability` (cross-cycle availability of
20,670 variables in the 9 instruments present in two or more cycles:
`in_2018` / `in_2022` / `in_2025`, per-cycle labels, `n_cycles`; 6,280
variables appear in 2+ cycles and 1,335 in all three) live
both as DuckDB tables and as Parquet under `data/catalog/`. `explorer.catalog.search()` is the retrieval
function the AI layer calls: it ranks *variables* (not rows) and returns each
selected variable's row from every cycle and instrument it exists in, so a
variable card reads `REPEAT (stu_qqq_2018, stu_qqq_2022, stu_qqq_2025)` and
the planner can see cross-cycle availability directly. Only the ~40
top-ranked variables ever reach a prompt. 2018 value labels are recovered by combining each instrument's
`.FORMAT.SAS` (variable -> format name) with its `.SAS7BCAT` catalog
(format -> labels, latin1); 2022 and 2025 labels come from the SPSS files.

## Methodology rules (implemented in `explorer/estimator.py`)

- Population statistics always use the final student weight `W_FSTUWT`.
- Point estimates for achievement average over all 10 plausible values.
- Standard errors: Fay's BRR (k = 0.5) over `W_FSTURWT1`–`W_FSTURWT80`,
  combined with between-PV imputation variance (Rubin's rules).
- Group gaps are differenced replicate-wise (correct covariance handling);
  cross-cycle trends add variances (independent samples) plus the OECD link
  error for mean scores, a share-specific link error for proficiency shares,
  and no link error for group differences, where it cancels (Annex A7).
- **Validated:** country means AND standard errors reproduce the published
  PISA 2018/2022 figures (e.g. USA 2022 math 465 SE 4.0, FIN 484 SE 1.9,
  KOR 527 SE 3.9). For 2025, `pipeline/check_2025_published.py` reproduces
  the Technical Report's Annex 14 sample sizes, weighted populations and
  mean-score standard errors for all 90 economies in science, reading and
  mathematics.
- 2025 gotchas the planner knows: Uzbekistan has science PVs only; gender
  comes from `MALE` in 2025 (see above); the LDW PVs exist only for
  computer-based economies.
- Codes stay in the data; labels come from the catalog and are applied at
  display time.
