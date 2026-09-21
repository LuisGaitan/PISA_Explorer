# Deploying PISA Explorer to Cloud Run

The image is fully self-contained (app + ~5.0 GB Parquet for the three cycles
+ DuckDB rebuilt at build time), so there is nothing to provision besides
Cloud Run itself — no BigQuery, no database service, no bucket.

## One-time setup

```powershell
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>       # a NEW project is cleanest
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Store the Gemini key as a secret (never as a plain env var in scripts).
**Do not pipe it from PowerShell** (`echo key | gcloud ...`): PowerShell 5.1
prepends a UTF-8 BOM and appends CRLF, and the corrupted key then breaks the
HTTP header at runtime (`'latin-1' codec can't encode character '﻿'`).
Write it to a BOM-free file instead:

```powershell
gcloud services enable secretmanager.googleapis.com
$tmp = Join-Path $env:TEMP "gk.txt"
[IO.File]::WriteAllText($tmp, "<YOUR_GEMINI_KEY>", [Text.UTF8Encoding]::new($false))
gcloud secrets create gemini-api-key --data-file=$tmp      # or: versions add
Remove-Item $tmp -Force
# verify the bytes (no EF BB BF prefix, no 0D 0A suffix):
gcloud secrets versions access latest --secret=gemini-api-key --out-file=$tmp
[IO.File]::ReadAllBytes($tmp)[0..2] | ForEach-Object { $_.ToString('X2') }
Remove-Item $tmp -Force
```

(The app also strips a BOM/whitespace from the key defensively since v2 of
the code, but a clean secret is still the right fix.)

## Deploy (from the repo root)

`gcloud run deploy --source .` does NOT work for this repo: the multi-GB
source context exceeds gcloud's single-request upload window (it dies with
`ReadTimeout` after uploading for ~20 minutes). Use the three-step route
instead — archive, resumable upload, build from the bucket — which is what
produced the live deployment:

```powershell
# 0. one-time: a staging bucket and access for Cloud Run to the secret
gcloud storage buckets create gs://<PROJECT_ID>-build --location=us-central1 --uniform-bucket-level-access
$pn = gcloud projects describe <PROJECT_ID> --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding gemini-api-key `
  --member="serviceAccount:$pn-compute@developer.gserviceaccount.com" `
  --role="roles/secretmanager.secretAccessor"

# 1. archive the source (same exclusions as .gcloudignore; ~5 GB with 2025)
tar -cf $env:TEMP\pisa-source.tgz -z --options gzip:compression-level=1 `
  --exclude=./.git --exclude=./.env --exclude=__pycache__ --exclude=*.pyc `
  --exclude=./data/pisa.duckdb --exclude=./data/pisa.duckdb.wal `
  --exclude=./data/metadata --exclude=*.log -C . .

# 2. resumable upload
gcloud storage cp $env:TEMP\pisa-source.tgz gs://<PROJECT_ID>-build/pisa-source.tgz

# 3. build the image from the bucket, then deploy it
gcloud builds submit gs://<PROJECT_ID>-build/pisa-source.tgz `
  --tag us-central1-docker.pkg.dev/<PROJECT_ID>/cloud-run-source-deploy/pisa-explorer:v1 `
  --region us-central1 --machine-type e2-highcpu-8 --timeout 2400s

gcloud run deploy pisa-explorer `
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/cloud-run-source-deploy/pisa-explorer:v1 `
  --region us-central1 `
  --memory 2Gi --cpu 2 `
  --concurrency 4 `
  --max-instances 2 `
  --min-instances 0 `
  --allow-unauthenticated `
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest `
  --set-env-vars PISA_ADMIN_CODE=<secret>,PISA_RATE_LIMIT=20,PISA_GLOBAL_RATE=200
```

(The Artifact Registry repo `cloud-run-source-deploy` is created automatically
by a first `gcloud run deploy --source` attempt; otherwise create it with
`gcloud artifacts repositories create cloud-run-source-deploy --repository-format=docker --location=us-central1`.)

The deploy command prints the public URL. Share the URL with testers; the UI
asks each visitor for their institution or organization once and remembers it
on that device (there is no password).

## Code-only redeploy (the normal case — minutes, no data upload)

Before building, run the regression gates (see CONTRIBUTING.md):

```powershell
pytest tests/                              # offline + golden plans (exact numbers)
python scripts/golden_live.py              # golden questions through the model
python scripts/replay_events.py --days 7   # yesterday's real questions, diffed
```

The `_TAG` you build with is stamped into every answer's provenance
(`build v21, method v3` — `explorer/version.py`), so keep tags unique and
increasing: a cited number can then be traced to the code that produced it.

`Dockerfile.code` layers the current code onto the existing data image, so a
code change never re-uploads the ~5 GB of Parquet:

```powershell
gcloud builds submit --config cloudbuild.code.yaml --ignore-file .gcloudignore.code `
  --substitutions _TAG=v3 --region us-central1 .
gcloud run deploy pisa-explorer `
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/cloud-run-source-deploy/pisa-explorer:v3 `
  --region us-central1
```

(`gcloud run deploy --image` keeps the service's existing env vars, secrets
and limits.) Repeat the full archive route only when the data or the pipeline
changes, then point `_DATA_IMAGE` in `cloudbuild.code.yaml` at the new data tag.

The code-only image also carries `data/parquet/pisa2018/vnm_pv.parquet`
(Viet Nam's separately released 2018 plausible values, built by
`python pipeline/load_vnm_2018.py`; `build_db.py` in the image joins them
into the 2018 view; it also builds the joined views `stu_sch_<cycle>` and
`stu_crt_2022` — creative-thinking plausible values with student weights —
from the parquet files already in the data image) and `data/catalog/coverage.parquet` (which
economies collected each questionnaire variable, per cycle — built by
`python pipeline/build_coverage.py` in a few seconds from the local DuckDB).
`.gcloudignore.code` re-includes exactly that file from the otherwise
excluded `data/`; if it is missing locally the Docker `COPY` fails the build
rather than shipping an image without coverage checks.

## Data updates (e.g. adding a PISA cycle)

Adding PISA 2025 changed the data, so the live `v1` data image (2018 + 2022
only) cannot be reused by a code-only build: the 2025 views would be missing
and every 2025 question would fail. After a data change:

1. rebuild locally: `python pipeline/convert.py`, `build_db.py`,
   `build_catalog.py`, `build_coverage.py`, `validate.py` (and
   `check_2025_published.py`);
2. run the **full archive route** above with a new tag (e.g. `v3`), then
   `gcloud run deploy --image ...:v3`;
3. set the tag in `_DATA_IMAGE` in `cloudbuild.code.yaml` (the project ID
   is filled in from the build's own `$PROJECT_ID`) so later code-only
   builds layer on it.

The three-cycle image is about 40% larger than the 2018+2022 one; the
`e2-highcpu-8` build machine and the 2400 s timeout still suffice. Memory
per instance is governed by the heaviest query shape, not by the data size
(see "Production settings" below).

## Institution gate, admin, analytics

```powershell
gcloud run services update pisa-explorer --region us-central1 --update-env-vars `
  "PISA_ADMIN_CODE=<secret>"
```

- **No password.** The gate asks for the visitor's institution or
  organization (free text, 2–80 characters). The browser stores it and sends
  it as the `X-Institution` header on every request; the server normalizes
  it (single spaces, length cap) and records it on each event, so the admin
  dashboard's "Questions by institution" chart groups by what people typed.
  Spelling variants of the same institution therefore show as separate rows.
  The former `PISA_ACCESS_CODES` / `PISA_ACCESS_CODE` variables are ignored.
- `PISA_ADMIN_CODE` unlocks the usage dashboard at `/admin` (enter the code
  once, or open `/admin?code=<secret>` — it stores the code and scrubs the
  URL). The dashboard reads `/api/admin/events`; export the raw events as CSV
  from its header for deeper mining.
- Events (one Firestore document per question, collection `events`) record:
  institution, session, question, route (data / clarify / conversational /
  error), template, instrument, cycles, countries, variables, rows, raw-SQL
  and substitution flags, regions, the answer text (first 4,000 characters),
  latency and LLM-call counts, and — reported by the
  browser — which chart form rendered, device width / mobile, thumbs up/down
  with comment, and CSV exports. Firestore is enabled per project:
  `gcloud firestore databases create --location=us-central1 --type=firestore-native`
  plus `roles/datastore.user` for the Cloud Run service account.

## Production settings (300+ students plus public traffic)

The feedback-phase defaults (2 instances, 20 questions/hour per session,
200/hour global) are far too tight for a real audience. The production
configuration, applied 2026-09-13:

```powershell
gcloud run services update pisa-explorer --region us-central1 `
  --memory 4Gi --cpu 2 --concurrency 2 --max-instances 20 --min-instances 1 --session-affinity `
  --update-env-vars PISA_RATE_LIMIT=60,PISA_GLOBAL_RATE=5000
```

Why these numbers:

- **4Gi memory, one analysis at a time per instance** (an `_agent_lock` in
  `app.py`): the DuckDB connection is not thread-safe, and measured peaks are
  1.4 GB for an all-economies mean and 2.4 GB for a three-cycle all-economies
  gender gap (the replicate-weight frames plus their Arrow copy). A 2Gi
  instance was killed for exceeding its limit during the first load test
  (two 503s). `explorer/db.py` also caps DuckDB's own buffer pool at 1 GB
  (`PISA_DUCKDB_MEMORY`) — its default is ~80% of RAM, which would have
  competed with the frames.
  Throughput therefore comes from *instances*: each handles ~6 questions a
  minute, so 20 instances give ~7,000 questions/hour. `--concurrency 2` lets
  an instance accept one running question plus one waiting (and static files
  and analytics beacons in between), which makes the autoscaler spread load
  early instead of queueing four deep.
- **`--min-instances 1`** keeps one instance warm so the first visitor of the
  day does not wait through a cold start (~40 USD/month; set it back to 0 to
  scale to zero).
- **`--session-affinity`** routes a browser back to the instance holding its
  conversation history, so follow-up questions keep working across many
  instances (best effort; a scale-down still resets a conversation).
- **Rate limits** are now circuit breakers, not usage caps: 60/hour per
  session never interrupts a real person; 5,000/hour global (per instance,
  since counters live in instance memory) bounds runaway spend. At
  gemini-2.5-flash prices a question costs well under one cent, so even a
  saturated hour is a few dollars; the project budget alert is 200 USD/month
  (50/90/100%).
- **Gemini key**: paid tier (tier 2) — the free tier's daily request cap
  would allow under 100 questions a day. `llm.py` retries 429/5xx and
  network errors twice with short backoff.

Not yet in place (add if abuse appears): a per-IP limit via Cloud Armor
(campus NAT makes naive per-IP limits punish whole institutions), an uptime
check with alerting on 5xx, and a custom domain.

## Cost & protection model

- **Scale to zero**: `--min-instances 0` means you pay nothing while idle;
  a cold start (loading the image) takes a few seconds.
- **Bounded spend**: `--max-instances 2` caps compute; `PISA_GLOBAL_RATE`
  caps total Gemini calls at ~3 x 200 requests/hour across all users;
  the per-session limit stops any one tester from hogging it. Also set a
  budget alert in Google Cloud Billing and a quota cap on the Gemini key in
  Google AI Studio.
- **Open endpoint**: with no password, anyone who finds the URL can ask
  questions; the per-session and global hourly rate limits are what bound
  the Gemini spend, so keep them set on a public deployment.
- **Query safety**: the DuckDB connection is read-only and raw SQL is
  single-statement SELECT-only, so the worst a malicious query can do is
  read public OECD data it could download anyway.

## Test the image locally first (optional, needs Docker Desktop)

```powershell
docker build -t pisa-explorer .
docker run --rm -p 8080:8080 --env-file .env pisa-explorer
# then open http://localhost:8080
```

## Notes

- Sessions and rate counters live in instance memory: a scale-down or new
  revision resets conversation history (fine for a feedback deployment).
  `--concurrency 4` is safe because the agent serializes queries internally.
- The app binds 0.0.0.0 only because `BIND_HOST` is set in the Dockerfile;
  running locally without Docker still defaults to 127.0.0.1.
- Raw OECD data never leaves your machine except as the derived Parquet
  tables baked into the image — which are the public-use files anyway.
