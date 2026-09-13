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
  --set-env-vars PISA_ACCESS_CODE=<pick-a-code>,PISA_RATE_LIMIT=20,PISA_GLOBAL_RATE=200
```

(The Artifact Registry repo `cloud-run-source-deploy` is created automatically
by a first `gcloud run deploy --source` attempt; otherwise create it with
`gcloud artifacts repositories create cloud-run-source-deploy --repository-format=docker --location=us-central1`.)

The deploy command prints the public URL. Share the URL + access code with
testers; the UI asks for the code once and remembers it.

## Code-only redeploy (the normal case — minutes, no data upload)

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

## Data updates (e.g. adding a PISA cycle)

Adding PISA 2025 changed the data, so the live `v1` data image (2018 + 2022
only) cannot be reused by a code-only build: the 2025 views would be missing
and every 2025 question would fail. After a data change:

1. rebuild locally: `python pipeline/convert.py`, `build_db.py`,
   `build_catalog.py`, `validate.py` (and `check_2025_published.py`);
2. run the **full archive route** above with a new tag (e.g. `v3`), then
   `gcloud run deploy --image ...:v3`;
3. set `_DATA_IMAGE` in `cloudbuild.code.yaml` (and the `DATA_IMAGE` default
   in `Dockerfile.code`) to that tag so later code-only builds layer on it.

The three-cycle image is about 40% larger than the 2018+2022 one; the
`e2-highcpu-8` build machine and the 2400 s timeout still suffice, and the
2Gi Cloud Run instance is unchanged (queries stream Parquet through DuckDB;
only the projected columns of one cycle table are held in memory at a time).

## Access codes, admin, analytics

```powershell
gcloud run services update pisa-explorer --region us-central1 --update-env-vars `
  "PISA_ACCESS_CODES=penn-2026:University of Pennsylvania,ucla-2026:UCLA,pisa2026:General,PISA_ADMIN_CODE=<secret>"
```

- One code per institution: the code *is* the attribution (no typos, no
  extra form field) and any one can be revoked by removing it. The legacy
  single `PISA_ACCESS_CODE` still works (recorded as "General").
- `PISA_ADMIN_CODE` unlocks the usage dashboard at `/admin` (enter the code
  once, or open `/admin?code=<secret>` — it stores the code and scrubs the
  URL). The dashboard reads `/api/admin/events`; export the raw events as CSV
  from its header for deeper mining.
- Events (one Firestore document per question, collection `events`) record:
  institution, session, question, route (data / clarify / conversational /
  error), template, instrument, cycles, countries, variables, rows, raw-SQL
  and substitution flags, latency and LLM-call counts, and — reported by the
  browser — which chart form rendered, device width / mobile, thumbs up/down
  with comment, and CSV exports. Firestore is enabled per project:
  `gcloud firestore databases create --location=us-central1 --type=firestore-native`
  plus `roles/datastore.user` for the Cloud Run service account.

## Cost & protection model

- **Scale to zero**: `--min-instances 0` means you pay nothing while idle;
  a cold start (loading the image) takes a few seconds.
- **Bounded spend**: `--max-instances 2` caps compute; `PISA_GLOBAL_RATE`
  caps total Gemini calls at ~3 x 200 requests/hour across all users;
  the per-session limit stops any one tester from hogging it. Also set a
  budget alert in Google Cloud Billing and a quota cap on the Gemini key in
  Google AI Studio.
- **Access code**: without `PISA_ACCESS_CODE` the endpoint is open to anyone
  who finds the URL — always set it on a public deployment.
- **Query safety**: the DuckDB connection is read-only and raw SQL is
  single-statement SELECT-only, so the worst a malicious query can do is
  read public OECD data it could download anyway.

## Test the image locally first (optional, needs Docker Desktop)

```powershell
docker build -t pisa-explorer .
docker run --rm -p 8080:8080 --env-file .env -e PISA_ACCESS_CODE=test123 pisa-explorer
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
