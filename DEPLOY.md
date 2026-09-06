# Deploying PISA Explorer to Cloud Run

The image is fully self-contained (app + 3.7 GB Parquet + DuckDB rebuilt at
build time), so there is nothing to provision besides Cloud Run itself — no
BigQuery, no database service, no bucket.

## One-time setup

```powershell
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>       # a NEW project is cleanest
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

Store the Gemini key as a secret (never as a plain env var in scripts):

```powershell
gcloud services enable secretmanager.googleapis.com
echo <YOUR_GEMINI_KEY> | gcloud secrets create gemini-api-key --data-file=-
```

## Deploy (from the repo root)

`gcloud run deploy --source .` does NOT work for this repo: the ~3.4 GB
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

# 1. archive the source (same exclusions as .gcloudignore; ~3.4 GB)
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

To update after code changes: bump the image tag (`:v2`, …) and repeat
steps 1–3. Code-only changes still re-upload the data (it is baked into the
image); that is the price of a self-contained, zero-dependency service.
To change the access code or limits without a rebuild:
`gcloud run services update pisa-explorer --region us-central1 --update-env-vars PISA_ACCESS_CODE=newcode`.

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
