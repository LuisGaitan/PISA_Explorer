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

```powershell
gcloud run deploy pisa-explorer `
  --source . `
  --region us-central1 `
  --memory 2Gi --cpu 2 `
  --concurrency 4 `
  --max-instances 2 `
  --min-instances 0 `
  --allow-unauthenticated `
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest `
  --set-env-vars PISA_ACCESS_CODE=<pick-a-code>,PISA_RATE_LIMIT=20,PISA_GLOBAL_RATE=200
```

Cloud Build builds the Dockerfile (the 3.7 GB context upload takes a while
the first time), and the command prints the public URL when done. Share the
URL + access code with testers; the UI asks for the code once and remembers it.

To update after code changes: rerun the same `gcloud run deploy` command.

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
