# PISA Explorer — self-contained image: app + Parquet data + DuckDB views.
# The database is rebuilt inside the image so its views point at container
# paths (never at Windows paths from the dev machine).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ pipeline/
COPY explorer/ explorer/

# Data: Parquet tables, the variable catalog, and the ESCS trend file.
# (data/pisa.duckdb is NOT copied — it is rebuilt right here.)
COPY data/parquet/ data/parquet/
COPY data/catalog/ data/catalog/
COPY data/escs_trend.csv data/escs_trend.csv
RUN python pipeline/build_db.py

ENV BIND_HOST=0.0.0.0
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "explorer.app"]
