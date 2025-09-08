# emBed
Indexing and Retrieval


A small, practical scaffold to build image/PDF similarity search with **LanceDB** on **MinIO/S3**, plus a friendly CLI and a simple web search box.

---

## Quick Start

**1) Install**
```bash
pip install -r requirements.txt
````

**2) Configure**

```bash
# copy the sample and fill in your values
cp .env.sample .env
# edit .env with your endpoint, keys, region, and LANCEDB_URI
```

**3) Sanity check**

```bash
python -m src.main config
python -m src.main health
```

**4) Ingest from your bucket**

```bash
# optional: see what's there
python -m src.main ls-s3 --bucket YOUR_BUCKET --limit 40

# build/update the table (default name: demo)
python -m src.main ingest-s3 --bucket YOUR_BUCKET --table demo
```

**5) Search (CLI)**

```bash
# PDFs only
python -m src.main search --table demo --modality text --query "invoice" --where "modality = 'pdf'"
```

**6) Web UI (search bar)**

```bash
uvicorn src.webapp:app --reload --port 8000
# open http://127.0.0.1:8000
```

---

## Handy Commands

```bash
# list objects under a prefix
python -m src.main ls-s3 --bucket YOUR_BUCKET --prefix docs/ --limit 40

# dry-run ingest (no writes)
python -m src.main ingest-s3 --bucket YOUR_BUCKET --table demo --dry-run

# ingest only some types / limit count / append
python -m src.main ingest-s3 --bucket YOUR_BUCKET --table demo \
  --include-ext .pdf .jpg .png --max-files 100 --mode append

# peek and stats
python -m src.main peek --table demo --n 5
python -m src.main stats --table demo
```

---

## Notes

* `LANCEDB_URI` can be local (`data/lancedb`) or S3 (`s3://bucket/prefix`).
* For S3 LanceDB, the key needs write access to that prefix (Put/Get/Delete/AbortMultipartUpload).
* If your endpoint is `http://`, set `LANCEDB_ALLOW_HTTP=1`.

---

## Files

* `src/config.py` — loads `.env`, bridges MINIO↔AWS vars, prepares LanceDB options.
* `src/storage.py` — thin MinIO/S3 helper (list/get/upload) via `boto3`.
* `src/lancedb_manager.py` — connects to LanceDB and manages tables.
* `src/embedding.py` — deterministic dummy embedders (text/image/lidar) for testing.
* `src/multimodal_dataset.py` — tiny synthetic dataset generator.
* `src/ingest.py` — ingests PDFs/images from S3, extracts PDF text, writes LanceDB (filters/dry-run/append).
* `src/search.py` — similarity search with optional row filtering.
* `src/cli_utils.py` — pretty CLI output (`rich`).
* `src/main.py` — CLI with subcommands: `config`, `health`, `ls-s3`, `ingest-s3`, `search`, `peek`, `stats`, `reset`.
* `src/webapp.py` — minimal FastAPI search page (set table via `WEBAPP_TABLE`).

---

## Troubleshooting

* **403 on ingest to S3** → grant write to your `LANCEDB_URI` prefix or use a local path.
* **No web results** → confirm table name (banner shows it) and that rows > 0.
* **Noisy PDF logs** → expected on messy PDFs; unreadable files are skipped.
