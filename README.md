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

# Choose distance metric (l2 | cosine | dot)
python -m src.main search --table demo --modality text --query "invoice" --metric cosine
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

## Index Creation (IVF-PQ)

Build or rebuild a search index after ingestion to accelerate queries while keeping good recall. You can let the tool choose parameters automatically, or provide explicit values.

Examples

```bash
# Adaptive (heuristic) index based on rows and embedding dim
python -m src.main create-index --table multimodal

# Explicit overrides (advanced)
python -m src.main create-index --table multimodal \
  --partitions 2048 \
  --m 32 \
  --metric l2 \
  --num-bits 8 \
  --opq \
  --max-train-rows 500000
```

Heuristic and Env Overrides

- Partitions (nlist): approx `4 * sqrt(N)` clamped to `[512, 8192]` by default. Override with:
  - CLI: `--partitions`
  - Env: `INDEX_NUM_PARTITIONS`, `INDEX_MIN_PARTITIONS`, `INDEX_MAX_PARTITIONS`
- PQ sub-vectors `m`: largest divisor of dim from `{64,48,32,24,16,12,8,4,2,1}`. Override with:
  - CLI: `--m`
  - Env: `INDEX_M`
- Index metric: default `l2`. Override with:
  - CLI: `--metric {l2,cosine}`
  - Env: `LANCEDB_INDEX_METRIC`
- Training knobs (if supported by your lancedb version):
  - CLI: `--num-bits`, `--opq`, `--max-train-rows`
  - Env: `INDEX_NUM_BITS`, `INDEX_USE_OPQ`, `INDEX_MAX_TRAIN_ROWS`
- Minimum rows to build an index: default `5000` (to avoid poor codebooks on tiny datasets). Override with:
  - CLI: `--min-rows`
  - Env: `INDEX_MIN_ROWS`

Notes

- You can re-run index creation safely after new data is ingested. The command replaces the previous index.
- Query-time recall/latency is controlled by `LANCEDB_NPROBES` and `LANCEDB_REFINE_FACTOR` (see next section).

---

## Ingestion Performance

For large datasets (e.g., millions of files), tune these env vars in `.env` to improve throughput:

- INGEST_WORKERS: S3 download concurrency. Try 32–128 depending on your MinIO and network.
- INGEST_EMBED_BATCH: Batch size for text/image/pdf embeddings. Increase with GPU/CPU memory (e.g., 128–512).
- LIDAR_EMBED_BATCH: Explicit LiDAR batch size. If unset (0), it uses `INGEST_EMBED_BATCH // LIDAR_BATCH_DIVISOR`.
- LIDAR_BATCH_DIVISOR: Lower value → larger default LiDAR batch (e.g., 4).
- INGEST_WRITE_CHUNK: Rows per DB write chunk (e.g., 10000–50000) to reduce write overhead.
- INGEST_PDF_MAX_PAGES: Fewer pages per PDF speeds ingestion.
- INGEST_PROGRESS_EVERY: Print progress every N processed items (0 = per-batch prints).
- MINIO_SUPPRESS_TLS_WARN: Set to 1 to suppress HTTPS warnings when `MINIO_VERIFY=false`.

Example high-throughput settings:

```env
INGEST_WORKERS=64
INGEST_EMBED_BATCH=256
LIDAR_EMBED_BATCH=64
INGEST_WRITE_CHUNK=20000
INGEST_PROGRESS_EVERY=1000
```

---

## Distance Metric & Recall Tuning

- Metric selection:
  - CLI: add `--metric {l2,cosine,dot}` (default from `LANCEDB_METRIC`, falls back to `l2`).
  - Web UI: use the Metric dropdown (Cosine/Dot recommended). The results table shows Distance (L2) or Similarity (Cosine/Dot).
  - Note: embeddings are L2-normalized, so `dot` is equivalent to `cosine` in practice.

- ANN recall vs latency:
  - `LANCEDB_NPROBES` (default 32): more probes → higher recall, slower.
  - `LANCEDB_REFINE_FACTOR` (default 50): re-rank more candidates with exact distances.
  - Set these env vars in `.env` to tune web UI and API behavior; CLI also honors them.

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
