# emBed
Indexing and Retrieval

A small, practical scaffold to build image/PDF/LiDAR similarity search with **LanceDB** on **MinIO/S3**, plus a friendly CLI and a simple web search box.

---

## Quick Start

1. **Install**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure**
   ```bash
   # copy the sample and fill in your values
   cp .env.sample .env
   # edit .env with your endpoint, keys, region, and LANCEDB_URI
   ```
3. **Sanity check**
   ```bash
   python -m src.main config
   python -m src.main health
   ```
4. **Ingest from your bucket**
   ```bash
   # optional: see what's there
   python -m src.main ls-s3 --bucket YOUR_BUCKET --limit 40

   # build/update the table (default name: demo)
   python -m src.main ingest-s3 --bucket YOUR_BUCKET --table demo
   ```
5. **Search via CLI**
   ```bash
   # PDFs only
   python -m src.main search --table demo --modality text --query "invoice" --where "modality = 'pdf'"

   # choose distance metric (l2 | cosine | dot)
   python -m src.main search --table demo --modality text --query "invoice" --metric cosine
   ```
6. **Launch the web UI**
   ```bash
   uvicorn src.webapp:app --reload --port 8000
   # open http://127.0.0.1:8000
   ```

---

## Architecture Overview

- **Ingestion** – `src/ingest/pipeline.py:ingest_s3_objects` streams objects from MinIO/S3 via `src/storage/minio_client.py:MinIOClient`, batches them by modality, and hands them to the embedder singletons exposed in `src/embedding_runtime.py`.
- **Embedding backends** – `ImageEmbedder`, `TextEmbedder`, and `LidarBEVEmbedder` produce L2-normalised vectors so LanceDB can compare modalities with cosine or L2 distance (`src/embedding/image.py`, `src/embedding/text.py`, `src/embedding/lidar.py`).
- **Persistence** – `src/vectordb/lance.py:LanceDBManager` manages table creation, S3 mirroring, and IVF-PQ index policy before data lands in LanceDB.
- **Query & ranking** – `src/search/service.py:SearchService` vectorises user payloads, applies LanceDB recalls knobs, and post-filters the results shared by the CLI and FastAPI UI.

```
[Object storage / local files]
          |
          v
Download & batch (src/ingest/io.py:download_many)
          |
          v
Embed & enrich (src/ingest/pipeline.py:ingest_s3_objects)
          |
          v
LanceDB table (src/vectordb/lance.py:LanceDBManager)
          |\
          | \-- Vector search (src/search/service.py)
          |        |
          |        v
          |   Ranked results (shared Pandas output)
          |
          +--> CLI & Web UI (src/main.py, src/webapp.py)
```

---

## Pipeline Details

### Embedding Backends

- **Image (`ImageEmbedder`)** – Loads the OpenCLIP encoder defined by `IMAGE_MODEL` (default `ViT-B-32#openai`) on the device chosen by `DEVICE` (`src/embedding/image.py:21`). Raw bytes or PIL images are resized with the model-specific preprocessor, encoded to float32 vectors, and L2-normalised so cosine, dot, and L2 behave consistently downstream.
- **Text (`TextEmbedder`)** – Switches between sentence-transformers (`all-MiniLM-L6-v2`) and OpenCLIP text towers depending on `TEXT_MODEL` (`src/embedding/text.py:19`). Text is lower-cased/tokenised by the backend, batched on GPU/CPU, and normalised before persisting. Using the `clip:` prefix guarantees the vectors land in the same space as the image encoder, enabling text↔image or text↔LiDAR retrieval without extra adapters.
- **LiDAR (`LidarBEVEmbedder`)** – Accepts numpy point clouds from `.pcd` or `.bin` files, projects them into a configurable bird's-eye-view raster, and forwards the raster through the image encoder (`src/embedding/lidar.py:18`). Intensity columns are preserved when present; otherwise height is normalised to pseudo-intensity.
- **Runtime selection** – `src/embedding/runtime.py` keeps singleton instances and honours `EMBEDDING_BACKEND=dummy` for deterministic hash vectors during tests or when GPUs are unavailable.

### Ingestion Flow

1. **Discovery & staging** – CLI command `ingest-s3` wires up MinIO/LanceDB configuration (`src/config`) and streams matching keys with concurrency chosen via `INGEST_WORKERS` (`src/ingest/pipeline.py:285`).
2. **Download & decoding** – `download_many` pulls objects locally and returns a blob map. Modality-specific helpers convert bytes into working formats: `decode_text_bytes` for plain text, `extract_pdf_text` for PDFs, and `load_lidar_bytes` for point clouds (`src/ingest/text_pdf.py`, `src/ingest/lidar.py:63`).
3. **Batch embedding** – `_process_one_batch` groups blobs by extension, respects `INGEST_EMBED_BATCH` / `LIDAR_EMBED_BATCH`, and calls the relevant embedder. Errors are isolated per file, and small batches fallback to per-item embedding (`src/ingest/pipeline.py:116`).
4. **Row assembly** – Each embedded item becomes a LanceDB row with `id`, `modality`, `embedding`, `path`, and modality payloads (`text`, `image`, `lidar`) populated when available. PDFs carry extracted text (or `Empty PDF document` markers) for later filtering (`src/ingest/pipeline.py:197`).
5. **Persistence & indexing** – `UpsertWriter` ensures tables exist (respecting overwrite/append mode), then streams batches into LanceDB via `LanceDBManager`, which can auto-build IVF-PQ indexes when row counts cross the configured threshold (`src/ingest/writer.py`, `src/vectordb/lance.py:90`).

### Query & Cross-Modal Search

**Vector preparation**

- CLI arguments, FastAPI payloads, or UI uploads are normalised into `SearchInputs` objects containing modality hints and raw payloads (`src/search/service.py:23`).
- `vectorize_modalities` embeds each requested modality on-demand using the same singleton embedders as ingestion, ensuring embedding parity (`src/search/service.py:179`).

**Single-modality searches**

- *Text* – `SearchInputs.text_query` is validated to avoid empty searches, lower-cased/tokenised by the active NLP backend, and embedded as a `(1, dim)` vector. By default the model is the MiniLM-based sentence-transformer `all-MiniLM-L6-v2` (`src/embedding/text.py:28`); when `TEXT_MODEL` carries a `clip:` prefix we load the OpenCLIP text tower to stay aligned with the vision encoder. The flattened output feeds LanceDB directly, guaranteeing identical statistics to ingested rows (`src/search/service.py:149`).
- *Image* – Clients may upload a file (`image_blob`) or reference server-side storage (`image_path`). The helper opens the image, converts it to RGB, applies the OpenCLIP preprocessing transform (resize, center crop, normalise), and runs it through the ViT-B/32 encoder specified by `IMAGE_MODEL` (`src/embedding/image.py:24`). This mirrors ingestion so CLIP’s vision transformer features line up with text and LiDAR representations (`src/search/service.py:155`).
- *LiDAR* – `.pcd` (text/binary) and `.bin` blobs are parsed with `load_lidar_points`, producing an `N x {3,4,5}` float matrix. `LidarBEVEmbedder` projects the cloud into a bird’s-eye-view raster and feeds it through the same OpenCLIP vision tower, effectively treating LiDAR as an image modality (`src/embedding/lidar.py:18`). Deployments that do not expose LiDAR set `allow_lidar=False`, short-circuiting the request with a user-facing error (`src/search/service.py:163`).

**Cross-modal & joint queries**

- Multi-modal requests populate `SearchInputs.modalities` with ordered modality labels. `vectorize_modalities` runs each branch separately, stacks the resulting arrays, computes their mean, and re-normalises to unit length. Because images, text, and LiDAR all share CLIP-derived embeddings (or MiniLM for text-only setups), the averaged vector stays in a coherent joint space while remaining compatible with LanceDB’s metric selection (`src/search/service.py:179`).
- Joint queries are executed the same way the web UI builds payloads today: form inputs drive `normalize_modalities` to capture the selected toggles, uploads are read into `image_blob` / `lidar_blob`, and a `SearchInputs` instance aggregates everything (`src/web/app.py:1012`). When the request reaches `vectorize_modalities`, each modality runs through its dedicated embedder and the vectors are averaged, yielding a single fused query (`src/search/service.py:179`).
- Per-modality preprocessing mirrors ingestion: PDF pages are OCR’d into text strings (`src/ingest/text_pdf.py:80`), images are centre-cropped and normalised by OpenCLIP, and LiDAR point clouds become BEV rasters before embedding. Because those transformations happen both during ingest and at query time, a multimodal example such as “car near lamppost” plus an uploaded `.pcd` file retrieves the same joint space features used for storage.
- Because every embedder outputs unit vectors (CLIP or MiniLM features normalised via `l2_normalize`), cosine and dot metrics yield identical ordering, and switching to L2 simply converts similarity to distance. This means cross-modal search quality hinges on the chosen backbone models and averaging strategy, not on metric quirks.

**Filtering, ranking, and thresholds**

- `SearchService.run` applies LanceDB metric selection, `nprobes`, and `refine_factor` tunables before executing the ANN search. Optional `where` clauses support SQL-like row filters for modality or metadata (`src/search/service.py:47`).
- Post-search, `apply_metric_threshold` can drop rows below a similarity cut-off (cosine/dot) or above a distance limit (L2), which is surfaced in the web UI and respected by the CLI when provided (`src/search/service.py:199`).
- Results are returned as Pandas data frames enriched with distances, paths, and stored payloads so downstream code (streamlit, API responses, tests) can present consistent views.

---

## CLI Cheatsheet

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

## Configuration & Tuning

### Index Creation (IVF-PQ)

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

### Ingestion Performance

For large datasets (e.g., millions of files), tune these env vars in `.env` to improve throughput:

- `INGEST_WORKERS`: S3 download concurrency. Try 32–128 depending on your MinIO and network.
- `INGEST_EMBED_BATCH`: Batch size for text/image/pdf embeddings. Increase with GPU/CPU memory (e.g., 128–512).
- `LIDAR_EMBED_BATCH`: Explicit LiDAR batch size. If unset (0), it uses `INGEST_EMBED_BATCH // LIDAR_BATCH_DIVISOR`.
- `LIDAR_BATCH_DIVISOR`: Lower value → larger default LiDAR batch (e.g., 4).
- `INGEST_WRITE_CHUNK`: Rows per DB write chunk (e.g., 10000–50000) to reduce write overhead.
- `INGEST_PDF_MAX_PAGES`: Fewer pages per PDF speeds ingestion.
- `INGEST_PROGRESS_EVERY`: Print progress every N processed items (0 = per-batch prints).
- `MINIO_SUPPRESS_TLS_WARN`: Set to 1 to suppress HTTPS warnings when `MINIO_VERIFY=false`.

Example high-throughput settings:

```env
INGEST_WORKERS=64
INGEST_EMBED_BATCH=256
LIDAR_EMBED_BATCH=64
INGEST_WRITE_CHUNK=20000
INGEST_PROGRESS_EVERY=1000
```

### Distance Metric & Recall Tuning

- **Metric selection**
  - CLI: add `--metric {l2,cosine,dot}` (default from `LANCEDB_METRIC`, falls back to `l2`).
  - Web UI: use the Metric dropdown (Cosine/Dot recommended). The results table shows Distance (L2) or Similarity (Cosine/Dot).
  - Note: embeddings are L2-normalized, so `dot` is equivalent to `cosine` in practice.

- **ANN recall vs latency**
  - `LANCEDB_NPROBES` (default 32): more probes → higher recall, slower.
  - `LANCEDB_REFINE_FACTOR` (default 50): re-rank more candidates with exact distances.
  - Set these env vars in `.env` to tune web UI and API behavior; CLI also honors them.

---

## Notes

- `LANCEDB_URI` can be local (`data/lancedb`) or S3 (`s3://bucket/prefix`).
- For S3 LanceDB, the key needs write access to that prefix (Put/Get/Delete/AbortMultipartUpload).
- If your endpoint is `http://`, set `LANCEDB_ALLOW_HTTP=1`.

---

## Module Map

- `src/config/` — loads `.env`, normalises MINIO ↔ AWS vars, and exposes typed config helpers.
- `src/storage/` — thin MinIO/S3 helper (list/get/upload) via `boto3`.
- `src/vectordb/` — LanceDB manager implementation plus shared base classes (`src/lancedb_manager.py` is a compatibility shim).
- `src/embedding/` — modality embedders and shared utilities (`ImageEmbedder`, `TextEmbedder`, `LidarBEVEmbedder`).
- `src/embedding_runtime.py` — factory returning real or dummy embedders for runtime/tests.
- `src/core/` — cross-cutting runtime primitives (search helpers, orchestration utilities).
- `src/ingest/` — modular ingestion helpers and pipeline for PDFs/images/text/LiDAR.
- `src/search/` — similarity search orchestration (`SearchService`, `MultiModalSearcher`).
- `src/services.py` — shim that re-exports the container from `src/core/services`.
- `src/cli/` — CLI application and utilities (`src/main.py` is the entry point).
- `src/web/` — FastAPI search UI (`src/webapp.py` is a compatibility shim).

---

## Troubleshooting

- **403 on ingest to S3** → grant write to your `LANCEDB_URI` prefix or use a local path.
- **No web results** → confirm table name (banner shows it) and that rows > 0.
- **Noisy PDF logs** → expected on messy PDFs; unreadable files are skipped.
