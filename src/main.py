"""User-friendly CLI for your LanceDB + MinIO workflow.

Highlights
- Safe health checks using public MinIOClient methods (no direct `_client` access)
- Integrated env snapshot and improved progress/spinner UX when available
- Normalized include/exclude extensions
- Optional automatic index creation after ingestion with `--create-index` flag
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.config import MinioConfig, LanceDBConfig
from src.storage import MinIOClient
from src.lancedb_manager import LanceDBManager
from src.search import MultiModalSearcher
from src.ingest import ingest_s3_objects, DEFAULT_EXTS
from src.cli_utils import (
    echo, heading, kv_table, simple_table, panel, mask, env_snapshot,
    status
)

# ----- helpers -----

def _is_remote(uri: str) -> bool:
    return uri.startswith(("s3://", "gs://", "az://"))

def _clear_local(uri: str) -> None:
    if _is_remote(uri):
        echo("[yellow]LanceDB URI is remote; skipping local clear.[/yellow]")
        return
    path = os.path.abspath(uri)
    if os.path.isdir(path):
        shutil.rmtree(path)
        echo(f"[green]Cleared local LanceDB path:[/green] {path}")
    else:
        echo(f"[yellow]No local LanceDB path at:[/yellow] {path}")

def _summarize_keys(keys: Iterable[str], show: int = 20):
    from collections import Counter, defaultdict
    exts = Counter()
    prefixes = defaultdict(int)
    for k in keys:
        ext = os.path.splitext(k)[1].lower()
        exts[ext] += 1
        p = k.split("/", 1)[0] if "/" in k else ""
        prefixes[p] += 1
    top_ext = ", ".join(f"{e or '(none)'}:{c}" for e, c in exts.most_common(10))
    top_pref = ", ".join(
        f"{p or '(root)'}:{c}"
        for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:10]
    )
    return top_ext, top_pref, list(keys)[:show]

# ----- commands -----

def cmd_config(minio: MinioConfig, lcfg: LanceDBConfig) -> None:
    heading("Active Configuration")

    # Show LanceDB + S3 options (mask secrets)
    kv_table("LanceDB", {
        "URI": lcfg.uri,
        "Remote?": str(_is_remote(lcfg.uri)),
        "Storage Options": lcfg.storage_options(),
    })

    kv_table("MinIO/S3", {
        "Endpoint": minio.endpoint,
        "Region": minio.region,
        "Buckets": minio.buckets,
        "AWS_ENDPOINT": os.getenv("AWS_ENDPOINT", ""),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", ""),
        "AWS_ACCESS_KEY_ID": mask(os.getenv("AWS_ACCESS_KEY_ID")),
        "AWS_SECRET_ACCESS_KEY": mask(os.getenv("AWS_SECRET_ACCESS_KEY")),
    })

    env_snapshot(
        [
            "MINIO_ENDPOINT", "MINIO_REGION", "MINIO_VERIFY", "MINIO_ALLOW_HTTP",
            "MINIO_FORCE_PATH_STYLE", "MINIO_VIRTUAL_HOSTED_STYLE",
            "MINIO_BUCKETS", "LANCEDB_URI", "LANCEDB_ALLOW_HTTP",
            "AWS_ENDPOINT", "AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        ],
        title="Environment Snapshot"
    )

    panel(
        """
Tips:
- If using HTTP (not HTTPS), set LANCEDB_ALLOW_HTTP=1 and/or MINIO_ALLOW_HTTP=true.
- For S3 LanceDB URI, ensure AWS_* envs are set or MINIO_* are bridged.
- You can keep LanceDB local (e.g., data/lancedb) and still ingest FROM S3.
""",
        title="Notes"
    )

def cmd_health(minio_cfg: MinioConfig, lancedb_cfg: LanceDBConfig) -> None:
    """Connectivity checks for S3 list and LanceDB remote open."""
    heading("Health Check")

    s3 = MinIOClient(minio_cfg)
    bucket = (minio_cfg.buckets or "").split(",")[0].strip() or ""

    # S3 READ
    if not bucket:
        kv_table("S3 READ list", {"Result": "FAIL", "Reason": "No bucket in MINIO_BUCKETS"})
    else:
        with status("Listing up to 1000 objects"):
            try:
                objs = s3.list_objects(bucket=bucket, prefix="", limit=1000)
                kv_table("S3 READ list", {"Bucket": bucket, "Listed": len(objs), "Result": "OK"})
            except Exception as e:
                kv_table("S3 READ list", {"Bucket": bucket, "Result": "FAIL", "Error": str(e)})

    # S3 WRITE/DELETE
    if bucket:
        key = f"_healthcheck/{uuid.uuid4().hex}.ping"
        with status("S3 write/delete"):
            try:
                # Write temp file
                with tempfile.TemporaryDirectory() as td:
                    fp = os.path.join(td, "ping.txt")
                    with open(fp, "wb") as f:
                        f.write(b"ok")
                    s3.upload_file(fp, bucket, key)
                s3.delete_objects(bucket, [key])
                kv_table("S3 WRITE/DELETE", {"Bucket": bucket, "Key": key, "Result": "OK"})
            except Exception as e:
                kv_table("S3 WRITE/DELETE", {"Bucket": bucket, "Key": key, "Result": "FAIL", "Error": str(e)})

    # LanceDB connect
    with status("LanceDB connect/open"):
        try:
            mgr = LanceDBManager(lancedb_cfg)
            mgr.connect(lancedb_cfg.uri, lancedb_cfg.storage_options())
            kv_table("LanceDB", {"URI": lancedb_cfg.uri, "Connected": "Yes"})
        except Exception as e:
            kv_table("LanceDB", {"URI": lancedb_cfg.uri, "Connected": "No", "Error": str(e)})

def cmd_ls_s3(bucket: str, prefix: str, limit: int, minio: MinioConfig) -> None:
    s3 = MinIOClient(minio)
    with status(f"Listing s3://{bucket}/{prefix or ''}"):
        keys = s3.list_objects(bucket, prefix or "")
    heading(f"S3 List: s3://{bucket}/{prefix}")
    if not keys:
        echo("[yellow]No objects found.[/yellow]")
        return
    ext_summary, pref_summary, examples = _summarize_keys(keys, show=min(limit, 50))
    kv_table("Summary", {
        "Objects": len(keys),
        "Top extensions": ext_summary,
        "Top first-level prefixes": pref_summary
    })
    simple_table("Examples", ["key"], [[k] for k in examples])

def cmd_ingest(
    bucket: str, prefix: str, table: str,
    minio: MinioConfig, lcfg: LanceDBConfig,
    include_ext: Optional[list[str]], exclude_ext: Optional[list[str]],
    max_files: Optional[int], dry_run: bool, mode: str, create_index: bool
) -> None:
    manager = LanceDBManager(lcfg)
    s3 = MinIOClient(minio)

    def _norm(exts: Optional[list[str]]) -> Optional[list[str]]:
        if not exts:
            return None
        out = []
        for e in exts:
            e = e.lower()
            if not e.startswith("."):
                e = f".{e}"
            out.append(e)
        return sorted(set(out))

    include_ext = _norm(include_ext)
    exclude_ext = _norm(exclude_ext)

    rows_ingested = ingest_s3_objects(
        manager, s3, bucket, prefix or "", table,
        include_ext=include_ext, exclude_ext=exclude_ext,
        max_files=max_files, dry_run=dry_run, mode=mode
    )

    # Auto-index
    if create_index and not dry_run and rows_ingested > 0:
        echo("\n--- Auto-creating index post-ingestion ---")
        cmd_create_index(table, lcfg)

def cmd_create_index(table: str, lcfg: LanceDBConfig) -> None:
    t0 = time.perf_counter()
    manager = LanceDBManager(lcfg)
    try:
        manager.create_index(table)
        duration = time.perf_counter() - t0
        echo(f"[green]Successfully created index for table '{table}' in {duration:.2f}s.[/green]")
    except Exception as e:
        echo(f"[red]Error creating index: {e}[/red]")

def _snippet(row: Dict[str, Any], max_chars: int = 120) -> str:
    txt = row.get("text") or ""
    if not isinstance(txt, str):
        return ""
    txt = " ".join(txt.split())
    return (txt[:max_chars] + "…") if len(txt) > max_chars else txt

def cmd_search(
    table: str, modality: str, query: str,
    topk: int, where: Optional[str], lcfg: LanceDBConfig
) -> None:
    manager = LanceDBManager(lcfg)
    tbl = manager.get_table(table)
    searcher = MultiModalSearcher(tbl)
    df: pd.DataFrame = searcher.search(query, modality, topk, where)
    heading("Search Results")
    rows = []
    for _, r in df.iterrows():
        rows.append([
            r.get("id", ""),
            r.get("modality", ""),
            f"{r.get('_distance', 0.0):.4f}" if "_distance" in r else "",
            _snippet(r.to_dict()),
            r.get("path", ""),
        ])
    simple_table(f"Top {len(rows)}", ["id", "mod", "dist", "snippet", "path"], rows)

def cmd_peek(table: str, n: int, lcfg: LanceDBConfig) -> None:
    manager = LanceDBManager(lcfg)
    tbl = manager.get_table(table)
    df = tbl.to_pandas().head(n)
    heading(f"Peek table='{table}'")
    rows = []
    for _, r in df.iterrows():
        rows.append([r.get("id", ""), r.get("modality", ""), r.get("path", ""), _snippet(r.to_dict())])
    simple_table(f"First {len(rows)} rows", ["id", "mod", "path", "snippet"], rows)

def cmd_stats(table: str, lcfg: LanceDBConfig) -> None:
    manager = LanceDBManager(lcfg)
    tbl = manager.get_table(table)
    df = tbl.to_pandas()
    by_mod = df.groupby("modality").size().reset_index(name="count")
    heading(f"Stats table='{table}'")
    rows = [[m, int(c)] for m, c in zip(by_mod["modality"], by_mod["count"])]
    simple_table("Rows by modality", ["modality", "count"], rows)
    total = len(df)
    kv_table("Totals", {"rows": total, "columns": ", ".join(list(df.columns))})

def cmd_reset(lcfg: LanceDBConfig) -> None:
    if _is_remote(lcfg.uri):
        panel(
            """
Refusing to wipe a REMOTE LanceDB URI from this CLI for safety.

If you truly want to clear the S3 prefix, change LANCEDB_URI to a local path and run:
  python -m src.main reset

Or delete the 'lancedb/' prefix manually using your S3/MinIO admin tool.
""",
            title="Safety"
        )
        return
    _clear_local(lcfg.uri)

# ----- argparse -----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multimodal LanceDB CLI (friendly)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="Show resolved configuration & envs")
    sub.add_parser("health", help="Connectivity & write test")

    pls = sub.add_parser("ls-s3", help="List objects under a bucket/prefix")
    pls.add_argument("--bucket", required=True)
    pls.add_argument("--prefix", nargs="?", const="", default="", help="Optional prefix (empty = root)")
    pls.add_argument("--limit", type=int, default=30, help="Show up to N example keys")

    pi = sub.add_parser("ingest-s3", help="Ingest images/PDFs/LiDAR from S3/MinIO")
    pi.add_argument("--bucket", required=True)
    pi.add_argument("--prefix", nargs="?", const="", default="", help="Optional prefix (empty = root)")
    pi.add_argument("--table", default="multimodal")
    pi.add_argument(
        "--include-ext", nargs="*", default=None,
        help=f"Whitelist extensions (default: {sorted(DEFAULT_EXTS)})"
    )
    pi.add_argument("--exclude-ext", nargs="*", default=None, help="Blacklist extensions")
    pi.add_argument("--max-files", type=int, default=None, help="Limit number of files processed")
    pi.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    pi.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")
    pi.add_argument("--create-index", action="store_true", help="Create a search index after ingestion completes")

    p_idx = sub.add_parser("create-index", help="Create a performance index for a table (run after ingest)")
    p_idx.add_argument("--table", default="multimodal", help="Name of the table to index")

    ps = sub.add_parser("search", help="Vector search")
    ps.add_argument("--table", default="multimodal")
    ps.add_argument("--modality", choices=["text", "image", "lidar"], default="text")
    ps.add_argument("--query", required=True)
    ps.add_argument("--topk", type=int, default=5)
    ps.add_argument("--where", type=str, default=None, help="e.g. \"modality = 'pdf'\"")

    pp = sub.add_parser("peek", help="Show first N rows")
    pp.add_argument("--table", default="multimodal")
    pp.add_argument("--n", type=int, default=10)

    pst = sub.add_parser("stats", help="Basic table stats")
    pst.add_argument("--table", default="multimodal")

    sub.add_parser("reset", help="Delete local LanceDB directory (safe; no S3 delete)")

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    minio = MinioConfig()
    lcfg = LanceDBConfig()

    if args.cmd == "config":
        cmd_config(minio, lcfg)
    elif args.cmd == "health":
        cmd_health(minio, lcfg)
    elif args.cmd == "ls-s3":
        cmd_ls_s3(args.bucket, args.prefix, args.limit, minio)
    elif args.cmd == "ingest-s3":
        cmd_ingest(
            args.bucket, args.prefix, args.table, minio, lcfg,
            args.include_ext, args.exclude_ext, args.max_files,
            args.dry_run, args.mode, args.create_index
        )
    elif args.cmd == "create-index":
        cmd_create_index(args.table, lcfg)
    elif args.cmd == "search":
        cmd_search(args.table, args.modality, args.query, args.topk, args.where, lcfg)
    elif args.cmd == "peek":
        cmd_peek(args.table, args.n, lcfg)
    elif args.cmd == "stats":
        cmd_stats(args.table, lcfg)
    elif args.cmd == "reset":
        cmd_reset(lcfg)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
