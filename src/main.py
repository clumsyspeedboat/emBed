"""User-friendly CLI for your LanceDB + MinIO workflow."""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict, Optional, Iterable

import pandas as pd

from .config import MinioConfig, LanceDBConfig
from .storage import MinioClient
from .lancedb_manager import LanceDBManager
from .search import MultiModalSearcher
from .ingest import ingest_s3_objects, DEFAULT_EXTS
from .cli_utils import echo, heading, kv_table, simple_table, panel, mask


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


def cmd_config(minio: MinioConfig, lcfg: LanceDBConfig) -> None:
    heading("Active Configuration")
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
    panel("""
Tips:
- If using HTTP (not HTTPS), set LANCEDB_ALLOW_HTTP=1 and/or ALLOW_HTTP=true.
- For S3 LanceDB URI, ensure AWS_* envs are set or MINIO_* are bridged.
- You can keep LanceDB local (data/lancedb) and still ingest from S3.""", title="Notes")


def cmd_health(minio: MinioConfig, lcfg: LanceDBConfig) -> None:
    heading("Health Check")
    s3 = MinioClient(minio)
    rows = []
    for bucket in [b.strip() for b in minio.buckets.split(",") if b.strip()]:
        try:
            keys = s3.list_objects(bucket, "")
            rows.append([f"s3://{bucket}/", "READ list", f"OK ({len(keys)} objects listed)"])
        except Exception as e:
            rows.append([f"s3://{bucket}/", "READ list", f"FAIL ({e})"])
    if _is_remote(lcfg.uri) and lcfg.uri.startswith("s3://"):
        try:
            _, rest = lcfg.uri.split("s3://", 1)
            bucket, *parts = rest.split("/", 1)
            prefix = parts[0] if parts else ""
            test_key = (prefix.rstrip("/") + "/_health_write_test.txt").lstrip("/")
            s3.upload_file(file_path=__file__, bucket=bucket, key=test_key)
            rows.append([lcfg.uri, "WRITE put", "OK"])
            s3._client.delete_object(Bucket=bucket, Key=test_key)
            rows.append([lcfg.uri, "WRITE delete", "OK"])
        except Exception as e:
            rows.append([lcfg.uri, "WRITE put/delete", f"FAIL ({e})"])
    else:
        rows.append([lcfg.uri, "Local path", "OK"])
    simple_table("Connectivity", ["Target", "Operation", "Result"], rows)


def _summarize_keys(keys: Iterable[str], show=20):
    # small helper to display prefixes and ext counts
    from collections import Counter, defaultdict
    exts = Counter()
    prefixes = defaultdict(int)
    for k in keys:
        exts[os.path.splitext(k)[1].lower()] += 1
        p = k.split("/", 1)[0] if "/" in k else ""
        prefixes[p] += 1
    top_ext = ", ".join(f"{e or '(none)'}:{c}" for e, c in exts.most_common(10))
    top_pref = ", ".join(f"{p or '(root)'}:{c}" for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:10])
    return top_ext, top_pref, list(keys)[:show]


def cmd_ls_s3(bucket: str, prefix: str, limit: int, minio: MinioConfig) -> None:
    s3 = MinioClient(minio)
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


def cmd_ingest(bucket: str, prefix: str, table: str, minio: MinioConfig, lcfg: LanceDBConfig,
               include_ext: Optional[list[str]], exclude_ext: Optional[list[str]],
               max_files: Optional[int], dry_run: bool, mode: str) -> None:
    manager = LanceDBManager(lcfg)
    s3 = MinioClient(minio)
    ingest_s3_objects(manager, s3, bucket, prefix or "", table,
                      include_ext=include_ext, exclude_ext=exclude_ext,
                      max_files=max_files, dry_run=dry_run, mode=mode)


def _snippet(row: Dict[str, Any], max_chars: int = 120) -> str:
    txt = row.get("text") or ""
    if not isinstance(txt, str):
        return ""
    txt = " ".join(txt.split())
    return (txt[:max_chars] + "…") if len(txt) > max_chars else txt


def cmd_search(table: str, modality: str, query: str, topk: int, where: Optional[str],
               lcfg: LanceDBConfig) -> None:
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
            f"{r.get('_distance', ''):.4f}" if "_distance" in r else "",
            _snippet(r.to_dict()),
            r.get("path", ""),
        ])
    simple_table(f"Top {len(rows)}", ["id", "mod", "dist", "snippet", "path"], rows)


def cmd_peek(table: str, n: int, lcfg: LanceDBConfig) -> None:
    manager = LanceDBManager(lcfg)
    tbl = manager.get_table(table)
    df = tbl.to_pandas()
    df = df.head(n)
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
        panel("""
Refusing to wipe a REMOTE LanceDB URI from this CLI for safety.

If you truly want to clear the S3 prefix, change LANCEDB_URI to a local path and run:
  python -m src.main reset

Or delete the 'lancedb/' prefix manually using your S3/MinIO admin tool.
""", title="Safety")
        return
    _clear_local(lcfg.uri)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multimodal LanceDB CLI (friendly)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="Show resolved configuration & envs")
    sub.add_parser("health", help="Connectivity & write test")

    pls = sub.add_parser("ls-s3", help="List objects under a bucket/prefix")
    pls.add_argument("--bucket", required=True)
    pls.add_argument("--prefix", nargs="?", const="", default="", help="Optional prefix (empty = root)")
    pls.add_argument("--limit", type=int, default=30, help="Show up to N example keys")
    

    pi = sub.add_parser("ingest-s3", help="Ingest images/PDFs from S3/MinIO")
    pi.add_argument("--bucket", required=True)
    pi.add_argument("--prefix", nargs="?", const="", default="", help="Optional prefix (empty = root)")
    pi.add_argument("--table", default="multimodal")
    pi.add_argument("--include-ext", nargs="*", default=None,
                    help=f"Whitelist extensions (default: {sorted(DEFAULT_EXTS)})")
    pi.add_argument("--exclude-ext", nargs="*", default=None,
                    help="Blacklist extensions")
    pi.add_argument("--max-files", type=int, default=None, help="Limit number of files processed")
    pi.add_argument("--dry-run", action="store_true", help="Preview only, don't write")
    pi.add_argument("--mode", choices=["overwrite", "append"], default="overwrite")

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
        cmd_ingest(args.bucket, args.prefix, args.table, minio, lcfg,
                   args.include_ext, args.exclude_ext, args.max_files, args.dry_run, args.mode)
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
