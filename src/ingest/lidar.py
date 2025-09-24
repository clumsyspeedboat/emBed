"""Purpose: parse LiDAR file formats into numpy arrays consumable by embedders.
Why extend: add support for more sensors or metadata while keeping the ingest pipeline clean.
How extend: extend `load_lidar_bytes` to branch on new extensions or expose helper functions that normalise additional channels.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _pcd_parse_header(blob: bytes) -> dict:
    header_lines = []
    pos = 0
    while True:
        nl = blob.find(b"\n", pos)
        if nl == -1:
            break
        line = blob[pos:nl].decode("utf-8", errors="ignore").strip()
        header_lines.append(line)
        pos = nl + 1
        if line.upper().startswith("DATA"):
            break
    meta = {"header_len": pos}
    for ln in header_lines:
        up = ln.upper()
        if up.startswith("FIELDS"):
            meta["FIELDS"] = ln.split()[1:]
        elif up.startswith("SIZE"):
            meta["SIZE"] = [int(x) for x in ln.split()[1:]]
        elif up.startswith("TYPE"):
            meta["TYPE"] = ln.split()[1:]
        elif up.startswith("COUNT"):
            meta["COUNT"] = [int(x) for x in ln.split()[1:]]
        elif up.startswith("WIDTH"):
            meta["WIDTH"] = int(ln.split()[1])
        elif up.startswith("HEIGHT"):
            meta["HEIGHT"] = int(ln.split()[1])
        elif up.startswith("POINTS"):
            try:
                meta["POINTS"] = int(ln.split()[1])
            except Exception:
                pass
        elif up.startswith("DATA"):
            meta["DATA"] = ln.split()[1].lower()
    return meta


def _pcd_dtype(fields: List[str], sizes: List[int], types: List[str], counts: List[int]) -> np.dtype:
    type_map = {
        ("F", 4): np.float32,
        ("F", 8): np.float64,
        ("U", 1): np.uint8,
        ("U", 2): np.uint16,
        ("U", 4): np.uint32,
        ("I", 1): np.int8,
        ("I", 2): np.int16,
        ("I", 4): np.int32,
    }
    fields_expanded = []
    for f, sz, tp, cnt in zip(fields, sizes, types, counts):
        if cnt == 1:
            fields_expanded.append((f, type_map.get((tp.upper(), sz), np.float32)))
        else:
            fields_expanded.append((f, (type_map.get((tp.upper(), sz), np.float32), cnt)))
    return np.dtype(fields_expanded)


def _pcd_to_numpy(blob: bytes) -> np.ndarray:
    meta = _pcd_parse_header(blob)
    data_mode = meta.get("DATA", "ascii")
    start = meta.get("header_len", 0)
    fields = meta.get("FIELDS", [])
    sizes = meta.get("SIZE", [4] * len(fields))
    types = meta.get("TYPE", ["F"] * len(fields))
    counts = meta.get("COUNT", [1] * len(fields))
    n_points = meta.get("POINTS")
    width = meta.get("WIDTH")
    height = meta.get("HEIGHT")

    def idx_of(name: str) -> int | None:
        try:
            return fields.index(name)
        except ValueError:
            return None

    ix_x, ix_y, ix_z = idx_of("x"), idx_of("y"), idx_of("z")
    ix_i = idx_of("intensity")

    if any(v is None for v in (ix_x, ix_y, ix_z)):
        txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
        arr = np.asarray([float(x) for x in txt], dtype=np.float32)
        cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
        return arr.reshape(-1, cols)[:, :3].astype(np.float32)

    if data_mode == "ascii":
        pts = []
        for ln in blob[start:].decode("utf-8", errors="ignore").splitlines():
            parts = ln.split()
            if len(parts) < max(ix_x, ix_y, ix_z) + 1:
                continue
            x = float(parts[ix_x]); y = float(parts[ix_y]); z = float(parts[ix_z])
            if ix_i is not None and len(parts) > ix_i:
                try:
                    i = float(parts[ix_i]); pts.append((x, y, z, i)); continue
                except Exception:
                    pass
            pts.append((x, y, z))
        return np.asarray(pts, dtype=np.float32)

    if "binary" in data_mode:
        dt = _pcd_dtype(fields, sizes, types, counts)
        if n_points is not None:
            n = n_points
        elif width and height:
            n = int(width) * int(height)
        else:
            n = None
        buf = memoryview(blob)[start:]
        arr = np.frombuffer(buf, dtype=dt, count=n)
        out = np.zeros((arr.shape[0], 3 + (1 if ix_i is not None else 0)), dtype=np.float32)
        out[:, 0] = np.asarray(arr["x"], dtype=np.float32)
        out[:, 1] = np.asarray(arr["y"], dtype=np.float32)
        out[:, 2] = np.asarray(arr["z"], dtype=np.float32)
        if ix_i is not None and "intensity" in arr.dtype.names:
            out[:, 3] = np.asarray(arr["intensity"], dtype=np.float32)
        return out

    txt = blob[start:].decode("utf-8", errors="ignore").strip().split()
    arr = np.asarray([float(x) for x in txt], dtype=np.float32)
    cols = 3 if (arr.size % 3 == 0) else 4 if (arr.size % 4 == 0) else 3
    return arr.reshape(-1, cols)[:, :3].astype(np.float32)


def load_lidar_bytes(ext: str, blob: bytes) -> np.ndarray:
    if ext == ".bin":
        arr = np.frombuffer(blob, dtype=np.float32)
        cols = 5 if arr.size % 5 == 0 else 4 if arr.size % 4 == 0 else 3
        return arr.reshape(-1, cols)
    if ext == ".pcd":
        return _pcd_to_numpy(blob)
    raise ValueError("Unsupported LiDAR extension")


__all__ = ["load_lidar_bytes"]
