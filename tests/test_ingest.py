import numpy as np

from src.ingest import ingest_s3_objects, TEXT_EXTS, IMAGE_EXTS, LIDAR_EXTS
from src.embedding_runtime import DummyTextEmbedder, DummyImageEmbedder, DummyLidarEmbedder, reset_singletons_for_tests

# ---- fakes for S3 & LanceDB ----

class _FakeS3:
    def __init__(self, objects):
        # objects: dict key->bytes
        self._objects = dict(objects)

    def list_objects(self, bucket, prefix=""):
        # flat listing with prefix filter
        return [k for k in self._objects.keys() if k.startswith(prefix)]

    def get_object_bytes(self, bucket, key):
        return self._objects[key]

class _FakeTable:
    def __init__(self):
        self.rows = []

    def delete(self, where):
        # support id IN ('a','b',...) and equality
        if "IN (" in where:
            ids = where.split("IN (", 1)[1].rstrip(")").strip()
            ids = [x.strip().strip("'") for x in ids.split(",")]
            self.rows = [r for r in self.rows if r["id"] not in ids]
        elif " = " in where:
            key, val = where.split(" = ", 1)
            key = key.strip()
            val = val.strip().strip("'")
            self.rows = [r for r in self.rows if str(r.get(key)) != val]

    def add(self, rows):
        self.rows.extend(rows)

class _FakeVDB:
    def __init__(self):
        self.tables = {}

    def create_table(self, name, rows, mode="overwrite"):
        self.tables[name] = _FakeTable()
        # create with schema by adding provided rows (if any)
        if rows:
            self.tables[name].add(rows)

    def get_table(self, name):
        return self.tables[name]

# ---- tests ----

def test_end_to_end_text_image_lidar(monkeypatch):
    reset_singletons_for_tests()

    # dummy embedders (fast + deterministic)
    text = DummyTextEmbedder(dim=8)
    image = DummyImageEmbedder(dim=8)
    lidar = DummyLidarEmbedder(dim=8)

    # build three modalities
    txt_bytes = b"hello world"
    img_bytes = b"\x89PNG\r\n\x1a\nRAWIMG"
    lidar_pts = np.array([[0, 0, 0, 1], [1, 2, 3, 0]], dtype=np.float32).reshape(-1)
    lidar_bytes = lidar_pts.tobytes()

    s3 = _FakeS3({
        "p/doc.txt": txt_bytes,
        "p/img.jpg": img_bytes,
        "p/scan.bin": lidar_bytes,
        "p/ignore.xyz": b"nope",
    })
    vdb = _FakeVDB()

    rows = ingest_s3_objects(
        manager=vdb,
        s3=s3,
        bucket="bucket",
        prefix="p/",
        table_name="tbl",
        include_ext=TEXT_EXTS | IMAGE_EXTS | LIDAR_EXTS,
        mode="overwrite",
        workers=4,
        embed_batch=16,
        write_chunk=100,
        # DI embedders
        text_embedder=text,
        image_embedder=image,
        lidar_embedder=lidar,
    )
    assert rows == 3

    tbl = vdb.get_table("tbl")
    assert len(tbl.rows) == 3
    mods = sorted({r["modality"] for r in tbl.rows})
    assert mods == ["image", "lidar", "text"]
    for r in tbl.rows:
        assert len(r["embedding"]) == 8
        assert r["path"].startswith("s3://bucket/")

def test_append_mode_is_idempotent(monkeypatch):
    reset_singletons_for_tests()
    t = DummyTextEmbedder(dim=4)
    i = DummyImageEmbedder(dim=4)
    l = DummyLidarEmbedder(dim=4)

    img_bytes = b"\x89PNG\r\n\x1a\nIMG"
    s3 = _FakeS3({"p/a.jpg": img_bytes})
    vdb = _FakeVDB()

    # first pass
    n1 = ingest_s3_objects(
        vdb, s3, "bucket", "p/", "tbl",
        include_ext=IMAGE_EXTS,
        mode="append",
        text_embedder=t, image_embedder=i, lidar_embedder=l,
        workers=2, embed_batch=8, write_chunk=10
    )
    assert n1 == 1
    assert len(vdb.get_table("tbl").rows) == 1

    # second pass (same key) should not duplicate rows
    n2 = ingest_s3_objects(
        vdb, s3, "bucket", "p/", "tbl",
        include_ext=IMAGE_EXTS,
        mode="append",
        text_embedder=t, image_embedder=i, lidar_embedder=l,
        workers=2, embed_batch=8, write_chunk=10
    )
    assert n2 == 1
    assert len(vdb.get_table("tbl").rows) == 1

def test_lidar_is_batched_when_supported(monkeypatch):
    reset_singletons_for_tests()

    class _CountingLidar:
        def __init__(self, dim=6):
            self.dim = dim
            self.calls = 0
        def embed_pointclouds(self, clouds):
            self.calls += 1
            # return a zero vector for each cloud
            return np.stack([np.zeros(self.dim, dtype=np.float32) for _ in clouds], axis=0)

    lcount = _CountingLidar(dim=6)
    i = DummyImageEmbedder(dim=6)
    t = DummyTextEmbedder(dim=6)

    # five lidar files -> expect one embed_pointclouds() call when batch size is large
    pts = np.array([[0, 0, 0, 1], [1, 2, 3, 0]], dtype=np.float32).reshape(-1).tobytes()
    s3 = _FakeS3({f"p/{k}.bin": pts for k in range(5)})
    vdb = _FakeVDB()

    n = ingest_s3_objects(
        vdb, s3, "bucket", "p/", "tbl",
        include_ext=LIDAR_EXTS,
        mode="overwrite",
        text_embedder=t, image_embedder=i, lidar_embedder=lcount,
        workers=2, embed_batch=64, write_chunk=100
    )
    assert n == 5
    assert lcount.calls == 1  # batched once
