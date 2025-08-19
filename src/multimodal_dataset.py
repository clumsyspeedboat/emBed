"""Generate synthetic multimodal data for testing LanceDB."""

import numpy as np
from PIL import Image
import io
from .embedding import DummyTextEmbedder, DummyImageEmbedder, DummyLidarEmbedder


def generate_dummy_dataset(n: int = 5) -> list[dict]:
    text_embedder = DummyTextEmbedder()
    img_embedder = DummyImageEmbedder()
    lidar_embedder = DummyLidarEmbedder()

    dataset = []
    for i in range(n):
        # text
        text = f"sample text {i}"
        dataset.append({
            "id": f"text-{i}",
            "modality": "text",
            "embedding": text_embedder.embed(text).tolist(),
            "text": text,
            "image": None,
            "lidar": None,
        })

        # image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        dataset.append({
            "id": f"image-{i}",
            "modality": "image",
            "embedding": img_embedder.embed(img_bytes).tolist(),
            "text": None,
            "image": img_bytes,
            "lidar": None,
        })

        # lidar
        pts = np.random.rand(10, 3).tolist()
        dataset.append({
            "id": f"lidar-{i}",
            "modality": "lidar",
            "embedding": lidar_embedder.embed(pts).tolist(),
            "text": None,
            "image": None,
            "lidar": pts,
        })
    return dataset
