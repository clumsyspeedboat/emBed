"""Search utilities for querying a multimodal LanceDB table."""
import pandas as pd
from .embedding import DummyTextEmbedder, DummyImageEmbedder, DummyLidarEmbedder

class MultiModalSearcher:
    def __init__(self, table) -> None:
        self.table = table
        self.text_embedder = DummyTextEmbedder()
        self.image_embedder = DummyImageEmbedder()
        self.lidar_embedder = DummyLidarEmbedder()

    def search(self, query, modality: str, top_k: int = 3, where: str | None = None) -> pd.DataFrame:
        if modality == "text":
            vec = self.text_embedder.embed(query).tolist()
        elif modality == "image":
            with open(query, "rb") as f:
                vec = self.image_embedder.embed(f.read()).tolist()
        elif modality == "lidar":
            vec = self.lidar_embedder.embed(query).tolist()
        else:
            raise ValueError(f"Unknown modality: {modality}")

        q = self.table.search(vec)
        if where:
            q = q.where(where)
        return q.limit(top_k).to_pandas()
