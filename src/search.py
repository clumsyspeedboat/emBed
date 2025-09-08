"""Search utilities for querying a multimodal LanceDB table."""
import pandas as pd
import time
from src.embedding import TextEmbedder, ImageEmbedder, LidarBEVEmbedder

class MultiModalSearcher:
    def __init__(self, table) -> None:
        self.table = table
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.lidar_embedder = LidarBEVEmbedder(self.image_embedder)

    def search(self, query, modality: str, top_k: int = 3, where: str | None = None) -> pd.DataFrame:
        t0 = time.perf_counter()

        if modality == "text":
            vec = self.text_embedder.embed(query).tolist()
        elif modality == "image":
            with open(query, "rb") as f:
                vec = self.image_embedder.embed(f.read()).tolist()
        elif modality == "lidar":
            # This part might need adjustment based on how LiDAR data is passed.
            # Assuming 'query' would be a numpy array or path to a file for LiDAR.
            # For now, let's assume it's a placeholder for actual data.
            vec = self.lidar_embedder.embed(query).tolist()
        else:
            raise ValueError(f"Unknown modality: {modality}")

        q = self.table.search(vec)
        if where:
            q = q.where(where)
        
        results = q.limit(top_k).to_pandas()
        
        elapsed = time.perf_counter() - t0
        print(f"\nSearch completed in: {elapsed:.4f} seconds")
        
        return results