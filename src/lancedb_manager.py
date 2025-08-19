import os
import lancedb
from .config import LanceDBConfig

class LanceDBManager:
    def __init__(self, config: LanceDBConfig) -> None:
        self._config = config
        uri = config.uri

        # Make local path absolute and ensure directory exists
        if not (uri.startswith("s3://") or uri.startswith("gs://") or uri.startswith("az://")):
            uri = os.path.abspath(uri)
            os.makedirs(uri, exist_ok=True)
            self._db = lancedb.connect(uri=uri)  # local FS: no storage_options
        else:
            opts = config.storage_options()
            # Only pass storage_options if we actually have any
            self._db = lancedb.connect(uri=uri, storage_options=opts if opts else None)

    def create_table_with_data(self, name: str, data: list[dict], mode: str = "overwrite"):
        # Let LanceDB infer schema from data
        return self._db.create_table(name, data=data, mode=mode)

    def get_table(self, name: str):
        return self._db.open_table(name)

    def upsert_data(self, name: str, data: list[dict]):
        tbl = self.get_table(name)
        tbl.add(data)
        return tbl
