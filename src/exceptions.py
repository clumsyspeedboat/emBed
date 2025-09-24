"""Purpose: centralise custom exception types used across ingestion, search, and storage modules.
Why extend: introduce domain-specific errors so higher layers can react without inspecting generic exceptions.
How extend: define new subclasses here (e.g. `class AudioEmbeddingError(EmbeddingError): ...`) and raise them from the relevant modules.
"""

class S3DownloadError(Exception):
    """Custom exception for S3 download errors."""
    pass

class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass

class VectorDBError(Exception):
    """Custom exception for vector database errors."""
    pass