"""Purpose: make concrete vector database managers importable from one namespace.
Why extend: register additional backends (Faiss, Pinecone, etc.) without changing consumers.
How extend: import new manager classes defined under `src.vectordb` and append them to `__all__`.
"""
from __future__ import annotations

from src.vectordb.base import VectorDBManager
from src.vectordb.lance import LanceDBManager

__all__ = ["VectorDBManager", "LanceDBManager"]
