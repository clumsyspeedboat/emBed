"""Purpose: share foundational utilities (device selection, normalisation) across embedder implementations.
Why extend: centralise additional numeric helpers as you introduce new modalities.
How extend: add functions here and call them from modality-specific modules to avoid repeated Tensor/Numpy glue.
"""
from __future__ import annotations

import os
import numpy as np

__all__ = ["pick_device", "l2_normalize"]


def pick_device(default: str = "auto") -> str:
    """Determine the desired torch device based on environment and availability."""
    import torch

    device_setting = os.getenv("DEVICE", default).lower()
    if device_setting == "cpu":
        return "cpu"
    if device_setting == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last dimension."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm
