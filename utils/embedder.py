"""
utils/embedder.py
─────────────────
Singleton wrapper around the Tier-L encoder so every module
shares one loaded model in RAM.
"""

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_CACHE: SentenceTransformer | None = None


# ─────────────────────────────────────────────────────────────
def _get_model() -> SentenceTransformer:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "encoder"
        _MODEL_CACHE = SentenceTransformer(str(model_path))
    return _MODEL_CACHE


# ─────────────────────────────────────────────────────────────
def embed(text: str) -> np.ndarray:
    """Embed a single sentence → 768-d numpy vector (L2-normalised)."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a list of sentences → (N, 768) array."""
    model = _get_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
