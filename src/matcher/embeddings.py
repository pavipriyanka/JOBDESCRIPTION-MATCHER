"""Sentence embeddings. Any object with ``encode(list[str]) -> np.ndarray`` works,
which lets the tests use a lightweight offline encoder."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # slow import, loaded lazily

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class HashingEncoder:
    """Offline bag-of-words encoder (feature hashing). Used in tests; no model download."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9+#]+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of a and every row of b (inputs are L2-normalised)."""
    return a @ b.T
