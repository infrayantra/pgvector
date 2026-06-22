"""Embedding helpers — SentenceTransformers with deterministic fallback."""

from __future__ import annotations

import hashlib
import re

import numpy as np

EMBED_DIM = 384
_MODEL = None
_MODEL_AVAILABLE = None


def _has_sentence_transformers() -> bool:
    global _MODEL_AVAILABLE
    if _MODEL_AVAILABLE is None:
        try:
            import sentence_transformers  # noqa: F401
            _MODEL_AVAILABLE = True
        except ImportError:
            _MODEL_AVAILABLE = False
    return _MODEL_AVAILABLE


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model (all-MiniLM-L6-v2)... first run may download ~90MB")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic pseudo-embedding when SentenceTransformers is unavailable."""
    tokens = re.findall(r"\w+", text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        h = hashlib.sha256(token.encode()).digest()
        for i in range(0, min(len(h), dim), 4):
            idx = int.from_bytes(h[i : i + 4], "big") % dim
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed(texts: list[str], normalize: bool = True) -> list[np.ndarray]:
    if _has_sentence_transformers():
        model = get_model()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return [v.astype(np.float32) for v in vectors]
    return [_hash_embed(t) for t in texts]


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def embedding_backend() -> str:
    return "SentenceTransformers (all-MiniLM-L6-v2)" if _has_sentence_transformers() else "hash fallback (install sentence-transformers for real semantics)"
