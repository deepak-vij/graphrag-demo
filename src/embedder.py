"""Wraps a local, open-source embedding model.

This is deliberately separate from the Anthropic client: Anthropic does not
serve an embeddings endpoint, and mixing "the LLM that extracts/generates"
with "the model that embeds" is exactly the dual-store split the GraphRAG
architecture calls for (Extractor/Generator LLM vs. embedding model).
"""
from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        from .config import EMBEDDING_MODEL_NAME

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings into an (N, D) float32 matrix, L2-normalized
    so that cosine similarity reduces to a dot product."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")
