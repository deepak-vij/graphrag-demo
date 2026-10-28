"""A deliberately minimal vector index: no server, no ANN index structure,
just a matrix and a dot product. This is the "Vector DB" half of the
dual-store architecture -- swap this module for FAISS/Chroma/pgvector
without touching anything else, since the interface is just
add()/search()."""
from __future__ import annotations

import json

import numpy as np


class SimpleVectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.ids.extend(ids)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Probabilistic nearest-neighbor lookup (Steps 4-6): cosine similarity
        since both query and stored vectors are L2-normalized at embed time."""
        sims = self.vectors @ query_vector
        top_idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top_idx]

    def save(self, path) -> None:
        np.savez(path, ids=np.array(self.ids, dtype=object), vectors=self.vectors)

    @classmethod
    def load(cls, path) -> "SimpleVectorStore":
        data = np.load(path, allow_pickle=True)
        store = cls()
        store.ids = list(data["ids"])
        store.vectors = data["vectors"]
        return store
