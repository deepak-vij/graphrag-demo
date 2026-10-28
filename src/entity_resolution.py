"""Named Entity Resolution: merge aliases like "Auth DB" / "Authentication
Database" / "Auth Database" into one canonical node before they ever reach
the graph. Skipping this step is exactly what causes graph fragmentation.

Approach: embed every distinct entity name with the same open-source
embedding model used for the graph's vector index, then union-find any
pair above a similarity threshold. The canonical label for a cluster is
its longest member, on the theory that the fuller name is more
descriptive (matches the example in the brief).

Names with numbers in them (timestamps, hostnames, percentages) only merge
when their digits match exactly. Embeddings see "03:04 UTC" and "04:04 UTC"
as near-identical strings, but one digit is the whole meaning.

Common abbreviations are expanded before embedding ("Payments Svc" ->
"Payments Service"), since the embedding model doesn't know "Svc" means
"Service". The canonical label still comes from the original names.
"""
from __future__ import annotations

import re

from .embedder import embed

SIMILARITY_THRESHOLD = 0.80
ABBREVIATIONS = {"svc": "service", "db": "database"}


def _digits(name: str) -> str:
    return "".join(re.findall(r"\d", name))


def _expand(name: str) -> str:
    return re.sub(r"[A-Za-z]+", lambda m: ABBREVIATIONS.get(m.group().lower(), m.group()), name)


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def resolve_entities(triples: list[dict]) -> tuple[list[dict], dict[str, str]]:
    names = sorted({t["subject"] for t in triples} | {t["object"] for t in triples})
    if not names:
        return triples, {}

    vectors = embed([_expand(n) for n in names])
    uf = _UnionFind(names)

    sims = vectors @ vectors.T
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sims[i, j] >= SIMILARITY_THRESHOLD and _digits(names[i]) == _digits(names[j]):
                uf.union(names[i], names[j])

    clusters: dict[str, list[str]] = {}
    for name in names:
        clusters.setdefault(uf.find(name), []).append(name)

    canonical_map: dict[str, str] = {}
    for members in clusters.values():
        canonical = max(members, key=len)
        for member in members:
            canonical_map[member] = canonical

    resolved = [
        {
            **t,
            "subject": canonical_map[t["subject"]],
            "object": canonical_map[t["object"]],
        }
        for t in triples
    ]
    merges = {k: v for k, v in canonical_map.items() if k != v}
    return resolved, merges
