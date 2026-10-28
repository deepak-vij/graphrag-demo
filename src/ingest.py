"""OFFLINE PIPELINE: Graph Construction & Embedding (Steps 1-3).

Run with:  python -m src.ingest
           python -m src.ingest --model claude-opus-5
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from . import config
from .embedder import embed
from .entity_resolution import resolve_entities
from .graph_store import build_graph, edge_representation, node_representation, save_graph
from .llm_client import extract_triples
from .vector_store import SimpleVectorStore


def chunk_corpus(text: str) -> list[str]:
    """Split the raw log file into per-entry chunks. This stands in for
    whatever document-loading step a real pipeline would use -- the point
    of GraphRAG is that we do NOT keep these chunks as the retrieval unit,
    we only use them as input to the extractor."""
    entries = re.split(r"(?=^=== Log Entry)", text, flags=re.MULTILINE)
    return [e.strip() for e in entries if e.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline GraphRAG ingestion pipeline.")
    parser.add_argument("--model", default=None, help=f"Extractor LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()

    config.STORE_DIR.mkdir(exist_ok=True)

    print(f"Step 1-2: Extractor LLM ({args.model or config.ANTHROPIC_MODEL}) -> entities + relationships")
    raw_text = config.DATA_FILE.read_text()
    chunks = chunk_corpus(raw_text)
    print(f"  {len(chunks)} log entries to process")

    all_triples: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        t0 = time.perf_counter()
        triples = extract_triples(chunk, model=args.model)
        dt = time.perf_counter() - t0
        print(f"  [{i}/{len(chunks)}] extracted {len(triples)} triples ({dt:.2f}s)")
        all_triples.extend(triples)
    print(f"  total raw triples: {len(all_triples)}")

    print("\nStep 1-2 (cont.): Named Entity Resolution -- deduping aliases")
    resolved_triples, merges = resolve_entities(all_triples)
    if merges:
        for alias, canonical in sorted(merges.items()):
            print(f"  merged '{alias}' -> '{canonical}'")
    else:
        print("  no aliases needed merging")

    print("\nStep 3: building the Knowledge Graph")
    graph = build_graph(resolved_triples)
    print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    save_graph(graph, config.GRAPH_PATH)

    print("\nStep 3: embedding graph topology (nodes + edges) into the Vector DB")
    node_ids, node_texts = [], []
    for node in graph.nodes:
        node_ids.append(f"node::{node}")
        node_texts.append(node_representation(graph, node))

    edge_ids, edge_texts, edge_meta = [], [], {}
    for u, v, data in graph.edges(data=True):
        eid = f"edge::{u}|||{data['relation']}|||{v}"
        edge_ids.append(eid)
        edge_texts.append(edge_representation(u, v, data["relation"]))
        edge_meta[eid] = {"u": u, "v": v, "relation": data["relation"]}

    all_ids = node_ids + edge_ids
    all_texts = node_texts + edge_texts
    vectors = embed(all_texts)

    store = SimpleVectorStore()
    store.add(all_ids, vectors)
    store.save(config.VECTORS_PATH)

    node_meta = {
        "nodes": node_ids,
        "edges": edge_meta,
        "node_texts": dict(zip(node_ids, node_texts)),
        "edge_texts": dict(zip(edge_ids, edge_texts)),
    }
    config.NODE_META_PATH.write_text(json.dumps(node_meta, indent=2))
    config.RAW_CHUNKS_PATH.write_text(json.dumps(resolved_triples, indent=2))

    print(f"  embedded {len(all_ids)} items ({len(node_ids)} nodes, {len(edge_ids)} edges)")
    print(f"\nOffline pipeline complete. Store written to {config.STORE_DIR}/")


if __name__ == "__main__":
    main()
