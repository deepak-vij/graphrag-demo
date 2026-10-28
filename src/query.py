"""ONLINE PIPELINE: Query Execution & Traversal (Steps 4-9).

Run with:  python -m src.query "Why did revenue drop during the incident?"
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from .embedder import embed
from .graph_store import load_graph
from .llm_client import generate_answer
from .vector_store import SimpleVectorStore


def load_store():
    graph = load_graph(config.GRAPH_PATH)
    store = SimpleVectorStore.load(config.VECTORS_PATH)
    node_meta = json.loads(config.NODE_META_PATH.read_text())
    return graph, store, node_meta


def semantic_anchors(question: str, store: SimpleVectorStore, node_meta: dict, k: int) -> list[str]:
    """Steps 4-6: embed the prompt, do nearest-neighbor lookup, resolve
    whatever comes back (a node OR an edge) into anchor node names."""
    q_vec = embed([question])[0]
    hits = store.search(q_vec, k=k)

    anchors: list[str] = []
    print("  nearest neighbors:")
    for item_id, score in hits:
        print(f"    {score:.3f}  {item_id}")
        if item_id.startswith("node::"):
            anchors.append(item_id.removeprefix("node::"))
        else:
            edge = node_meta["edges"][item_id]
            anchors.extend([edge["u"], edge["v"]])
    return list(dict.fromkeys(anchors))  # dedupe, keep order


def traverse(graph, anchors: list[str], max_hops: int) -> list[tuple[str, str, str, int]]:
    """Step 7: deterministic multi-hop graph walk (BFS) from the semantic
    anchors, in both directions, collecting every edge touched."""
    visited_nodes = set(anchors)
    frontier = set(anchors)
    collected: list[tuple[str, str, str, int]] = []

    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for node in frontier:
            if node not in graph:
                continue
            for _, v, data in graph.out_edges(node, data=True):
                collected.append((node, data["relation"], v, hop))
                if v not in visited_nodes:
                    next_frontier.add(v)
            for u, _, data in graph.in_edges(node, data=True):
                collected.append((u, data["relation"], node, hop))
                if u not in visited_nodes:
                    next_frontier.add(u)
        visited_nodes |= next_frontier
        frontier = next_frontier
        if not frontier:
            break
    return collected


def compile_context(edges: list[tuple[str, str, str, int]], max_triples: int) -> str:
    """Step 8: serialize the causal chain as compressed triples, closest
    hops first, capped at a hard triple budget. This is the 'boundary
    control' that keeps an unrestrained multi-hop query from bloating the
    prompt window."""
    seen = set()
    deduped = []
    for u, rel, v, hop in sorted(edges, key=lambda e: e[3]):
        key = (u, rel, v)
        if key not in seen:
            seen.add(key)
            deduped.append((u, rel, v, hop))

    truncated = deduped[:max_triples]
    lines = [f"({u}) -[{rel}]-> ({v})  [hop {hop}]" for u, rel, v, hop in truncated]

    dropped = len(deduped) - len(truncated)
    if dropped > 0:
        lines.append(f"... {dropped} additional triples truncated by MAX_CONTEXT_TRIPLES")
    return "\n".join(lines)


def run(question: str, top_k: int = None, max_hops: int = None, max_triples: int = None, model: str = None) -> str:
    top_k = top_k or config.TOP_K_ANCHORS
    max_hops = max_hops or config.MAX_HOPS
    max_triples = max_triples or config.MAX_CONTEXT_TRIPLES

    timings = {}
    graph, store, node_meta = load_store()

    print(f"\nQuestion: {question}\n")

    print("Steps 4-6: semantic anchoring")
    t0 = time.perf_counter()
    anchors = semantic_anchors(question, store, node_meta, top_k)
    timings["anchor_lookup"] = time.perf_counter() - t0
    print(f"  anchors -> {anchors}")

    print("\nStep 7: topological traversal")
    t0 = time.perf_counter()
    edges = traverse(graph, anchors, max_hops)
    timings["graph_traversal"] = time.perf_counter() - t0
    print(f"  visited {len(edges)} edge-hops across {max_hops} hop(s)")

    print("\nStep 8: context compilation")
    context = compile_context(edges, max_triples)
    print(context)

    print(f"\nStep 9: grounded generation ({model or config.ANTHROPIC_MODEL})")
    t0 = time.perf_counter()
    answer = generate_answer(question, context, model=model)
    timings["generation"] = time.perf_counter() - t0

    print("\n--- Answer (static GraphRAG) ---")
    print(answer)

    print("\n--- Latency budget (the tax this architecture accepts) ---")
    for step, dt in timings.items():
        print(f"  {step}: {dt * 1000:.0f} ms")
    print(f"  total: {sum(timings.values()) * 1000:.0f} ms")

    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Static GraphRAG: fixed 2-hop walk, one LLM call.")
    parser.add_argument("question", nargs="?", default="Why did revenue drop during the incident, and who fixed it?")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--model", default=None, help=f"Generator LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()
    run(args.question, args.top_k, args.max_hops, args.max_triples, args.model)


if __name__ == "__main__":
    main()
