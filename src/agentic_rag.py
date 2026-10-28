"""Agentic GraphRAG: same graph and vector store as query.py, but Claude
decides what to retrieve.

query.py runs a fixed pipeline (2 anchors, 2 hops, 12 triples) and calls
Claude once. Here Claude gets two tools over the same dual store and calls
them in a loop until it has enough to answer:

  search_graph(query)   -> nearest nodes/edges in the vector index
  get_neighbors(node)   -> every edge touching one node in the graph

So if the first hop shows "PM-2216 opened_by Priya Nair", the model can
choose to look at Priya Nair's edges next, which the fixed 2-hop walk in
query.py never reaches.

Run with:  python -m src.agentic_rag "Why did revenue drop during the incident, and who fixed it?"
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from .embedder import embed
from .llm_client import _get_client
from .query import load_store

MAX_STEPS = 8

_TOOLS = [
    {
        "name": "search_graph",
        "description": (
            "Semantic search over the knowledge graph. Returns the nodes and edges whose text "
            "is closest to the query. Use it to find entry points, or to look up the exact "
            "name of an entity before calling get_neighbors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words."},
                "k": {"type": "integer", "description": "How many results to return (default 4)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_neighbors",
        "description": (
            "Return every edge touching one node, in both directions, as "
            "(subject) -[relation]-> (object) triples. The node name must match exactly; "
            "use search_graph first if unsure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "Exact node name, e.g. 'PM-2216'."},
            },
            "required": ["node"],
        },
    },
]

_SYSTEM = (
    "You answer questions about an incident using only a knowledge graph, which you explore "
    "with the search_graph and get_neighbors tools. Each triple is a verified fact: "
    "subject -[relation]-> object. Follow edges until you can answer every part of the "
    "question, then trace the chain explicitly in your answer. If the graph doesn't contain "
    "the answer, say so instead of guessing."
)


def _search_graph(store, node_meta, query: str, k: int = 4) -> str:
    q_vec = embed([query])[0]
    lines = []
    for item_id, score in store.search(q_vec, k=k):
        if item_id.startswith("node::"):
            lines.append(f"{score:.3f}  node: {item_id.removeprefix('node::')}")
        else:
            lines.append(f"{score:.3f}  edge: {node_meta['edge_texts'][item_id]}")
    return "\n".join(lines)


def _get_neighbors(graph, node: str) -> tuple[str, bool]:
    if node not in graph:
        return f"No node named '{node}'. Use search_graph to find the exact name.", True
    triples = [f"({node}) -[{d['relation']}]-> ({v})" for _, v, d in graph.out_edges(node, data=True)]
    triples += [f"({u}) -[{d['relation']}]-> ({node})" for u, _, d in graph.in_edges(node, data=True)]
    return "\n".join(dict.fromkeys(triples)) or f"'{node}' has no edges.", False


def run(question: str, max_steps: int = MAX_STEPS, model: str = None) -> str:
    graph, store, node_meta = load_store()
    client = _get_client()
    messages = [{"role": "user", "content": question}]

    print(f"\nQuestion: {question}\n")
    t0 = time.perf_counter()
    answer = ""
    for step in range(1, max_steps + 1):
        resp = client.messages.create(
            model=model or config.ANTHROPIC_MODEL,
            max_tokens=16000,
            system=_SYSTEM,
            tools=_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            if resp.stop_reason != "end_turn":
                answer += f"\n\n[stopped early: stop_reason={resp.stop_reason}]"
            break

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            print(f"  step {step}: {block.name}({json.dumps(block.input)})")
            if block.name == "search_graph":
                output, is_error = _search_graph(store, node_meta, block.input["query"], block.input.get("k", 4)), False
            elif block.name == "get_neighbors":
                output, is_error = _get_neighbors(graph, block.input["node"])
            else:
                output, is_error = f"Unknown tool '{block.name}'.", True
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        answer = f"[gave up after {max_steps} steps without a final answer]"

    print("\n--- Answer (agentic GraphRAG) ---")
    print(answer)
    print(f"\n  total: {(time.perf_counter() - t0) * 1000:.0f} ms over {step} LLM call(s)")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic GraphRAG: Claude explores the graph with tools.")
    parser.add_argument("question", nargs="?", default="Why did revenue drop during the incident, and who fixed it?")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--model", default=None, help=f"Agent LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()
    run(args.question, args.max_steps, args.model)


if __name__ == "__main__":
    main()
