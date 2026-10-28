"""Baseline for comparison: standard flat-text vector RAG, no graph.

Embeds each raw log entry as its own chunk and retrieves the top-k most
similar chunks to the question, then answers from that raw text. Run this
side by side with `query.py` on a multi-hop question (e.g. "why did revenue
drop?") to see the failure mode described at the top of the brief: the
query is semantically closest to the Revenue Dashboard chunk, so the root
cause several hops away (the Auth DB failure) can get left out.

Run with:  python -m src.naive_vector_rag "Why did revenue drop during the incident?"
"""
from __future__ import annotations

import argparse

from . import config
from .embedder import embed
from .ingest import chunk_corpus
from .llm_client import generate_naive_answer
from .vector_store import SimpleVectorStore

TOP_K_CHUNKS = 2


def run(question: str, top_k: int = TOP_K_CHUNKS, model: str = None) -> str:
    raw_text = config.DATA_FILE.read_text()
    chunks = chunk_corpus(raw_text)

    print(f"Embedding {len(chunks)} raw chunks (no graph, no entity resolution)...")
    vectors = embed(chunks)
    store = SimpleVectorStore()
    store.add([f"chunk::{i}" for i in range(len(chunks))], vectors)

    q_vec = embed([question])[0]
    hits = store.search(q_vec, k=top_k)

    print(f"\nQuestion: {question}\n")
    print("Top-k retrieved chunks (flat vector similarity, no traversal):")
    retrieved = []
    for item_id, score in hits:
        idx = int(item_id.split("::")[1])
        print(f"  {score:.3f}  chunk {idx}")
        retrieved.append(chunks[idx])

    context = "\n\n".join(retrieved)
    answer = generate_naive_answer(question, context, model=model)

    print("\n--- Answer (naive vector RAG) ---")
    print(answer)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline flat vector-RAG for comparison.")
    parser.add_argument("question", nargs="?", default="Why did revenue drop during the incident, and who fixed it?")
    parser.add_argument("--top-k", type=int, default=TOP_K_CHUNKS)
    parser.add_argument("--model", default=None, help=f"Generator LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()
    run(args.question, args.top_k, args.model)


if __name__ == "__main__":
    main()
