# GraphRAG Demo (minimal, pure Python)

A runnable, end-to-end implementation of the two-phase GraphRAG architecture:
offline graph construction + embedding, then online semantic anchoring +
graph traversal + grounded generation. Everything runs in-process (NetworkX
for the graph, numpy for the vector index) so you can read every line of the
pipeline without a database server in the way. The only external calls are
to Claude (as the Extractor LLM and the Generator LLM) and to a local,
open-source embedding model (`sentence-transformers/all-MiniLM-L6-v2`).

The architecture follows Vishal Dalmiya's
[LinkedIn post on GraphRAG system architecture](https://www.linkedin.com/posts/vishaldalmiya_graphrag-systemarchitecture-generativeai-share-7505483330897809408-FBJy/).

## Setup

```bash
cd graphrag_demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

The first run will download the ~90MB embedding model from Hugging Face.

## Run it

```bash
# Offline pipeline: read data/incident_logs.txt -> extract triples ->
# resolve entity aliases -> build the graph -> embed it -> save to store/
python -m src.ingest

# Static GraphRAG (online pipeline): fixed 2-hop walk, one LLM call
python -m src.query "Why did revenue drop during the incident, and who fixed it?"

# Compare against plain vector RAG over the same raw text (no graph)
python -m src.naive_vector_rag "Why did revenue drop during the incident, and who fixed it?"

# Dynamic GraphRAG: Claude explores the same graph with search_graph /
# get_neighbors tools in a loop instead of a fixed 2-hop walk
python -m src.dynamic_graphrag "Why did revenue drop during the incident, and who fixed it?"
```

Try other questions too, e.g. `"What does Checkout Service depend on?"` or
`"Was Search Index Service affected by the outage?"` (it wasn't — a good
check that the graph doesn't hallucinate a connection).

## The dataset

`data/incident_logs.txt` is a fabricated microservices incident: an
Authentication Database disk failure cascades through Login Service ->
Checkout Service -> Payment Service -> Revenue Dashboard, four hops away
from the root cause. It also names the same entities inconsistently on
purpose — "Auth DB" / "Authentication Database" / "Auth Database", and
"Payments Svc" / "Payment Service" — to exercise entity resolution. A last,
unrelated log entry (Search Index Service) checks that traversal doesn't
wander into unconnected parts of the graph.

This is exactly the shape standard vector RAG struggles with: the answer to
"why did revenue drop" requires chaining facts from five different log
entries, and the query text is semantically closest to the Revenue
Dashboard entry, not the Auth DB entry where the root cause actually lives.

## Results

On the revenue question, scored against what the logs actually say (one run
each with `claude-sonnet-5`; wording, tool-call counts and timings will vary
a little from run to run):

| | Plain vector RAG | Static GraphRAG | Dynamic GraphRAG |
|---|---|---|---|
| Why did revenue drop? (Auth DB disk failure) | Not found; guessed "upstream, e.g. Checkout Service" | Found, via the postmortem's blast-radius list | Found, edge by edge: Auth DB -> Login -> Checkout -> Payment Service -> orders |
| Who fixed it? (Priya Nair, failover to db-auth-04) | Not found | Not found; only saw that she opened the postmortem | Found |
| Invented facts? | No | No | No |
| Query-time LLM calls | 1 | 1 | 7 (15 tool calls) |
| Total time | — | 14.3 s | 24.8 s |

Plain vector RAG retrieves the Revenue Dashboard and Payments entries; the
postmortem ranks 7th of 9 and the root-cause entry never comes back. Static
GraphRAG reaches the root cause at hop 2 but not the fix, whose edge is 3
hops from the anchor. Dynamic GraphRAG decides how far to walk, so it gets
both, at the cost of more LLM calls.

## Where each step lives

**Offline pipeline**

| Step | What | File |
|---|---|---|
| 1-2 | Extractor LLM -> triples (Claude, forced tool-use for structured output) | `src/llm_client.py: extract_triples` |
| 1-2 | Named Entity Resolution (expand abbreviations like Svc/DB, embed entity names, union-find merge above a similarity threshold, names with numbers merge only if their digits match, canonical = longest alias) | `src/entity_resolution.py` |
| 3 | Build the Knowledge Graph (NetworkX `MultiDiGraph`) | `src/graph_store.py: build_graph` |
| 3 | Synthesize node/edge text for embedding ("node label + attributes", "edge -> descriptive string") | `src/graph_store.py: node_representation`, `edge_representation` |
| 3 | Embed + persist the dual store (graph as GraphML, vectors as `.npz`) | `src/ingest.py` |

**Online pipeline**

| Step | What | File |
|---|---|---|
| 4-6 | Embed the query, nearest-neighbor lookup for semantic anchors | `src/query.py: semantic_anchors` |
| 7 | Deterministic multi-hop BFS from the anchors | `src/query.py: traverse` |
| 8 | Compress + dedupe + truncate to a fixed triple budget (`MAX_CONTEXT_TRIPLES`) | `src/query.py: compile_context` |
| 9 | Grounded generation, constrained to the compiled triples | `src/llm_client.py: generate_answer` |
| 4-9 | Dynamic GraphRAG: Claude runs steps 4-9 itself with `search_graph` / `get_neighbors` tools in a loop (max 8 rounds) | `src/dynamic_graphrag.py` |

`src/query.py` also times each step and prints a latency breakdown at the
end, so you can see the "latency tax" from the trade-offs section directly:
graph traversal is a second round-trip on top of the vector lookup, and
generation still dominates total latency.

## Choosing a model

The default (`claude-sonnet-5`) is set in `src/config.py`. To use a different
model for a single run, without editing any code:

```bash
# per-command override
python -m src.ingest --model claude-opus-5
python -m src.query "..." --model claude-haiku-4-5
python -m src.naive_vector_rag "..." --model claude-haiku-4-5

# or set it for the whole shell session
export ANTHROPIC_MODEL=claude-opus-5
python -m src.query "..."
```

`--model` (when passed) always wins over `ANTHROPIC_MODEL`, which wins over
the `src/config.py` default. This only changes the Extractor/Generator LLM —
the embedding model is separate and fixed to the local
`sentence-transformers` model regardless of what you pick here.

## Knobs worth playing with

All in `src/config.py`:

- `TOP_K_ANCHORS` — how many semantic anchors seed the traversal. Too low and
  you might miss the right entry point; too high and irrelevant nodes creep
  into the graph walk.
- `MAX_HOPS` — traversal depth. Try dropping this to `1` on the revenue
  question and watch the answer lose the causal chain back to the Auth DB.
- `MAX_CONTEXT_TRIPLES` — the hard boundary control mentioned in the brief.
  Set it very low to see the model start saying "I don't have enough
  information."

## Swapping in real infrastructure

Nothing here is tied to in-process storage on purpose:

- `src/vector_store.py`'s `SimpleVectorStore` has exactly two methods
  (`add`, `search`) — replace its body with FAISS, Chroma, or pgvector calls
  and nothing else in the pipeline changes.
- `src/graph_store.py` only uses NetworkX's `.nodes`/`.edges`/neighbor
  iteration — replace with Cypher queries against Neo4j/Memgraph behind the
  same three functions (`build_graph`, `node_representation`,
  `edge_representation`) and `query.py` is unaffected.
