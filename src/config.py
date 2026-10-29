import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "incident_logs.txt"
STORE_DIR = ROOT_DIR / "store"

GRAPH_PATH = STORE_DIR / "graph.graphml"
VECTORS_PATH = STORE_DIR / "vectors.npz"
NODE_META_PATH = STORE_DIR / "node_meta.json"
RAW_CHUNKS_PATH = STORE_DIR / "raw_chunks.json"

# Open Knowledge Format bundle written by src/build_okf.py
OKF_DIR = ROOT_DIR / "okf_bundle"

# Extractor / Generator LLM (Anthropic). Override per-run with `--model` on
# any of the CLI scripts, or by exporting ANTHROPIC_MODEL before running.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Embedding model (open-source, runs locally via sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Online pipeline knobs (this is where the "boundary controls" mentioned
# in the trade-offs section actually get enforced)
TOP_K_ANCHORS = 2
MAX_HOPS = 2
MAX_CONTEXT_TRIPLES = 12
