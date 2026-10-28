"""The 'Extractor LLM' (Steps 1-2) and the 'Generator LLM' (Step 9).

Both roles are filled by the same Claude model here, but they are called
with very different jobs: one turns unstructured text into structured
(subject, relation, object) triples; the other turns a compiled bundle of
those triples back into a natural-language, grounded answer.
"""
from __future__ import annotations

import json

import anthropic

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

_client = None

_EXTRACT_TOOL = {
    "name": "record_triples",
    "description": "Record the entities and relationships found in the text as knowledge-graph triples.",
    "input_schema": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Canonical-ish entity name, e.g. 'Authentication Database'"},
                        "subject_type": {"type": "string", "description": "e.g. Service, Database, Person, Dashboard, Incident"},
                        "relation": {"type": "string", "description": "Short verb phrase, e.g. 'depends_on', 'caused', 'restored_by'"},
                        "object": {"type": "string"},
                        "object_type": {"type": "string"},
                    },
                    "required": ["subject", "relation", "object"],
                },
            }
        },
        "required": ["triples"],
    },
}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the demo, e.g.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def extract_triples(chunk_text: str, model: str | None = None) -> list[dict]:
    """Step 1 & 2: unstructured text -> entities (nodes) + relationships (edges).

    We force the model to call a tool rather than free-generate JSON, so the
    output is always well-formed and directly usable downstream."""
    client = _get_client()
    resp = client.messages.create(
        model=model or ANTHROPIC_MODEL,
        max_tokens=4096,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_triples"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract knowledge-graph triples from this incident log excerpt. "
                    "Use short, consistent relation names (snake_case verbs). "
                    "Prefer the fullest form of an entity name when the text gives one "
                    "(e.g. keep 'Authentication Database' rather than abbreviating it), "
                    "but don't invent facts not stated in the text. "
                    "Attach each time, number or measurement only to the entity the text "
                    "gives it to: in 'Checkout failed at 03:09 because Login was failing', "
                    "03:09 belongs to Checkout, not Login.\n\n"
                    f"---\n{chunk_text}\n---"
                ),
            }
        ],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "record_triples":
            return block.input.get("triples", [])
    return []


def generate_naive_answer(question: str, raw_chunks: str, model: str | None = None) -> str:
    """The baseline this demo contrasts against: a plain vector-RAG
    generation call over raw retrieved text chunks, no graph involved."""
    client = _get_client()
    resp = client.messages.create(
        model=model or ANTHROPIC_MODEL,
        max_tokens=16000,
        system=(
            "Answer the question using only the retrieved log excerpts below. "
            "If they don't contain enough information, say so."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Retrieved log excerpts:\n{raw_chunks}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def generate_answer(question: str, context_block: str, model: str | None = None) -> str:
    """Step 9: grounded generation, constrained to the compiled graph context."""
    client = _get_client()
    resp = client.messages.create(
        model=model or ANTHROPIC_MODEL,
        max_tokens=16000,
        system=(
            "You answer strictly from the provided knowledge-graph triples. "
            "Each triple is a verified fact: subject -[relation]-> object. "
            "Trace the causal/dependency chain explicitly in your answer. "
            "If the triples don't contain enough information to answer, say so "
            "instead of guessing."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Knowledge graph context:\n{context_block}\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
