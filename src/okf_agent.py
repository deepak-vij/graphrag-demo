"""OKF agent: answer from the Open Knowledge Format bundle, with no graph,
no embeddings and no vector search.

Same loop as dynamic_graphrag.py, but the knowledge is markdown files and
the "edges" are the links between them. Claude gets two tools:

  read_index()          -> the bundle's index.md: every concept, one line each
  read_concept(path)    -> one concept document, frontmatter and body

and follows links from concept to concept until it can answer.

Build the bundle first:  python -m src.build_okf
Run with:  python -m src.okf_agent "Why did revenue drop during the incident, and who fixed it?"
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from .llm_client import _get_client

MAX_STEPS = 8

_TOOLS = [
    {
        "name": "read_index",
        "description": (
            "Read the bundle's index.md: a list of every concept document, grouped by type, "
            "with its path and a one-line description. Start here."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_concept",
        "description": (
            "Read one concept document: YAML frontmatter (type, title, description, sources) "
            "and a markdown body whose links point to related concepts. Pass the link's "
            "bundle-relative path exactly, e.g. '/services/login-service.md'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Bundle-relative path of a concept document."},
            },
            "required": ["path"],
        },
    },
]

_SYSTEM = (
    "You answer questions about an incident using only an Open Knowledge Format bundle: a "
    "folder of markdown concept documents that link to each other. Read the index first, then "
    "read concepts and follow their links until you can answer every part of the question. "
    "Trace the chain explicitly in your answer. Every link in that chain must be stated in a "
    "document you read: if two steps aren't connected yet, read the concept that would connect "
    "them instead of bridging the gap yourself. The same goes for every other claim: state only "
    "what a document you read says about that concept (don't extend a fact about one service to "
    "others), and cite only documents you read. If the bundle doesn't contain the answer, say so "
    "instead of guessing."
)


def _read_concept(path: str) -> tuple[str, bool]:
    target = (config.OKF_DIR / path.strip().lstrip("/")).resolve()
    if not target.is_relative_to(config.OKF_DIR.resolve()) or target.suffix != ".md":
        return f"'{path}' is not a concept document in this bundle.", True
    if not target.is_file():
        return f"No document at '{path}'. Use read_index to find the right path.", True
    return target.read_text(), False


def run(question: str, max_steps: int = MAX_STEPS, model: str = None) -> str:
    if not (config.OKF_DIR / "index.md").is_file():
        raise SystemExit(f"No OKF bundle at {config.OKF_DIR}/. Run `python -m src.build_okf` first.")
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
            if block.name == "read_index":
                output, is_error = _read_concept("/index.md")
            elif block.name == "read_concept":
                output, is_error = _read_concept(block.input["path"])
            else:
                output, is_error = f"Unknown tool '{block.name}'.", True
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        answer = f"[gave up after {max_steps} steps without a final answer]"

    print("\n--- Answer (OKF agent) ---")
    print(answer)
    print(f"\n  total: {(time.perf_counter() - t0) * 1000:.0f} ms over {step} LLM call(s)")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="OKF agent: Claude reads an Open Knowledge Format bundle.")
    parser.add_argument("question", nargs="?", default="Why did revenue drop during the incident, and who fixed it?")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--model", default=None, help=f"Agent LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()
    run(args.question, args.max_steps, args.model)


if __name__ == "__main__":
    main()
