"""Build an Open Knowledge Format (OKF v0.2) bundle from the incident logs.

OKF is a different answer to the same problem the graph solves: instead of
triples plus a vector index, knowledge is a folder of markdown files, one
per concept, with YAML frontmatter. Concepts link to each other with plain
markdown links, and the prose around a link says what the relationship is.
Spec: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

Here Claude reads all nine log entries in one call and writes the concept
files (curation, not extraction: one file per real-world thing, so aliases
collapse naturally). The root index.md is generated from the frontmatter,
so it always matches the files.

Run with:  python -m src.build_okf
"""
from __future__ import annotations

import argparse
import re
import shutil
import time

import yaml

from . import config
from .ingest import chunk_corpus
from .llm_client import _get_client

_WRITE_TOOL = {
    "name": "write_bundle",
    "description": "Write the concept documents of an OKF bundle.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Bundle-relative path, e.g. '/services/login-service.md'"},
                        "content": {"type": "string", "description": "The whole file: YAML frontmatter between --- lines, then the markdown body."},
                    },
                    "required": ["path", "content"],
                },
            }
        },
        "required": ["concepts"],
    },
}

_PROMPT = """Turn these incident log entries into an Open Knowledge Format (OKF v0.2) bundle.

Rules:
- One concept document per real-world thing: each service, database, host, person, team,
  incident and postmortem. If the logs use several names for one thing ("Auth DB",
  "Authentication Database"), write one document and mention the aliases in it.
- Paths are bundle-relative, lowercase, grouped by kind: /services/login-service.md,
  /people/priya-nair.md, /incidents/pm-2216.md, and so on. Never use index.md or log.md.
- Frontmatter: `type` (e.g. Service, Database, Host, Person, Team, Incident, Postmortem),
  `title`, a one-line `description`, `tags`, and `sources`: one entry per log entry the
  document draws on, with `id: entry-N`, `resource: /references/incident_logs.txt` and
  `title` set to that entry's header line.
- Body: state every fact the logs give about the concept, in plain sentences. Whenever a
  sentence relates this concept to another one, link the other concept with a
  bundle-relative markdown link, e.g. "Depends on the [Authentication Database](/databases/authentication-database.md)
  for credential lookups." Write each relationship in both documents it connects.
- Cite each fact with a footnote keyed to its source id ([^entry-N]) and define the
  footnotes at the end of the body.
- Copy times, numbers and names exactly as the logs give them, and keep each time with the
  thing the log gives it to. Don't infer facts the logs don't state. The time in an entry's
  "=== Log Entry ... ===" header is when the entry was written, not when its events happened:
  never use it as the time of an event.
- Each identifier (such as a postmortem ID) belongs to exactly one document; don't invent a
  separate document for something the logs only mention through it. A link's text must name
  the concept its path points to.

Log entries:
{entries}"""


def _split(content: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", content, flags=re.DOTALL)
    if not match:
        return {}, content
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def _write_index(concepts: dict[str, dict]) -> str:
    by_type: dict[str, list[str]] = {}
    for path, meta in sorted(concepts.items()):
        by_type.setdefault(meta.get("type", "Other"), []).append(
            f"- [{meta.get('title', path)}]({path}): {meta.get('description', '')}".rstrip(": ")
        )
    sections = [f"## {kind}\n\n" + "\n".join(lines) for kind, lines in sorted(by_type.items())]
    return '---\nokf_version: "0.2"\n---\n\n# Incident knowledge bundle\n\n' + "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an OKF bundle from the incident logs.")
    parser.add_argument("--model", default=None, help=f"Curator LLM override (default: {config.ANTHROPIC_MODEL})")
    args = parser.parse_args()

    entries = chunk_corpus(config.DATA_FILE.read_text())
    numbered = "\n\n".join(f"[entry-{i}]\n{e}" for i, e in enumerate(entries, 1))

    print(f"Curating {len(entries)} log entries into an OKF bundle ({args.model or config.ANTHROPIC_MODEL})...")
    t0 = time.perf_counter()
    resp = _get_client().messages.create(
        model=args.model or config.ANTHROPIC_MODEL,
        max_tokens=16000,
        tools=[_WRITE_TOOL],
        tool_choice={"type": "tool", "name": "write_bundle"},
        messages=[{"role": "user", "content": _PROMPT.format(entries=numbered)}],
    )
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("The bundle was cut off at max_tokens; raise it and rerun.")
    files = next(b.input["concepts"] for b in resp.content if b.type == "tool_use")
    print(f"  {len(files)} concept documents ({time.perf_counter() - t0:.1f}s)")

    if config.OKF_DIR.exists():
        shutil.rmtree(config.OKF_DIR)
    concepts: dict[str, dict] = {}
    for f in files:
        path = "/" + f["path"].strip().lstrip("/")
        if ".." in path or not path.endswith(".md") or path.rsplit("/", 1)[-1] in ("index.md", "log.md"):
            print(f"  skipped invalid path {f['path']!r}")
            continue
        meta, _ = _split(f["content"])
        if not meta.get("type"):
            print(f"  warning: {path} has no `type` in its frontmatter")
        concepts[path] = meta
        target = config.OKF_DIR / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f["content"].rstrip() + "\n")

    (config.OKF_DIR / "index.md").write_text(_write_index(concepts))
    refs = config.OKF_DIR / "references"
    refs.mkdir(exist_ok=True)
    shutil.copy(config.DATA_FILE, refs / "incident_logs.txt")

    broken = sorted(
        {(path, link) for path in concepts
         for link in re.findall(r"\]\((/[^)#\s]+\.md)", (config.OKF_DIR / path.lstrip("/")).read_text())
         if link not in concepts}
    )
    for path, link in broken:
        print(f"  broken link in {path}: {link}")
    print(f"\nBundle written to {config.OKF_DIR}/ ({len(concepts)} concepts, {len(broken)} broken links)")


if __name__ == "__main__":
    main()
