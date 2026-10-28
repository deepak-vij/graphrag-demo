"""The Knowledge Graph half of the dual-store architecture (Step 3).

Uses NetworkX as an in-process graph -- no server, easy to inspect, and
plenty for a demo of this size. Swap for Neo4j/Memgraph without touching
the online query logic, since everything downstream only needs
`.nodes`, `.edges`, and simple neighbor lookups.
"""
from __future__ import annotations

import networkx as nx


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for t in triples:
        graph.add_node(t["subject"], type=t.get("subject_type", "") or "")
        graph.add_node(t["object"], type=t.get("object_type", "") or "")
        graph.add_edge(t["subject"], t["object"], relation=t["relation"])
    return graph


def node_representation(graph: nx.MultiDiGraph, node: str) -> str:
    """Step 3: 'concatenate node labels with their attributes' -- a node's
    embeddable text is its name, type, and the facts it participates in."""
    node_type = graph.nodes[node].get("type") or "entity"
    facts = [f"{node} {d['relation']} {v}" for _, v, d in graph.out_edges(node, data=True)]
    facts += [f"{u} {d['relation']} {node}" for u, _, d in graph.in_edges(node, data=True)]
    facts_str = "; ".join(facts) if facts else "no recorded relations yet"
    return f"{node} (type: {node_type}). Known facts: {facts_str}"


def edge_representation(u: str, v: str, relation: str) -> str:
    """Step 3: 'translate directional edges into descriptive strings'."""
    return f"{u} --[{relation}]--> {v}"


def save_graph(graph: nx.MultiDiGraph, path) -> None:
    nx.write_graphml(graph, path)


def load_graph(path) -> nx.MultiDiGraph:
    return nx.read_graphml(path, force_multigraph=True)
