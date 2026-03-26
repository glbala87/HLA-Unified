"""Graph serialization using msgpack.

The original C++ uses Boost serialization which produces binary files
incompatible with Python. This module provides msgpack-based save/load
for the Python graph representation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import msgpack

from .graph import PRGGraph
from .node import NodeHaplotype

logger = logging.getLogger(__name__)


def save_graph(graph: PRGGraph, filepath: str | Path) -> None:
    """Serialize a PRGGraph to a msgpack file."""
    nodes_data = []
    for nid in sorted(graph.nodes.keys()):
        node = graph.nodes[nid]
        haplos = [(h.haplo_id, h.probability, h.index_in_haplotype_vector)
                  for h in node.haplotypes]
        nodes_data.append({
            "id": nid,
            "level": node.level,
            "terminal": node.terminal,
            "haplotypes": haplos,
        })

    edges_data = []
    for eid in sorted(graph.edges.keys()):
        edge = graph.edges[eid]
        edges_data.append({
            "id": eid,
            "from": edge.from_node,
            "to": edge.to_node,
            "count": edge.count,
            "emission": edge.emission,
            "locus_id": edge.locus_id,
            "label": edge.label,
            "pgf_protect": edge.pgf_protect,
            "is_graph_gap": edge.is_graph_gap,
        })

    data = {
        "version": 2,
        "nodes": nodes_data,
        "edges": edges_data,
        "num_levels": graph.num_levels,
        "filename": graph.filename_last_read,
        "gap_paths": graph.completed_gap_edge_paths,
    }

    with open(filepath, "wb") as fh:
        msgpack.pack(data, fh, use_bin_type=True)
    logger.info(f"Saved graph to {filepath} ({len(nodes_data)} nodes, {len(edges_data)} edges)")


def load_graph(filepath: str | Path) -> PRGGraph:
    """Deserialize a PRGGraph from a msgpack file."""
    with open(filepath, "rb") as fh:
        data = msgpack.unpack(fh, raw=False)

    graph = PRGGraph()

    for nd in data["nodes"]:
        node = graph.register_node(level=nd["level"], terminal=nd["terminal"],
                                   node_id=nd["id"])
        for h in nd.get("haplotypes", []):
            node.haplotypes.append(NodeHaplotype(
                haplo_id=h[0], probability=h[1],
                index_in_haplotype_vector=h[2],
            ))

    for ed in data["edges"]:
        from .edge import EMISSION_DECODE
        emission_char = EMISSION_DECODE.get(ed["emission"], "_")
        graph.register_edge(
            from_node=ed["from"],
            to_node=ed["to"],
            emission=emission_char,
            locus_id=ed.get("locus_id", ""),
            label=ed.get("label", ""),
            count=ed.get("count", 0.0),
            pgf_protect=ed.get("pgf_protect", False),
            is_graph_gap=ed.get("is_graph_gap", False),
            edge_id=ed["id"],
        )

    graph.completed_gap_edge_paths = data.get("gap_paths", [])
    graph.filename_last_read = data.get("filename", "")

    logger.info(f"Loaded graph from {filepath}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {graph.num_levels} levels")
    return graph
