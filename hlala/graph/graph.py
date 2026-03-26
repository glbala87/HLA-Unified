"""Population Reference Graph (PRG) for HLA typing.

This is the core data structure: a directed acyclic graph (DAG) where
nodes represent positions (levels) and edges represent allelic variants.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .node import Node, NodeHaplotype
from .edge import Edge, EMISSION_ENCODE

logger = logging.getLogger(__name__)


class PRGGraph:
    """Population Reference Graph.

    A DAG with integer-indexed nodes and edges.
    Nodes are organized by level (genomic position).
    """

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self.nodes_per_level: list[list[int]] = []  # level -> list of node IDs
        self._next_node_id = 0
        self._next_edge_id = 0
        self.filename_last_read: str = ""

        # Gap edge path data
        self.completed_gap_edge_paths: list[list[int]] = []  # list of edge ID lists
        self.pseudo_nodes: set[int] = set()
        self.pseudo_edges_to_gap_paths: dict[int, int] = {}  # edge_id -> gap_path_index
        self.gap_paths_forward: dict[int, dict[int, int]] = {}  # node_id -> {node_id -> edge_id}
        self.gap_paths_backward: dict[int, dict[int, int]] = {}
        self._have_computed_gap_edge_paths = False

    def register_node(self, level: int, terminal: bool = False,
                      node_id: int | None = None) -> Node:
        """Create and register a new node at the given level."""
        if node_id is None:
            node_id = self._next_node_id
            self._next_node_id += 1
        else:
            self._next_node_id = max(self._next_node_id, node_id + 1)

        node = Node(id=node_id, level=level, terminal=terminal)
        self.nodes[node_id] = node

        while len(self.nodes_per_level) <= level:
            self.nodes_per_level.append([])
        self.nodes_per_level[level].append(node_id)

        return node

    def register_edge(self, from_node: int, to_node: int,
                      emission: str = "_", locus_id: str = "",
                      label: str = "", count: float = 0.0,
                      pgf_protect: bool = False,
                      is_graph_gap: bool = False,
                      edge_id: int | None = None) -> Edge:
        """Create and register a new edge between two nodes."""
        if edge_id is None:
            edge_id = self._next_edge_id
            self._next_edge_id += 1
        else:
            self._next_edge_id = max(self._next_edge_id, edge_id + 1)

        emission_code = EMISSION_ENCODE.get(emission, 4)
        edge = Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            count=count,
            emission=emission_code,
            locus_id=locus_id,
            label=label,
            pgf_protect=pgf_protect,
            is_graph_gap=is_graph_gap,
        )
        self.edges[edge_id] = edge
        self.nodes[from_node].outgoing_edges.append(edge_id)
        self.nodes[to_node].incoming_edges.append(edge_id)
        return edge

    def unregister_node(self, node_id: int) -> None:
        """Remove a node from the graph."""
        node = self.nodes[node_id]
        self.nodes_per_level[node.level].remove(node_id)
        del self.nodes[node_id]

    def unregister_edge(self, edge_id: int) -> None:
        """Remove an edge from the graph."""
        edge = self.edges[edge_id]
        self.nodes[edge.from_node].outgoing_edges.remove(edge_id)
        self.nodes[edge.to_node].incoming_edges.remove(edge_id)
        del self.edges[edge_id]

    @property
    def num_levels(self) -> int:
        return len(self.nodes_per_level)

    def get_edges_from_level(self, level: int) -> list[Edge]:
        """Get all edges emanating from nodes at the given level."""
        result = []
        for node_id in self.nodes_per_level[level]:
            for edge_id in self.nodes[node_id].outgoing_edges:
                result.append(self.edges[edge_id])
        return result

    def get_one_locus_id_for_level(self, level: int) -> str:
        """Return one locus ID for edges at this level."""
        for node_id in self.nodes_per_level[level]:
            for edge_id in self.nodes[node_id].outgoing_edges:
                lid = self.edges[edge_id].locus_id
                if lid:
                    return lid
        return ""

    def get_assigned_loci(self) -> list[str]:
        """Return all unique locus IDs in order of first occurrence."""
        seen: set[str] = set()
        loci: list[str] = []
        for level in range(self.num_levels):
            lid = self.get_one_locus_id_for_level(level)
            if lid and lid not in seen:
                seen.add(lid)
                loci.append(lid)
        return loci

    def get_level_info(self) -> list[dict]:
        """Return node/edge/symbol counts per level."""
        info = []
        for level_idx, node_ids in enumerate(self.nodes_per_level):
            edges = self.get_edges_from_level(level_idx)
            symbols = {e.get_emission() for e in edges}
            info.append({
                "nodes": len(node_ids),
                "edges": len(edges),
                "symbols": len(symbols),
            })
        return info

    def check_consistency(self, terminal_check: bool = True) -> None:
        """Verify graph structural integrity."""
        for node_id, node in self.nodes.items():
            assert node_id in self.nodes_per_level[node.level], \
                f"Node {node_id} not in nodes_per_level[{node.level}]"
            for edge_id in node.outgoing_edges:
                edge = self.edges[edge_id]
                assert edge.from_node == node_id
            for edge_id in node.incoming_edges:
                edge = self.edges[edge_id]
                assert edge.to_node == node_id

        for edge_id, edge in self.edges.items():
            assert edge.from_node in self.nodes
            assert edge.to_node in self.nodes
            from_level = self.nodes[edge.from_node].level
            to_level = self.nodes[edge.to_node].level
            assert to_level == from_level + 1 or edge.is_graph_gap, \
                f"Edge {edge_id}: from level {from_level} to {to_level}"

    def check_sequence_presence(self, sequence: str, starting_at: int = 0,
                                verbose: bool = False) -> bool:
        """Check if a sequence can be traced through the graph from a given level."""
        if not sequence:
            return True
        if starting_at >= self.num_levels:
            return False

        current_nodes = set(self.nodes_per_level[starting_at])
        for i, char in enumerate(sequence):
            next_nodes: set[int] = set()
            for node_id in current_nodes:
                for edge_id in self.nodes[node_id].outgoing_edges:
                    edge = self.edges[edge_id]
                    if edge.get_emission() == char:
                        next_nodes.add(edge.to_node)
            if not next_nodes:
                if verbose:
                    logger.warning(f"Sequence breaks at position {i}, char '{char}'")
                return False
            current_nodes = next_nodes
        return True

    def trim_graph(self) -> int:
        """Remove edges with zero count. Return number of edges removed."""
        to_remove = [eid for eid, e in self.edges.items() if e.count == 0]
        for eid in to_remove:
            self.unregister_edge(eid)
        # Remove orphan nodes
        orphan_nodes = [nid for nid, n in self.nodes.items()
                        if not n.incoming_edges and not n.outgoing_edges
                        and n.level > 0]
        for nid in orphan_nodes:
            self.unregister_node(nid)
        return len(to_remove)

    @staticmethod
    def read_graph_loci(graph_dir: str | Path) -> list[str]:
        """Read the list of loci from a graph directory's segments file."""
        segments_file = Path(graph_dir) / "segments.txt"
        if not segments_file.exists():
            return []
        with open(segments_file) as fh:
            return [line.strip() for line in fh if line.strip()]

    def build_from_haplotypes(self, haplotype_ids: list[str],
                              haplotype_sequences: dict[str, str],
                              locus_ids: list[str],
                              want_pgf_protection: bool = True,
                              suffix_length: int = 0) -> None:
        """Build the graph from a panel of haplotype sequences.

        Each haplotype is a string of nucleotides aligned to the same positions.
        The graph represents all observed variation at each position.
        """
        if not haplotype_ids:
            return

        seq_length = len(next(iter(haplotype_sequences.values())))

        # Create nodes: one entry node per level, plus terminal
        for level in range(seq_length + 1):
            self.register_node(level=level, terminal=(level == seq_length))

        # For each position, create edges for observed alleles
        for level in range(seq_length):
            alleles_at_level: dict[str, list[str]] = defaultdict(list)
            for hap_id in haplotype_ids:
                seq = haplotype_sequences[hap_id]
                char = seq[level]
                alleles_at_level[char].append(hap_id)

            locus = locus_ids[level] if level < len(locus_ids) else ""
            from_nodes = self.nodes_per_level[level]
            to_nodes = self.nodes_per_level[level + 1]

            for allele_char, hap_list in alleles_at_level.items():
                for from_nid in from_nodes:
                    for to_nid in to_nodes:
                        self.register_edge(
                            from_node=from_nid,
                            to_node=to_nid,
                            emission=allele_char,
                            locus_id=locus,
                            count=len(hap_list),
                        )
