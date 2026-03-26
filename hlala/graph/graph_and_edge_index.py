"""K-mer index over graph edges for fast seed finding.

Translates GraphAndEdgeIndex from C++. Builds a dict mapping k-mer strings
to lists of edge chains (paths) in the graph that spell that k-mer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .graph import PRGGraph
from .edge import EMISSION_DECODE

logger = logging.getLogger(__name__)


@dataclass
class KMerInGraphSpec:
    """A k-mer occurrence in the graph, specified by the edges traversed."""
    traversed_edges: list[int]  # edge IDs


@dataclass
class KMerEdgeChain:
    """A chain of k-mer matches along a sequence."""
    sequence_begin: int = -1
    sequence_end: int = -1
    traversed_edges: list[int] = field(default_factory=list)


class GraphAndEdgeIndex:
    """K-mer index over graph edges for O(1) seed lookup.

    Builds an index mapping k-mer strings to their occurrences (as edge paths)
    in the population reference graph. Used for finding seed alignments.
    """

    def __init__(self, graph: PRGGraph, k: int = 25) -> None:
        self.graph = graph
        self.k = k
        self.kmers: dict[str, list[KMerInGraphSpec]] = {}
        self.nodes_jump_over_gaps: dict[int, list[int]] = {}  # node_id -> [edge_ids]

    def index(self) -> None:
        """Build the k-mer index by enumerating all k-length paths through the graph."""
        logger.info(f"Building k-mer index with k={self.k}...")
        self.kmers.clear()

        num_levels = self.graph.num_levels
        if num_levels < self.k:
            logger.warning("Graph has fewer levels than k-mer size")
            return

        for start_level in range(num_levels - self.k):
            self._enumerate_kmers_from_level(start_level)

        logger.info(f"Indexed {len(self.kmers)} unique k-mers")

    def _enumerate_kmers_from_level(self, start_level: int) -> None:
        """Enumerate all k-mers starting from a given graph level."""
        # BFS/DFS through k levels collecting edge paths
        # Each state: (current_edges_list, current_kmer_chars, current_level)
        initial_states: list[tuple[list[int], list[str]]] = []

        for node_id in self.graph.nodes_per_level[start_level]:
            for edge_id in self.graph.nodes[node_id].outgoing_edges:
                edge = self.graph.edges[edge_id]
                if not edge.is_graph_gap:
                    char = edge.get_emission()
                    if char != "_":
                        initial_states.append(([edge_id], [char]))

        stack = initial_states
        while stack:
            edges_so_far, chars_so_far = stack.pop()
            if len(chars_so_far) == self.k:
                kmer = "".join(chars_so_far)
                spec = KMerInGraphSpec(traversed_edges=list(edges_so_far))
                if kmer not in self.kmers:
                    self.kmers[kmer] = []
                self.kmers[kmer].append(spec)
                continue

            last_edge = self.graph.edges[edges_so_far[-1]]
            to_node = last_edge.to_node
            for edge_id in self.graph.nodes[to_node].outgoing_edges:
                edge = self.graph.edges[edge_id]
                if not edge.is_graph_gap:
                    char = edge.get_emission()
                    if char != "_":
                        stack.append((edges_so_far + [edge_id], chars_so_far + [char]))

    def query_index(self, kmer: str) -> list[KMerInGraphSpec]:
        """Look up all occurrences of a k-mer in the graph."""
        return self.kmers.get(kmer, [])

    def find_chains(self, sequence: str) -> list[KMerEdgeChain]:
        """Find all k-mer seed chains for a query sequence.

        Scans the sequence for k-mers present in the index and returns
        their positions and graph edge paths.
        """
        chains: list[KMerEdgeChain] = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            hits = self.query_index(kmer)
            for hit in hits:
                chain = KMerEdgeChain(
                    sequence_begin=i,
                    sequence_end=i + self.k - 1,
                    traversed_edges=list(hit.traversed_edges),
                )
                chains.append(chain)
        return chains

    def fill_edge_jumper(self) -> None:
        """Build gap-jumping edge paths for nodes adjacent to gap edges."""
        self.nodes_jump_over_gaps.clear()
        for edge_id, edge in self.graph.edges.items():
            if edge.is_graph_gap:
                from_node = edge.from_node
                to_node = edge.to_node
                if from_node not in self.nodes_jump_over_gaps:
                    self.nodes_jump_over_gaps[from_node] = []
                # Collect non-gap edges from the destination node
                for next_edge_id in self.graph.nodes[to_node].outgoing_edges:
                    if not self.graph.edges[next_edge_id].is_graph_gap:
                        self.nodes_jump_over_gaps[from_node].append(next_edge_id)

    def get_indexed_kmers(self) -> list[str]:
        """Return all indexed k-mer strings."""
        return list(self.kmers.keys())
