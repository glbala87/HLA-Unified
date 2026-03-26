"""Base aligner: scoring parameters and graph traversal helpers.

Translates mapper::aligner::alignerBase from C++.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph.graph import PRGGraph
    from ..mapper.reads import VerboseSeedChain


class AlignerBase:
    """Base class for graph-based alignment.

    Provides scoring parameters and helper methods for navigating
    the ordered node/edge structure of the graph.
    """

    def __init__(self, graph: PRGGraph) -> None:
        self.graph = graph

        # Scoring parameters (match C++ defaults)
        self.S_match = 1.0
        self.S_mismatch = -3.0
        self.S_gap = -5.0          # simple gap penalty
        self.S_open_gap = -5.0     # affine gap open
        self.S_extend_gap = -2.0   # affine gap extend
        self.S_graph_gap = -5.0    # gap in graph

        # Precompute ordered node lists per level
        self.nodes_per_level_ordered: list[list[int]] = []
        self.nodes_per_level_ordered_rev: list[dict[int, int]] = []
        self._build_ordered_nodes()

    def _build_ordered_nodes(self) -> None:
        """Build ordered node lists and reverse lookup for each level."""
        self.nodes_per_level_ordered = []
        self.nodes_per_level_ordered_rev = []
        for level_nodes in self.graph.nodes_per_level:
            ordered = sorted(level_nodes)
            rev = {nid: idx for idx, nid in enumerate(ordered)}
            self.nodes_per_level_ordered.append(ordered)
            self.nodes_per_level_ordered_rev.append(rev)

    def get_previous_z_values_and_edges(self, x: int, z: int) -> list[tuple[int, int]]:
        """Get (z_index, edge_id) pairs for nodes at level x-1 that connect to node z at level x.

        Args:
            x: current graph level
            z: index of current node within its level (z-index)
        """
        if x <= 0 or x >= len(self.nodes_per_level_ordered):
            return []

        current_node_id = self.nodes_per_level_ordered[x][z]
        node = self.graph.nodes[current_node_id]
        results = []
        for edge_id in node.incoming_edges:
            edge = self.graph.edges[edge_id]
            from_node_id = edge.from_node
            from_level = self.graph.nodes[from_node_id].level
            if from_level == x - 1:
                z_prev = self.nodes_per_level_ordered_rev[x - 1].get(from_node_id, -1)
                if z_prev >= 0:
                    results.append((z_prev, edge_id))
        return results

    def get_next_z_values_and_edges(self, x: int, z: int) -> list[tuple[int, int]]:
        """Get (z_index, edge_id) pairs for nodes at level x+1 reachable from node z at level x."""
        if x < 0 or x >= len(self.nodes_per_level_ordered) - 1:
            return []

        current_node_id = self.nodes_per_level_ordered[x][z]
        node = self.graph.nodes[current_node_id]
        results = []
        for edge_id in node.outgoing_edges:
            edge = self.graph.edges[edge_id]
            to_node_id = edge.to_node
            to_level = self.graph.nodes[to_node_id].level
            if to_level == x + 1:
                z_next = self.nodes_per_level_ordered_rev[x + 1].get(to_node_id, -1)
                if z_next >= 0:
                    results.append((z_next, edge_id))
        return results

    @staticmethod
    def strands_valid(chain1: VerboseSeedChain, chain2: VerboseSeedChain) -> bool:
        """Check if paired read strands are in valid FR orientation."""
        return chain1.reverse != chain2.reverse

    @staticmethod
    def pair_distance_in_graph_levels(chain1: VerboseSeedChain,
                                      chain2: VerboseSeedChain) -> int:
        """Compute graph-level distance between paired chains."""
        l1 = chain1.alignment_last_level()
        l2 = chain2.alignment_first_level()
        if l1 == -1 or l2 == -1:
            return -1
        return abs(l2 - l1)
