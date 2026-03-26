"""Needleman-Wunsch alignment on a graph DAG.

Translates VirtualNWTable_Unique / NWPath / NWEdge from C++.
Implements banded Needleman-Wunsch dynamic programming where the
graph dimension is a DAG (not a linear sequence).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Constants for matrix indices in affine gap model
MATRIX_MATCH = 0
MATRIX_GAP_IN_GRAPH = 1
MATRIX_GAP_IN_SEQUENCE = 2


@dataclass
class LocalExtensionPath:
    """Result of a local extension alignment."""
    coordinates: list[tuple[int, int, int]] = field(default_factory=list)  # (x, y, z) tuples
    used_edges: list[int] = field(default_factory=list)  # edge IDs
    score: float = float('-inf')
    aligned_sequence: str = ""
    aligned_graph: str = ""
    aligned_graph_levels: list[int] = field(default_factory=list)


@dataclass
class NWEdge:
    """An edge in the NW alignment graph.

    Connects two (x, y, z) coordinates where:
    - x: graph level
    - y: sequence position
    - z: node z-index within level
    """
    from_x: int = -1
    from_y: int = -1
    from_z: int = -1
    to_x: int = -1
    to_y: int = -1
    to_z: int = -1
    used_graph_edge: int = -1  # edge ID, -1 for gap
    score_after_edge: float = float('-inf')
    score_computed: bool = False
    ends_free_previous_affine_seq_gap: bool = False

    def is_sequence_gap(self) -> bool:
        """True if this edge consumes graph but not sequence."""
        return self.to_y == self.from_y

    def is_graph_gap(self) -> bool:
        """True if this edge consumes sequence but not graph."""
        return self.to_x == self.from_x


class VirtualNWTable:
    """Virtual Needleman-Wunsch DP table for graph alignment.

    Manages the sparse DP table and alignment paths through the
    3D space of (graph_level, sequence_position, node_z_index).
    """

    def __init__(self) -> None:
        self.edges: list[NWEdge] = []
        self.edges_from: dict[tuple[int, int, int], list[int]] = {}  # coord -> edge indices
        self.edges_to: dict[tuple[int, int, int], list[int]] = {}

    def add_edge(self, edge: NWEdge) -> int:
        idx = len(self.edges)
        self.edges.append(edge)
        from_key = (edge.from_x, edge.from_y, edge.from_z)
        to_key = (edge.to_x, edge.to_y, edge.to_z)
        self.edges_from.setdefault(from_key, []).append(idx)
        self.edges_to.setdefault(to_key, []).append(idx)
        return idx

    def get_edges_from(self, x: int, y: int, z: int) -> list[NWEdge]:
        return [self.edges[i] for i in self.edges_from.get((x, y, z), [])]

    def get_edges_to(self, x: int, y: int, z: int) -> list[NWEdge]:
        return [self.edges[i] for i in self.edges_to.get((x, y, z), [])]

    def get_entry_edges(self) -> list[NWEdge]:
        """Return edges with no predecessors."""
        all_to_coords = set(self.edges_to.keys())
        all_from_coords = set(self.edges_from.keys())
        entry_coords = all_from_coords - {(e.to_x, e.to_y, e.to_z) for e in self.edges}
        # Actually, entry edges are those whose from coord has no incoming
        result = []
        for edge in self.edges:
            key = (edge.from_x, edge.from_y, edge.from_z)
            if key not in self.edges_to:
                result.append(edge)
        return result

    def clear(self) -> None:
        self.edges.clear()
        self.edges_from.clear()
        self.edges_to.clear()
