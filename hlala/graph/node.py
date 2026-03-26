"""Node in the population reference graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .edge import Edge


@dataclass
class NodeHaplotype:
    """Haplotype information attached to a node."""
    haplo_id: int
    probability: float
    index_in_haplotype_vector: int


@dataclass
class Node:
    """A node in the PRG DAG.

    Each node represents a position (level) in the graph.
    Nodes at the same level represent alternative alleles.
    """
    id: int
    level: int
    terminal: bool = False
    haplotypes: list[NodeHaplotype] = field(default_factory=list)

    # These are populated after graph construction
    incoming_edges: list[int] = field(default_factory=list)  # edge IDs
    outgoing_edges: list[int] = field(default_factory=list)  # edge IDs

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id
