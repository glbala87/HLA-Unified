"""Edge in the population reference graph."""

from __future__ import annotations

from dataclasses import dataclass

# Emission encoding: same as C++ (A=0, C=1, G=2, T=3, _=4/gap)
EMISSION_DECODE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "_", 5: "N"}
EMISSION_ENCODE = {"A": 0, "C": 1, "G": 2, "T": 3, "_": 4, "N": 5}


@dataclass
class Edge:
    """An edge in the PRG DAG.

    Each edge connects two nodes and carries:
    - emission: the nucleotide (or gap) this edge represents
    - locus_id: which HLA locus/gene region this edge belongs to
    - label: allele label for this edge
    """
    id: int
    from_node: int  # node ID
    to_node: int    # node ID
    count: float = 0.0
    emission: int = 4  # encoded nucleotide, default gap
    locus_id: str = ""
    label: str = ""
    pgf_protect: bool = False
    is_graph_gap: bool = False

    def get_emission(self) -> str:
        """Return the nucleotide string for this edge's emission."""
        return EMISSION_DECODE.get(self.emission, "?")

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.id == other.id
