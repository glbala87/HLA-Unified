"""Seed chain structures for graph-based alignment.

ProtoSeeds collect initial BAM alignments per read name,
which are then converted to proper seed chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProtoSeed:
    """Proto-seed: collects BAM alignments for a single read/pair.

    Translates protoSeeds from C++. Groups BAM alignments by read name
    before conversion to graph coordinates.
    """
    read1_alignments: list[tuple[str, int, Any, int]] = field(default_factory=list)
    read2_alignments: list[tuple[str, int, Any, int]] = field(default_factory=list)

    def take_alignment(self, contig: str, position: int,
                       alignment: Any, read_number: int) -> None:
        """Add an alignment record (read_number: 1 or 2)."""
        entry = (contig, position, alignment, read_number)
        if read_number == 1:
            self.read1_alignments.append(entry)
        else:
            self.read2_alignments.append(entry)

    def is_complete(self) -> bool:
        """Check if we have alignments for both reads of a pair."""
        return bool(self.read1_alignments) and bool(self.read2_alignments)


@dataclass
class SeedChain:
    """A chain of seeds linking read positions to graph levels."""
    sequence_begin: int = -1
    sequence_end: int = -1
    graph_level_begin: int = -1
    graph_level_end: int = -1
    edge_ids: list[int] = field(default_factory=list)
    is_reverse: bool = False
