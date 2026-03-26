"""Read data structures for HLA-LA alignment.

Translates oneRead, oneReadPair, and verboseSeedChain from C++.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.seq import reverse_complement


@dataclass
class OneRead:
    """A single sequencing read with name, sequence, and quality string."""
    name: str = ""
    sequence: str = ""
    quality: str = ""

    def __post_init__(self) -> None:
        if self.sequence and self.quality:
            assert len(self.sequence) == len(self.quality), \
                f"Sequence length {len(self.sequence)} != quality length {len(self.quality)}"

    def invert(self) -> None:
        """Reverse complement the read (in-place)."""
        self.sequence = reverse_complement(self.sequence)
        self.quality = self.quality[::-1]


@dataclass
class OneReadPair:
    """A paired-end read pair."""
    read1: OneRead = field(default_factory=OneRead)
    read2: OneRead = field(default_factory=OneRead)

    def invert(self) -> None:
        """Swap and invert both reads."""
        self.read1, self.read2 = self.read2, self.read1
        self.read1.invert()
        self.read2.invert()


@dataclass
class VerboseSeedChain:
    """A detailed alignment of a read to the graph.

    Translates verboseSeedChain from C++. Stores the full alignment path
    through the graph including edges traversed, levels, and aligned sequences.
    """
    sequence_begin: int = 0
    sequence_end: int = 0
    reverse: bool = False
    from_first_read: bool = True
    removed_columns_no_gap_restriction: int = 0
    improvement_through_bt: float = 0.0
    read_id: str = ""

    is_from_bwa_seed: list[bool] = field(default_factory=list)
    graph_aligned_edges: list[int] = field(default_factory=list)  # edge IDs
    graph_aligned_levels: list[int] = field(default_factory=list)
    graph_aligned: str = ""
    sequence_aligned: str = ""

    map_q: float = 0.0
    map_q_per_position: str = ""

    def extend_with_other(self, other: VerboseSeedChain, left: bool) -> None:
        """Extend this chain with another chain on the left or right side."""
        if left:
            self.graph_aligned_edges = other.graph_aligned_edges + self.graph_aligned_edges
            self.graph_aligned_levels = other.graph_aligned_levels + self.graph_aligned_levels
            self.graph_aligned = other.graph_aligned + self.graph_aligned
            self.sequence_aligned = other.sequence_aligned + self.sequence_aligned
            self.is_from_bwa_seed = other.is_from_bwa_seed + self.is_from_bwa_seed
            self.sequence_begin = other.sequence_begin
        else:
            self.graph_aligned_edges.extend(other.graph_aligned_edges)
            self.graph_aligned_levels.extend(other.graph_aligned_levels)
            self.graph_aligned += other.graph_aligned
            self.sequence_aligned += other.sequence_aligned
            self.is_from_bwa_seed.extend(other.is_from_bwa_seed)
            self.sequence_end = other.sequence_end

    def quality(self) -> tuple[int, int]:
        """Return (alignment_length, num_matches)."""
        assert len(self.graph_aligned) == len(self.sequence_aligned)
        length = len(self.graph_aligned)
        matches = sum(1 for g, s in zip(self.graph_aligned, self.sequence_aligned) if g == s)
        return length, matches

    def alignment_first_level(self) -> int:
        """Return the first non-gap graph level in the alignment."""
        for level in self.graph_aligned_levels:
            if level != -1:
                return level
        return -1

    def alignment_last_level(self) -> int:
        """Return the last non-gap graph level in the alignment."""
        for level in reversed(self.graph_aligned_levels):
            if level != -1:
                return level
        return -1

    def alignment_first_levels(self, n_max: int) -> list[int]:
        """Return up to n_max first non-gap levels."""
        result: list[int] = []
        for level in self.graph_aligned_levels:
            if level != -1:
                result.append(level)
                if len(result) >= n_max:
                    break
        return result

    def alignment_last_levels(self, n_max: int) -> list[int]:
        """Return up to n_max last non-gap levels."""
        result: list[int] = []
        for level in reversed(self.graph_aligned_levels):
            if level != -1:
                result.append(level)
                if len(result) >= n_max:
                    break
        return result

    def get_segments(self, level_names: list[str]) -> set[str]:
        """Get the set of graph segments/loci this alignment covers."""
        segments: set[str] = set()
        for level in self.graph_aligned_levels:
            if 0 <= level < len(level_names):
                seg = level_names[level]
                if seg:
                    segments.add(seg)
        return segments

    def check_level_contiguity(self) -> None:
        """Assert that graph levels increase monotonically without gaps."""
        assert len(self.graph_aligned) == len(self.graph_aligned_levels)
        assert len(self.sequence_aligned) == len(self.graph_aligned_levels)
        last_level = -1
        for level in self.graph_aligned_levels:
            if level != -1:
                if last_level != -1:
                    assert level == last_level + 1, \
                        f"Level contiguity error: {last_level} -> {level}"
                last_level = level


@dataclass
class VerboseSeedChainPair:
    """A pair of aligned seed chains for paired-end reads."""
    read_id: str = ""
    chain1: VerboseSeedChain = field(default_factory=VerboseSeedChain)
    chain2: VerboseSeedChain = field(default_factory=VerboseSeedChain)
    map_q: float = -1.0
