"""Alignment statistics collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentStats:
    """Statistics for a set of alignments."""
    total_reads: int = 0
    mapped_reads: int = 0
    paired_reads: int = 0
    unpaired_reads: int = 0
    mean_alignment_length: float = 0.0
    mean_match_fraction: float = 0.0

    def update(self, alignment_length: int, matches: int) -> None:
        n = self.total_reads
        self.total_reads += 1
        if alignment_length > 0:
            self.mapped_reads += 1
            frac = matches / alignment_length
            self.mean_alignment_length = (self.mean_alignment_length * n + alignment_length) / (n + 1)
            self.mean_match_fraction = (self.mean_match_fraction * n + frac) / (n + 1)

    def summary(self) -> str:
        return (
            f"Total reads: {self.total_reads}, "
            f"Mapped: {self.mapped_reads} ({self.mapped_reads / max(1, self.total_reads):.1%}), "
            f"Mean alignment length: {self.mean_alignment_length:.1f}, "
            f"Mean match fraction: {self.mean_match_fraction:.3f}"
        )
