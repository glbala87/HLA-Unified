"""Exon position: alignment position within an HLA exon.

Translates hla::oneExonPosition from C++.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OneExonPosition:
    """A single position within an HLA exon alignment.

    Captures the genotype call at a specific exon position from a read,
    along with quality metrics for the read and its pair.
    """
    position_in_exon: int = 0
    graph_level: int = -1
    genotype: str = ""
    alignment_edge_labels: str = ""
    qualities: str = ""

    this_read_id: str = ""
    this_read_fraction_ok: float = 0.0
    this_read_weighted_chars_ok: float = 0.0

    paired_read_id: str = ""
    paired_read_fraction_ok: float = 0.0
    paired_read_weighted_chars_ok: float = 0.0

    read1_id: str = ""

    pairs_strands_ok: bool = False
    pairs_strands_distance: float = 0.0

    map_q: float = 0.0
    map_q_genomic: float = 0.0
    map_q_position: float = 0.0

    alignment_columns_with_nonGap: int = 0
    running_novel_gap_either_direction: int = 0

    reverse: bool = False
    from_first_read: bool = True
