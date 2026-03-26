"""Read-allele alignment matrix construction.

Builds a binary/weighted matrix M[read, allele] indicating how well
each read aligns to each candidate allele. This matrix is the input
to both the ILP genotyper and the Bayesian confidence estimator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam

from ..reference.loci import parse_allele_name

logger = logging.getLogger(__name__)


@dataclass
class ReadAlleleMatrix:
    """Binary/weighted read-allele alignment matrix.

    Attributes:
        matrix: shape (n_reads, n_alleles), values are alignment scores
        read_names: list of read names (length n_reads)
        allele_names: list of allele names (length n_alleles)
        binary: shape (n_reads, n_alleles), 1 if read maps to allele
    """
    matrix: np.ndarray
    read_names: list[str]
    allele_names: list[str]

    @property
    def n_reads(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_alleles(self) -> int:
        return self.matrix.shape[1]

    @property
    def binary(self) -> np.ndarray:
        return (self.matrix > 0).astype(np.float64)

    def subset_locus(self, locus: str) -> ReadAlleleMatrix:
        """Extract sub-matrix for a single locus."""
        cols = []
        col_names = []
        for i, name in enumerate(self.allele_names):
            info = parse_allele_name(name)
            if info.locus == locus:
                cols.append(i)
                col_names.append(name)

        if not cols:
            return ReadAlleleMatrix(
                matrix=np.zeros((self.n_reads, 0)),
                read_names=self.read_names,
                allele_names=[],
            )

        sub_matrix = self.matrix[:, cols]
        # Remove reads with no alignment to this locus
        row_mask = sub_matrix.sum(axis=1) > 0
        return ReadAlleleMatrix(
            matrix=sub_matrix[row_mask],
            read_names=[self.read_names[i] for i, m in enumerate(row_mask) if m],
            allele_names=col_names,
        )


def build_matrix_from_bam(
    bam_path: str | Path,
    candidate_alleles: list[str],
    min_mapq: int = 0,
    min_alignment_score: int = 0,
) -> ReadAlleleMatrix:
    """Build read-allele matrix from a BAM file.

    Each cell contains the alignment score (AS tag) of that read
    to that allele. Zero means no alignment.
    """
    allele_to_idx = {a: i for i, a in enumerate(candidate_alleles)}
    read_scores: dict[str, dict[int, float]] = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary:
                continue
            if read.mapping_quality < min_mapq:
                continue

            ref = read.reference_name
            if ref not in allele_to_idx:
                continue

            try:
                score = float(read.get_tag("AS"))
            except KeyError:
                score = float(read.mapping_quality)

            if score < min_alignment_score:
                continue

            rname = read.query_name
            col = allele_to_idx[ref]

            if rname not in read_scores:
                read_scores[rname] = {}
            # Keep best score per read-allele pair
            read_scores[rname][col] = max(
                read_scores[rname].get(col, 0), score,
            )

    # Also collect supplementary/secondary alignments for multi-mapping
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or not read.is_secondary:
                continue
            ref = read.reference_name
            if ref not in allele_to_idx:
                continue
            try:
                score = float(read.get_tag("AS"))
            except KeyError:
                continue

            rname = read.query_name
            col = allele_to_idx[ref]
            if rname in read_scores:
                read_scores[rname][col] = max(
                    read_scores[rname].get(col, 0), score,
                )

    # Build dense matrix
    read_names = sorted(read_scores.keys())
    read_to_row = {r: i for i, r in enumerate(read_names)}
    n_reads = len(read_names)
    n_alleles = len(candidate_alleles)

    matrix = np.zeros((n_reads, n_alleles), dtype=np.float64)
    for rname, scores in read_scores.items():
        row = read_to_row[rname]
        for col, score in scores.items():
            matrix[row, col] = score

    # Normalize alignment scores per-row (per-read) to [0, 1]
    # This removes aligner-specific score scale bias: minimap2, bowtie2,
    # and BWA all use different scoring schemes. Row-wise normalization
    # ensures that a read's relative affinity across alleles is preserved
    # regardless of which aligner produced the scores.
    if n_reads > 0:
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0  # avoid division by zero
        matrix = matrix / row_max

    logger.info(
        "Read-allele matrix: %d reads x %d alleles, %.1f%% non-zero (scores normalized)",
        n_reads, n_alleles,
        100.0 * np.count_nonzero(matrix) / max(matrix.size, 1),
    )
    return ReadAlleleMatrix(
        matrix=matrix, read_names=read_names, allele_names=candidate_alleles,
    )
