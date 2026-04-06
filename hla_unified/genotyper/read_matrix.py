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
) -> ReadAlleleMatrix:
    """Build read-allele matrix from a BAM file.

    Each cell contains the alignment score (AS tag) of that read
    to that allele, transformed so higher = better alignment.

    Handles different aligner scoring conventions:
    - bowtie2: AS <= 0 (0 = perfect, negative = mismatches)
    - minimap2: AS >= 0 (higher = better)
    - BWA: AS >= 0 (higher = better)
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

            rname = read.query_name
            col = allele_to_idx[ref]

            if rname not in read_scores:
                read_scores[rname] = {}
            # Keep best (highest/least-negative) score per read-allele pair
            prev = read_scores[rname].get(col)
            if prev is None or score > prev:
                read_scores[rname][col] = score

    # Also collect secondary alignments for multi-mapping
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
                prev = read_scores[rname].get(col)
                if prev is None or score > prev:
                    read_scores[rname][col] = score

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
    # Handles both positive-is-better (minimap2/BWA) and
    # negative-is-penalty (bowtie2) scoring conventions.
    # Transform: shift each row so min=0, then scale so max=1.
    # A read that maps equally well to all alleles gets uniform scores;
    # a read with strong preference gets a sharp distribution.
    if n_reads > 0:
        row_min = matrix.min(axis=1, keepdims=True)
        matrix = matrix - row_min  # shift: worst score -> 0
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0  # avoid division by zero
        matrix = matrix / row_max  # scale: best score -> 1

        # For reads that only map to one allele, the shift+scale gives
        # a clear 1.0 for that allele and 0.0 for others. For reads
        # that map to multiple alleles, relative scores are preserved.

    logger.info(
        "Read-allele matrix: %d reads x %d alleles, %.1f%% non-zero (scores normalized)",
        n_reads, n_alleles,
        100.0 * np.count_nonzero(matrix) / max(matrix.size, 1),
    )
    return ReadAlleleMatrix(
        matrix=matrix, read_names=read_names, allele_names=candidate_alleles,
    )
