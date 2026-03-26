"""K-mer coverage analysis for HLA allele validation.

Implements k-mer-based goodness-of-fit testing to validate
inferred HLA types against observed read data.
"""

from __future__ import annotations

import math
from collections import Counter

from ..utils.seq import partition_into_kmers, kmer_canonical


def compute_kmer_coverage(allele_sequence: str, read_sequences: list[str],
                          k: int = 31) -> dict[str, int]:
    """Count how many reads contain each k-mer from an allele sequence.

    Args:
        allele_sequence: the reference allele sequence
        read_sequences: list of read sequences to check
        k: k-mer size

    Returns:
        Dict mapping canonical k-mers to read count.
    """
    allele_kmers = set(kmer_canonical(km) for km in partition_into_kmers(allele_sequence, k))

    coverage: dict[str, int] = {km: 0 for km in allele_kmers}
    for read_seq in read_sequences:
        read_kmers = set(kmer_canonical(km) for km in partition_into_kmers(read_seq, k))
        for km in read_kmers & allele_kmers:
            coverage[km] += 1

    return coverage


def proportion_kmers_covered(coverage: dict[str, int]) -> float:
    """Fraction of allele k-mers observed at least once."""
    if not coverage:
        return 0.0
    covered = sum(1 for v in coverage.values() if v > 0)
    return covered / len(coverage)


def kmer_chi_square(observed_counts: list[float],
                    expected_counts: list[float]) -> float:
    """Compute chi-square statistic for k-mer coverage uniformity.

    Tests whether k-mer coverage is uniform (expected under correct typing).

    Matches simpleChiSq from HLATyper.cpp.
    """
    if len(observed_counts) != len(expected_counts):
        raise ValueError("Observed and expected must have same length")

    chi_sq = 0.0
    for obs, exp in zip(observed_counts, expected_counts):
        if exp > 0:
            chi_sq += (obs - exp) ** 2 / exp
    return chi_sq


def compute_coverage_stats(coverage_values: list[float]) -> dict[str, float]:
    """Compute coverage statistics: mean, first decile, minimum.

    These are reported in the output alongside HLA type calls.
    """
    if not coverage_values:
        return {"mean": 0.0, "first_decile": 0.0, "minimum": 0.0}

    sorted_vals = sorted(coverage_values)
    n = len(sorted_vals)
    decile_idx = max(0, int(n * 0.1))

    return {
        "mean": sum(sorted_vals) / n,
        "first_decile": sorted_vals[decile_idx],
        "minimum": sorted_vals[0],
    }
