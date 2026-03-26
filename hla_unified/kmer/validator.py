"""K-mer-based validation (HLAforest / HLA-LA cross-check).

Provides an alignment-free orthogonal check on the genotype calls.
If k-mer-based and alignment-based calls disagree, the locus is
flagged for manual review.

Strategy:
1. Extract k-mer profiles from the called allele pair
2. Extract k-mer profiles from the reads
3. Check: are the expected allele k-mers covered by the reads?
4. Check: are there unexpected k-mers suggesting a different allele?
5. Compute chi-square uniformity test on k-mer coverage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..utils.seq import extract_canonical_kmers, extract_kmers, canonical_kmer

logger = logging.getLogger(__name__)


@dataclass
class KmerValidationResult:
    """Result of k-mer validation for one locus."""
    locus: str
    allele1: str
    allele2: str
    proportion_kmers_covered: float  # fraction of allele k-mers seen in reads
    kmer_coverage_uniformity: float  # chi-square p-value (high = uniform)
    mean_kmer_depth: float
    unexpected_kmer_fraction: float  # fraction of read k-mers not in alleles
    is_concordant: bool  # True if k-mer evidence supports the call
    flags: list[str]


class KmerValidator:
    """Alignment-free k-mer validation of HLA genotype calls."""

    def __init__(
        self,
        k: int = 31,
        min_coverage_fraction: float = 0.90,
        max_unexpected_fraction: float = 0.30,
        uniformity_alpha: float = 0.01,
    ) -> None:
        self.k = k
        self.min_coverage_fraction = min_coverage_fraction
        self.max_unexpected_fraction = max_unexpected_fraction
        self.uniformity_alpha = uniformity_alpha

    def validate(
        self,
        allele1_seq: str,
        allele2_seq: str,
        read_sequences: list[str],
        locus: str = "",
        allele1_name: str = "",
        allele2_name: str = "",
    ) -> KmerValidationResult:
        """Validate a diploid call using k-mer evidence.

        Args:
            allele1_seq, allele2_seq: Called allele sequences
            read_sequences: All reads mapping to this locus
            locus: Locus name for reporting
        """
        flags = []

        # Extract k-mers from called alleles
        allele_kmers = (
            extract_canonical_kmers(allele1_seq, self.k)
            | extract_canonical_kmers(allele2_seq, self.k)
        )

        if not allele_kmers:
            return KmerValidationResult(
                locus=locus, allele1=allele1_name, allele2=allele2_name,
                proportion_kmers_covered=0, kmer_coverage_uniformity=0,
                mean_kmer_depth=0, unexpected_kmer_fraction=1.0,
                is_concordant=False, flags=["no_allele_kmers"],
            )

        # Count k-mers in reads
        read_kmer_counts: dict[str, int] = {}
        for seq in read_sequences:
            for km in extract_kmers(seq, self.k):
                ck = canonical_kmer(km)
                read_kmer_counts[ck] = read_kmer_counts.get(ck, 0) + 1

        # Proportion of allele k-mers covered
        covered = sum(1 for km in allele_kmers if km in read_kmer_counts)
        prop_covered = covered / len(allele_kmers) if allele_kmers else 0.0

        if prop_covered < self.min_coverage_fraction:
            flags.append(f"low_kmer_coverage:{prop_covered:.2f}")

        # Mean depth of covered allele k-mers
        depths = [
            read_kmer_counts.get(km, 0) for km in allele_kmers
        ]
        mean_depth = float(np.mean(depths)) if depths else 0.0

        # Coverage uniformity (chi-square test)
        nonzero_depths = [d for d in depths if d > 0]
        if len(nonzero_depths) > 5:
            expected = np.mean(nonzero_depths)
            chi2_stat, p_value = stats.chisquare(
                nonzero_depths,
                f_exp=[expected] * len(nonzero_depths),
            )
            uniformity = float(p_value)
        else:
            uniformity = 0.0
            flags.append("too_few_kmers_for_uniformity_test")

        # Unexpected k-mers: read k-mers not in either allele
        total_read_kmers = len(read_kmer_counts)
        unexpected = sum(
            1 for km in read_kmer_counts if km not in allele_kmers
        )
        unexpected_frac = unexpected / max(total_read_kmers, 1)

        if unexpected_frac > self.max_unexpected_fraction:
            flags.append(f"high_unexpected_kmers:{unexpected_frac:.2f}")

        # Overall concordance
        is_concordant = (
            prop_covered >= self.min_coverage_fraction
            and unexpected_frac <= self.max_unexpected_fraction
        )

        return KmerValidationResult(
            locus=locus,
            allele1=allele1_name,
            allele2=allele2_name,
            proportion_kmers_covered=prop_covered,
            kmer_coverage_uniformity=uniformity,
            mean_kmer_depth=mean_depth,
            unexpected_kmer_fraction=unexpected_frac,
            is_concordant=is_concordant,
            flags=flags,
        )

    def validate_all_loci(
        self,
        calls: dict[str, tuple[str, str]],  # locus -> (allele1, allele2)
        allele_sequences: dict[str, str],     # allele_name -> sequence
        locus_reads: dict[str, list[str]],    # locus -> read sequences
    ) -> dict[str, KmerValidationResult]:
        """Validate calls for all loci."""
        results = {}
        for locus, (a1, a2) in calls.items():
            seq1 = allele_sequences.get(a1, "")
            seq2 = allele_sequences.get(a2, "")
            reads = locus_reads.get(locus, [])

            if not seq1 and not seq2:
                logger.warning("No sequences for %s calls: %s, %s", locus, a1, a2)
                continue

            results[locus] = self.validate(
                seq1, seq2, reads,
                locus=locus, allele1_name=a1, allele2_name=a2,
            )
        return results
