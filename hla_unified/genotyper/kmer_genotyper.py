"""K-mer-based HLA genotyper with depth-based heterozygous detection.

Alignment-free genotyping using k-mer coverage scoring for initial pair
selection, followed by depth-based rescoring to distinguish heterozygous
from homozygous calls. The depth step examines k-mer counts at
allele-specific positions to detect the second allele.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..reference.loci import (
    parse_allele_name,
    truncate_to_resolution,
    group_alleles_by_resolution,
)
from ..reference.frequencies import AlleleFrequencyDatabase, load_default_frequencies
from ..utils.seq import canonical_kmer, extract_canonical_kmers

logger = logging.getLogger(__name__)


@dataclass
class KmerGenotypeResult:
    """Result of k-mer genotyping for one locus."""
    locus: str
    allele1: str
    allele2: str
    score: float
    confidence: float
    kmers_supporting_a1: int
    kmers_supporting_a2: int
    n_alleles_evaluated: int
    method: str = "kmer"
    is_homozygous: bool = False


class KmerGenotyper:
    """Alignment-free HLA genotyper using k-mer coverage + depth."""

    def __init__(
        self,
        k: int = 21,
        frequency_weight: float = 0.03,
        min_kmer_support: int = 3,
    ) -> None:
        self.k = k
        self.frequency_weight = frequency_weight
        self.min_kmer_support = min_kmer_support
        self.freq_db = load_default_frequencies()

    def genotype_locus(
        self,
        locus: str,
        allele_sequences: dict[str, str],
        read_kmers: set[str],
        candidate_alleles: list[str] | None = None,
        read_kmer_counts: dict[str, int] | None = None,
    ) -> KmerGenotypeResult:
        """Genotype a single locus using k-mer evidence."""
        # Restrict to candidates if specified
        if candidate_alleles:
            allele_sequences = {
                a: s for a, s in allele_sequences.items()
                if a in candidate_alleles
            }

        # Only apply frequency filtering when no prefilter candidates
        if not candidate_alleles:
            common_alleles = {}
            for allele, seq in allele_sequences.items():
                two_field = truncate_to_resolution(allele, 2)
                two_field_clean = two_field.removeprefix("HLA-")
                freq = self.freq_db.get_frequency(two_field_clean)
                if freq > 1e-5:
                    common_alleles[allele] = seq
            if common_alleles:
                allele_sequences = common_alleles

        if not allele_sequences:
            return KmerGenotypeResult(
                locus=locus, allele1="", allele2="",
                score=0.0, confidence=0.0,
                kmers_supporting_a1=0, kmers_supporting_a2=0,
                n_alleles_evaluated=0,
            )

        # Extract k-mers for each allele
        allele_kmers: dict[str, set[str]] = {}
        for allele, seq in allele_sequences.items():
            allele_kmers[allele] = extract_canonical_kmers(seq, self.k)

        # Score each allele by k-mer coverage
        allele_scores: dict[str, float] = {}
        allele_supports: dict[str, int] = {}
        for allele, kmers in allele_kmers.items():
            n_total = len(kmers)
            if n_total == 0:
                allele_scores[allele] = 0.0
                allele_supports[allele] = 0
                continue
            n_found = sum(1 for km in kmers if km in read_kmers)
            coverage = n_found / n_total
            # Score by COVERAGE FRACTION, not absolute count.
            # Absolute count biases toward longer sequences (genomic
            # sequences vary from 5KB to 16KB). Coverage fraction
            # treats all alleles equally regardless of length.
            allele_scores[allele] = coverage * (1.0 + coverage)
            allele_supports[allele] = n_found

        # Filter alleles with insufficient support
        valid_alleles = [
            a for a, s in allele_supports.items()
            if s >= self.min_kmer_support
        ]
        if not valid_alleles:
            valid_alleles = sorted(
                allele_scores.keys(), key=lambda a: -allele_scores[a],
            )[:20]
        if not valid_alleles:
            return KmerGenotypeResult(
                locus=locus, allele1="", allele2="",
                score=0.0, confidence=0.0,
                kmers_supporting_a1=0, kmers_supporting_a2=0,
                n_alleles_evaluated=0,
            )

        # Top candidates with diverse 2-digit group representation.
        # Without this, the top 50 can be dominated by subtypes of one
        # allele group (e.g., 40 DRB1*04 subtypes), leaving no room
        # for the second allele's group (e.g., DRB1*15).
        # Strategy: take the top 5 alleles per 2-digit group, then
        # fill remaining slots by overall score.
        groups_2d = group_alleles_by_resolution(valid_alleles, level=1)
        top_alleles: list[str] = []
        per_group_limit = 5
        # First pass: top alleles per 2-digit group
        for group_name in sorted(groups_2d.keys(),
                                  key=lambda g: -max(allele_scores.get(a, 0)
                                                     for a in groups_2d[g])):
            members = sorted(groups_2d[group_name],
                           key=lambda a: -allele_scores[a])
            top_alleles.extend(members[:per_group_limit])
        # Second pass: fill to 50 by overall score
        remaining = [a for a in valid_alleles if a not in set(top_alleles)]
        remaining.sort(key=lambda a: -allele_scores[a])
        top_alleles.extend(remaining[:max(0, 50 - len(top_alleles))])
        top_alleles = top_alleles[:50]

        # Compute locus-relevant read k-mers: read k-mers that match
        # ANY candidate allele. This is the signal we want to explain.
        all_candidate_kmers: set[str] = set()
        for a in top_alleles:
            all_candidate_kmers |= allele_kmers[a]
        locus_read_kmers = read_kmers & all_candidate_kmers
        n_locus_signal = len(locus_read_kmers)

        # === Phase 1: "Explain the signal" pair scoring ===
        # Score = fraction of locus-relevant read k-mers explained by pair.
        # The true diploid pair explains ALL read k-mers (from both alleles).
        # A wrong homo pair misses the second allele's unique k-mers.
        best_pair: tuple[str, str] | None = None
        best_pair_score = -1.0

        for i, a1 in enumerate(top_alleles):
            for j in range(i, len(top_alleles)):
                a2 = top_alleles[j]
                pair_kmers = allele_kmers[a1] | allele_kmers[a2]
                pair_explained = len(locus_read_kmers & pair_kmers)

                # Primary: fraction of locus signal explained
                if n_locus_signal > 0:
                    signal_explained = pair_explained / n_locus_signal
                else:
                    signal_explained = 0.0

                # Secondary: pair self-coverage (what fraction of pair's
                # own k-mers are supported by reads)
                pair_cov = pair_explained / max(len(pair_kmers), 1)

                # Combined: mostly signal-explained, with self-coverage tiebreak
                cov_score = signal_explained + 0.1 * pair_cov

                # Light frequency tiebreaker
                f1 = self.freq_db.get_frequency(a1)
                f2 = self.freq_db.get_frequency(a2)
                freq_boost = 1.0 + ((f1 + f2) / 2) * self.frequency_weight * 10
                pair_score = cov_score * freq_boost

                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (a1, a2)

        if not best_pair:
            return KmerGenotypeResult(
                locus=locus, allele1="", allele2="",
                score=0.0, confidence=0.0,
                kmers_supporting_a1=0, kmers_supporting_a2=0,
                n_alleles_evaluated=len(valid_alleles),
            )

        a1, a2 = best_pair

        # === Phase 2: Depth-based het rescue ===
        # If coverage scoring picked a homo/close-het pair but depth
        # evidence supports a different second allele, override.
        if read_kmer_counts:
            a1_rescued, a2_rescued = self._depth_rescue_het(
                a1, a2, top_alleles, allele_kmers, read_kmers,
                read_kmer_counts,
            )
            if (a1_rescued, a2_rescued) != (a1, a2):
                a1, a2 = a1_rescued, a2_rescued

        confidence = min(1.0, best_pair_score / max(best_pair_score * 0.95, 1e-6))

        return KmerGenotypeResult(
            locus=locus, allele1=a1, allele2=a2,
            score=best_pair_score, confidence=confidence,
            kmers_supporting_a1=allele_supports.get(a1, 0),
            kmers_supporting_a2=allele_supports.get(a2, 0),
            n_alleles_evaluated=len(valid_alleles),
            is_homozygous=(a1 == a2),
        )

    def _depth_rescue_het(
        self,
        a1: str, a2: str,
        candidates: list[str],
        allele_kmers: dict[str, set[str]],
        read_kmers: set[str],
        kmer_counts: dict[str, int],
    ) -> tuple[str, str]:
        """Depth-based heterozygous rescue.

        If coverage scoring picked a homozygous or close-het pair,
        search ALL candidates for a second allele whose unique k-mers
        have depth consistent with a het pair (~half of shared depth).

        This catches cases where the second allele is from a different
        2-digit group (e.g., B*08:01 + B*56:01) that the old
        same-group-only disambiguation would miss.
        """
        k1 = allele_kmers.get(a1, set())
        if not k1:
            return (a1, a2)

        # Check if current call is effectively homozygous
        if a1 != a2:
            a1_2f = truncate_to_resolution(a1, 2)
            a2_2f = truncate_to_resolution(a2, 2)
            if a1_2f != a2_2f:
                # Already a clear het across different groups — keep it
                return (a1, a2)

        # Estimate baseline depth from a1 k-mers
        a1_depths = [kmer_counts.get(km, 0) for km in k1]
        a1_covered = sorted([d for d in a1_depths if d > 0])
        if len(a1_covered) < 10:
            return (a1, a2)
        median_depth = a1_covered[len(a1_covered) // 2]
        if median_depth < 3:
            return (a1, a2)

        expected_het_depth = median_depth / 2

        # Search ALL candidates for a second allele with depth evidence
        best_partner = None
        best_score = 0.0

        for cand in candidates:
            if cand == a1:
                continue
            # Skip candidates in the same 2-field group (already tested)
            cand_2f = truncate_to_resolution(cand, 2)
            a1_2f = truncate_to_resolution(a1, 2)

            k_cand = allele_kmers.get(cand, set())
            if not k_cand:
                continue

            # K-mers unique to the candidate (not shared with a1)
            unique_cand = k_cand - k1
            if len(unique_cand) < 3:
                continue

            # Measure depth of candidate's unique k-mers
            depths = [kmer_counts.get(km, 0) for km in unique_cand]
            covered = [d for d in depths if d > 0]

            # Need substantial fraction of unique k-mers to have depth
            cov_frac = len(covered) / len(unique_cand)
            if cov_frac < 0.3:
                continue

            median_cand = sorted(covered)[len(covered) // 2]

            # Does depth match het expectation? (~half of shared)
            ratio = median_cand / max(expected_het_depth, 1)
            if ratio < 0.2 or ratio > 5.0:
                continue

            depth_match = 1.0 - min(1.0, abs(1.0 - ratio) * 0.5)
            score = cov_frac * depth_match

            # Also weight by overall coverage of the candidate
            n_cand_found = sum(1 for km in k_cand if km in read_kmers)
            cand_cov = n_cand_found / max(len(k_cand), 1)
            score *= cand_cov

            if score > best_score:
                best_score = score
                best_partner = cand

        if best_partner and best_score >= 0.15:
            logger.info(
                "  %s: depth rescue: (%s, %s) -> (%s, %s) score=%.2f",
                parse_allele_name(a1).locus,
                truncate_to_resolution(a1, 2),
                truncate_to_resolution(a2, 2),
                truncate_to_resolution(a1, 2),
                truncate_to_resolution(best_partner, 2),
                best_score,
            )
            return (a1, best_partner)

        return (a1, a2)


def extract_read_kmers(
    fastq_paths: list[Path],
    k: int = 21,
    max_reads: int | None = None,
) -> set[str]:
    """Extract canonical k-mers from FASTQ files into a set."""
    import gzip

    kmers: set[str] = set()
    n_reads = 0

    for fq_path in fastq_paths:
        if not fq_path.exists():
            continue

        opener = gzip.open if str(fq_path).endswith(".gz") else open
        with opener(fq_path, "rt") as fh:
            line_no = 0
            for line in fh:
                line_no += 1
                if line_no % 4 == 2:
                    seq = line.strip().upper()
                    if "N" in seq:
                        for sub in seq.split("N"):
                            if len(sub) >= k:
                                for i in range(len(sub) - k + 1):
                                    kmers.add(canonical_kmer(sub[i:i + k]))
                    else:
                        for i in range(len(seq) - k + 1):
                            kmers.add(canonical_kmer(seq[i:i + k]))
                    n_reads += 1
                    if max_reads and n_reads >= max_reads:
                        logger.info(
                            "Extracted %d k-mers from %d reads (capped)",
                            len(kmers), n_reads,
                        )
                        return kmers

    logger.info("Extracted %d unique k-mers from %d reads", len(kmers), n_reads)
    return kmers


def extract_read_kmer_counts(
    fastq_paths: list[Path],
    k: int = 21,
    target_kmers: set[str] | None = None,
    max_reads: int | None = None,
) -> dict[str, int]:
    """Extract k-mer counts (depth) from FASTQ files."""
    import gzip

    counts: Counter = Counter()
    n_reads = 0

    for fq_path in fastq_paths:
        if not fq_path.exists():
            continue

        opener = gzip.open if str(fq_path).endswith(".gz") else open
        with opener(fq_path, "rt") as fh:
            line_no = 0
            for line in fh:
                line_no += 1
                if line_no % 4 == 2:
                    seq = line.strip().upper()
                    if "N" in seq:
                        for sub in seq.split("N"):
                            if len(sub) >= k:
                                for i in range(len(sub) - k + 1):
                                    km = canonical_kmer(sub[i:i + k])
                                    if target_kmers is None or km in target_kmers:
                                        counts[km] += 1
                    else:
                        for i in range(len(seq) - k + 1):
                            km = canonical_kmer(seq[i:i + k])
                            if target_kmers is None or km in target_kmers:
                                counts[km] += 1
                    n_reads += 1
                    if max_reads and n_reads >= max_reads:
                        return dict(counts)

    return dict(counts)


def kmer_genotype_all_loci(
    fastq_paths: list[Path],
    imgt_db,  # IMGTDatabase
    loci: list[str],
    k: int = 21,
    max_reads: int = 500_000,
    candidates_per_locus: dict[str, list[str]] | None = None,
    data_type: str = "short",
) -> dict[str, KmerGenotypeResult]:
    """Run k-mer genotyping across multiple loci with depth-based het rescue."""
    use_genomic = data_type in ("short", "pacbio", "hifi", "ont", "targeted_capture")

    logger.info("=== K-mer Genotyping (alignment-free, %s sequences, depth-aware) ===",
                 "genomic" if use_genomic else "CDS")
    logger.info("Extracting k-mers from %d FASTQ files (k=%d)...",
                 len(fastq_paths), k)

    # Extract both presence set AND depth counts
    read_kmers = extract_read_kmers(fastq_paths, k=k, max_reads=max_reads)

    if not read_kmers:
        logger.warning("No k-mers extracted from reads")
        return {}

    # Extract depth counts for het/homo discrimination
    read_kmer_counts = extract_read_kmer_counts(
        fastq_paths, k=k, target_kmers=read_kmers, max_reads=max_reads,
    )
    logger.info("Extracted %d unique k-mers with depth info", len(read_kmer_counts))

    genotyper = KmerGenotyper(k=k)
    results: dict[str, KmerGenotypeResult] = {}

    for locus in loci:
        logger.info("K-mer genotyping locus %s...", locus)
        if use_genomic:
            seqs = imgt_db.load_genomic(locus)
            if not seqs:
                seqs = imgt_db.load_cds(locus)
        else:
            seqs = imgt_db.load_cds(locus)
            if not seqs:
                seqs = imgt_db.load_genomic(locus)

        if not seqs:
            logger.warning("No sequences for locus %s", locus)
            continue

        candidates = candidates_per_locus.get(locus) if candidates_per_locus else None

        result = genotyper.genotype_locus(
            locus=locus,
            allele_sequences=seqs,
            read_kmers=read_kmers,
            candidate_alleles=candidates,
            read_kmer_counts=read_kmer_counts,
        )

        results[locus] = result
        logger.info(
            "  %s: %s / %s (score=%.1f, conf=%.2f, n_alleles=%d, %s)",
            locus, result.allele1, result.allele2,
            result.score, result.confidence, result.n_alleles_evaluated,
            "homo" if result.is_homozygous else "het",
        )

    return results
