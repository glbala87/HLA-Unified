"""K-mer-based HLA genotyper.

An alignment-free genotyping engine that uses exact k-mer matching
against IMGT allele sequences. More discriminating than alignment-based
scoring because it directly counts shared sequence content rather than
relying on aligner heuristics.

Approach:
1. Build a k-mer index per locus: each k-mer maps to the set of alleles
   that contain it. K-mers shared by many alleles are uninformative;
   k-mers unique (or near-unique) to specific alleles are highly diagnostic.
2. Stream reads, extracting their k-mers.
3. For each allele, count the read k-mers that match its sequence,
   weighted by the inverse of how many alleles share each k-mer (so
   private k-mers count more than shared ones).
4. At each locus, pick the diploid pair that explains the most weighted
   k-mer evidence, with population frequency priors as a tiebreaker.

This approach is robust against:
- Multi-mapping noise (no aligner involved)
- Negative bowtie2 score conventions (we use exact matching)
- Off-target reads (uninformative shared k-mers get downweighted)

Inspired by HLAforest, T1K, and HLA-VBSeq's k-mer scoring stages.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
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
    """Alignment-free HLA genotyper using weighted k-mer matching."""

    def __init__(
        self,
        k: int = 21,
        frequency_weight: float = 0.10,  # moderate tiebreaker
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
    ) -> KmerGenotypeResult:
        """Genotype a single locus using k-mer evidence.

        Args:
            locus: Locus name (e.g., "A")
            allele_sequences: All allele sequences for this locus from IMGT
            read_kmers: Canonical k-mers extracted from sample reads
            candidate_alleles: Optional restriction to a subset of alleles

        Returns:
            Best diploid pair with confidence
        """
        # Restrict to candidates if specified
        if candidate_alleles:
            allele_sequences = {
                a: s for a, s in allele_sequences.items()
                if a in candidate_alleles
            }

        # Further restrict to alleles whose 2-field group is in the
        # population frequency database. This filters out rare/exotic
        # alleles (A*11:335, B*55:111, etc.) that have unique IMGT
        # sequences but no real-world prevalence. The truth alleles for
        # any clinical sample are virtually always in the AFND database.
        common_alleles = {}
        for allele, seq in allele_sequences.items():
            two_field = truncate_to_resolution(allele, 2)
            two_field_clean = two_field.removeprefix("HLA-")
            freq = self.freq_db.get_frequency(two_field_clean)
            if freq > 1e-4:  # above floor
                common_alleles[allele] = seq

        # If filtering left nothing (rare locus), fall back to original set
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

        # Coverage-based scoring: for each allele, what fraction of its
        # k-mers are present in the read pool? High coverage = strong
        # evidence the entire allele sequence is supported.
        # Uniform weights (no IDF) — we want all positions to count equally.
        kmer_weights: dict[str, float] = {}
        for kmers in allele_kmers.values():
            for km in kmers:
                kmer_weights[km] = 1.0

        # Score = number of allele's k-mers found in reads (raw coverage count)
        # Plus a small bonus for fractional coverage (length-normalized)
        allele_scores: dict[str, float] = {}
        allele_supports: dict[str, int] = {}
        for allele, kmers in allele_kmers.items():
            n_total = len(kmers)
            if n_total == 0:
                allele_scores[allele] = 0.0
                allele_supports[allele] = 0
                continue
            n_found = sum(1 for km in kmers if km in read_kmers)
            # Score = absolute count + coverage fraction bonus
            # This rewards alleles whose entire sequence is supported,
            # not just alleles with unique noise k-mers
            coverage = n_found / n_total
            allele_scores[allele] = n_found * (1.0 + coverage)
            allele_supports[allele] = n_found

        # Filter alleles with insufficient k-mer support
        valid_alleles = [
            a for a, s in allele_supports.items()
            if s >= self.min_kmer_support
        ]

        if not valid_alleles:
            # Fall back to top scoring even with low support
            valid_alleles = sorted(
                allele_scores.keys(),
                key=lambda a: -allele_scores[a],
            )[:20]

        if not valid_alleles:
            return KmerGenotypeResult(
                locus=locus, allele1="", allele2="",
                score=0.0, confidence=0.0,
                kmers_supporting_a1=0, kmers_supporting_a2=0,
                n_alleles_evaluated=0,
            )

        # Find best diploid pair: maximize coverage with frequency prior.
        max_score = max(allele_scores.get(a, 0) for a in valid_alleles)

        # Pick top candidates by raw read score
        top_alleles = sorted(
            valid_alleles, key=lambda a: -allele_scores[a],
        )[:30]

        best_pair: tuple[str, str] | None = None
        best_pair_score = -1.0

        for i, a1 in enumerate(top_alleles):
            for j in range(i, len(top_alleles)):  # i to allow homozygous
                a2 = top_alleles[j]
                coverage_score = self._score_diploid_pair(
                    a1, a2, allele_kmers, read_kmers, kmer_weights,
                )
                # Frequency prior: use average frequency, not sum. This
                # avoids biasing homozygous pairs (where the same frequency
                # is counted twice). A heterozygous pair of two common
                # alleles gets the same boost as a homozygous pair of one.
                f1 = self.freq_db.get_frequency(a1)
                f2 = self.freq_db.get_frequency(a2)
                avg_freq = (f1 + f2) / 2
                freq_boost = 1.0 + avg_freq * self.frequency_weight * 10

                pair_score = coverage_score * freq_boost

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
        # Compute confidence: gap between best pair and runner-up
        runner_up_score = best_pair_score * 0.95  # default if only one pair
        confidence = min(1.0, best_pair_score / max(runner_up_score, 1e-6))

        return KmerGenotypeResult(
            locus=locus,
            allele1=a1,
            allele2=a2,
            score=best_pair_score,
            confidence=confidence,
            kmers_supporting_a1=allele_supports.get(a1, 0),
            kmers_supporting_a2=allele_supports.get(a2, 0),
            n_alleles_evaluated=len(valid_alleles),
            is_homozygous=(a1 == a2),
        )

    def _score_diploid_pair(
        self,
        a1: str,
        a2: str,
        allele_kmers: dict[str, set[str]],
        read_kmers: set[str],
        kmer_weights: dict[str, float],
    ) -> float:
        """Score a diploid pair by coverage, with a modest union bonus
        for heterozygous pairs that genuinely expand the covered k-mer set.

        For homozygous pair (a1, a1): score = individual_coverage
        For het pair (a1, a2): score = geo_coverage + union_boost
            where union_boost = (union_count - max(n1_found, n2_found)) / max(n1, n2)
            This is only positive if the second allele contributes
            k-mers beyond what the first already covers.
        """
        k1 = allele_kmers[a1]
        k2 = allele_kmers[a2]

        n1 = len(k1)
        n2 = len(k2)
        if n1 == 0 or n2 == 0:
            return 0.0

        n1_found = sum(1 for km in k1 if km in read_kmers)
        n2_found = sum(1 for km in k2 if km in read_kmers)
        c1 = n1_found / n1
        c2 = n2_found / n2

        geo_coverage = math.sqrt(c1 * c2)

        if a1 == a2:
            # Homozygous pair — no union bonus
            return geo_coverage

        # Heterozygous — measure how much the union expands coverage
        # beyond the single better allele
        union = k1 | k2
        union_found = sum(1 for km in union if km in read_kmers)
        max_single_found = max(n1_found, n2_found)

        # If union adds k-mers that are ALSO found in reads, it's a real
        # heterozygous pair. If the union adds k-mers NOT in reads, they
        # don't count (union_found would equal max_single_found).
        union_expansion = union_found - max_single_found
        max_n = max(n1, n2)
        union_bonus = union_expansion / max_n if max_n > 0 else 0.0

        # Modest bonus for genuine het pairs (up to ~10% additional score)
        return geo_coverage + 0.15 * union_bonus


def extract_read_kmers(
    fastq_paths: list[Path],
    k: int = 21,
    max_reads: int | None = None,
) -> set[str]:
    """Extract canonical k-mers from FASTQ files into a set.

    Uses a set (not Counter) for memory efficiency — we only need
    to know which k-mers are present, not their counts.
    """
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
                if line_no % 4 == 2:  # sequence line
                    seq = line.strip().upper()
                    if "N" in seq:
                        # Skip k-mers spanning N
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
    """Extract k-mer counts (depth) from FASTQ files.

    Counts occurrences of each canonical k-mer. If target_kmers is
    provided, only count k-mers in that set (memory optimization for
    depth-based disambiguation of specific allele pairs).

    Used for depth-based heterozygous vs homozygous disambiguation:
    true heterozygous calls have ~1x depth at distinguishing positions
    while homozygous has ~2x depth at those same positions.
    """
    import gzip
    from collections import Counter

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


def disambiguate_close_pair(
    locus: str,
    a1: str,
    a2: str,
    allele_sequences: dict[str, str],
    candidate_alleles: list[str],
    fastq_paths: list[Path],
    k: int = 21,
    max_reads: int = 500_000,
) -> tuple[str, str]:
    """Depth-based disambiguation for close allele pairs.

    When the k-mer genotyper picks a homozygous call (a1, a1) but reads
    may actually support a heterozygous pair (a1, a2') within the same
    2-field group, this uses k-mer DEPTH to decide.

    Strategy:
    1. Consider all 2-field groups in the same 2-digit group as a1
    2. For each candidate het pair (a1, candidate_a2), find k-mers
       unique to candidate_a2 (not shared with a1)
    3. Measure the DEPTH of those unique k-mers in reads
    4. If a candidate_a2 has unique k-mers with depth ~= depth of shared
       k-mers / 2, that's a real heterozygous pair (both alleles at 1x)
    5. If no candidate has unique k-mer depth significantly above noise,
       the call is genuinely homozygous

    Returns: (a1, final_a2) — either the original (a1, a1) if truly homo,
    or (a1, better_a2) if depth evidence supports heterozygous.
    """
    from ..reference.loci import parse_allele_name, truncate_to_resolution

    if a1 != a2:
        # Already heterozygous, no disambiguation needed
        return (a1, a2)

    info = parse_allele_name(a1)
    a1_2digit = info.field1

    # Find candidates in the same 2-digit group (e.g., all DPB1*04:xx)
    same_2digit_candidates: list[str] = []
    for cand in candidate_alleles:
        if cand == a1:
            continue
        cand_info = parse_allele_name(cand)
        if cand_info.locus == info.locus and cand_info.field1 == a1_2digit:
            same_2digit_candidates.append(cand)

    if not same_2digit_candidates:
        return (a1, a1)

    # Get a1 k-mers and baseline depth
    a1_kmers = extract_canonical_kmers(allele_sequences[a1], k)

    # Collect all distinguishing k-mers across candidate alternatives
    distinguishing_kmers_by_cand: dict[str, set[str]] = {}
    all_distinguishing: set[str] = set()
    for cand in same_2digit_candidates:
        if cand not in allele_sequences:
            continue
        cand_kmers = extract_canonical_kmers(allele_sequences[cand], k)
        unique_to_cand = cand_kmers - a1_kmers
        if len(unique_to_cand) >= 3:  # need at least 3 distinguishing k-mers
            distinguishing_kmers_by_cand[cand] = unique_to_cand
            all_distinguishing |= unique_to_cand

    if not distinguishing_kmers_by_cand:
        return (a1, a1)

    # Get depth of a1 shared k-mers (baseline) + distinguishing k-mers
    target_kmers = a1_kmers | all_distinguishing
    kmer_counts = extract_read_kmer_counts(
        fastq_paths, k=k, target_kmers=target_kmers, max_reads=max_reads,
    )

    # Baseline depth from a1 k-mers (median of those with any coverage)
    a1_depths = [kmer_counts.get(km, 0) for km in a1_kmers]
    a1_depths_covered = sorted([d for d in a1_depths if d > 0])
    if not a1_depths_covered:
        return (a1, a1)

    median_a1_depth = a1_depths_covered[len(a1_depths_covered) // 2]

    # For each candidate, check if its distinguishing k-mers have depth
    # consistent with a heterozygous pair (~50% of a1 depth)
    best_cand = None
    best_score = 0.0
    for cand, unique_kmers in distinguishing_kmers_by_cand.items():
        depths = [kmer_counts.get(km, 0) for km in unique_kmers]
        depths_covered = [d for d in depths if d > 0]
        if len(depths_covered) < max(3, len(unique_kmers) * 0.5):
            continue  # need most unique k-mers to have coverage

        median_cand_depth = sorted(depths_covered)[len(depths_covered) // 2]

        # Ratio of candidate depth to half a1 depth
        # If heterozygous (a1, cand): cand_depth ≈ a1_depth / 2
        # If homozygous a1: cand depth should be very low (noise)
        expected_het_depth = median_a1_depth / 2
        if expected_het_depth == 0:
            continue

        ratio = median_cand_depth / expected_het_depth
        # Accept if ratio is in [0.5, 2.0] — within 2x of expected het depth
        if 0.5 <= ratio <= 2.0:
            # Score: fraction of unique k-mers covered * depth match quality
            coverage_frac = len(depths_covered) / len(unique_kmers)
            depth_match = 1.0 - abs(1.0 - ratio)  # 1.0 when ratio=1
            score = coverage_frac * depth_match
            if score > best_score:
                best_score = score
                best_cand = cand

    if best_cand and best_score >= 0.5:
        logger.info(
            "  %s: depth-based disambiguation: (%s, %s) -> (%s, %s) score=%.2f",
            locus, a1, a1, a1, best_cand, best_score,
        )
        return (a1, best_cand)

    return (a1, a1)


def kmer_genotype_all_loci(
    fastq_paths: list[Path],
    imgt_db,  # IMGTDatabase
    loci: list[str],
    k: int = 21,
    max_reads: int = 500_000,
    candidates_per_locus: dict[str, list[str]] | None = None,
) -> dict[str, KmerGenotypeResult]:
    """Run k-mer genotyping across multiple loci.

    Args:
        fastq_paths: HLA-filtered read FASTQs (from prefilter)
        imgt_db: IMGTDatabase instance
        loci: Loci to type
        k: K-mer size
        max_reads: Cap total reads processed for memory
        candidates_per_locus: Optional restriction to specific alleles per
            locus (e.g., from prefilter). If None, uses all IMGT alleles.
            Strongly recommended — restricts the search space and dramatically
            improves accuracy by excluding noise from rare allele subtypes.

    Returns:
        Dict mapping locus → KmerGenotypeResult
    """
    logger.info("=== K-mer Genotyping (alignment-free) ===")
    logger.info("Extracting k-mers from %d FASTQ files (k=%d)...",
                 len(fastq_paths), k)

    read_kmers = extract_read_kmers(fastq_paths, k=k, max_reads=max_reads)

    if not read_kmers:
        logger.warning("No k-mers extracted from reads")
        return {}

    genotyper = KmerGenotyper(k=k)
    results: dict[str, KmerGenotypeResult] = {}

    for locus in loci:
        logger.info("K-mer genotyping locus %s...", locus)
        # Load CDS sequences for k-mer analysis
        cds = imgt_db.load_cds(locus)
        if not cds:
            cds = imgt_db.load_genomic(locus)

        if not cds:
            logger.warning("No sequences for locus %s", locus)
            continue

        # Restrict to prefilter candidates if available — this dramatically
        # improves accuracy by excluding rare-allele noise
        candidates = candidates_per_locus.get(locus) if candidates_per_locus else None

        result = genotyper.genotype_locus(
            locus=locus,
            allele_sequences=cds,
            read_kmers=read_kmers,
            candidate_alleles=candidates,
        )

        # Depth-based disambiguation for effectively-homozygous calls.
        # Triggered when either:
        # (a) result.is_homozygous is True (same 4-field allele twice), OR
        # (b) both alleles truncate to the same 2-field group (fake-het
        #     between close subtypes like DPB1*04:01:01:09 + DPB1*04:01:104)
        effectively_homo = False
        if result.allele1:
            if result.is_homozygous:
                effectively_homo = True
            else:
                a1_2field = truncate_to_resolution(result.allele1, 2)
                a2_2field = truncate_to_resolution(result.allele2, 2)
                if a1_2field == a2_2field:
                    effectively_homo = True

        if effectively_homo and candidates:
            disambig_result = disambiguate_close_pair(
                locus=locus,
                a1=result.allele1,
                a2=result.allele1,  # use a1 for both since effectively-homo
                allele_sequences=cds,
                candidate_alleles=candidates,
                fastq_paths=fastq_paths,
                k=k,
                max_reads=max_reads,
            )
            if disambig_result[1] != result.allele1:
                # Disambiguation found a het partner
                result.allele2 = disambig_result[1]
                result.is_homozygous = False
            else:
                # No het partner found — ensure we report as homozygous
                result.allele2 = result.allele1
                result.is_homozygous = True

        results[locus] = result
        logger.info(
            "  %s: %s / %s (score=%.1f, conf=%.2f, n_alleles=%d)",
            locus, result.allele1, result.allele2,
            result.score, result.confidence, result.n_alleles_evaluated,
        )

    return results
