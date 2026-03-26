"""Ambiguity classification for HLA genotype calls.

Classifies WHY a call is ambiguous, not just that it is. This is critical
for clinical reporting (transplant teams need to know if the ambiguity is
resolvable with more sequencing) and for research (understanding whether
the reference is incomplete or the data is insufficient).

Taxonomy of ambiguity reasons:
- UNAMBIGUOUS: single clear winner
- COVERAGE_GAP: insufficient reads in key exon/region
- PHASE_BREAK: heterozygous sites detected but reads don't span them
- CLOSE_ALLELES: top-2 pairs differ by very few positions
- EXON_ONLY_EVIDENCE: intronic differences unresolvable from exon data
- LOW_DEPTH: overall low coverage at this locus
- HOMOZYGOUS_AMBIGUOUS: can't distinguish true homozygous from close het
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .vb_estimator import ConfidenceResult
from ..genotyper.ilp_solver import ILPResult
from ..genotyper.read_matrix import ReadAlleleMatrix
from ..kmer.validator import KmerValidationResult
from ..phasing.haplotype_binner import PhasingResult

logger = logging.getLogger(__name__)


class AmbiguityReason(Enum):
    """Taxonomy of reasons a genotype call may be ambiguous."""
    UNAMBIGUOUS = "unambiguous"
    COVERAGE_GAP = "coverage_gap"
    PHASE_BREAK = "phase_break"
    CLOSE_ALLELES = "close_alleles"
    EXON_ONLY_EVIDENCE = "exon_only"
    LOW_DEPTH = "low_depth"
    HOMOZYGOUS_AMBIGUOUS = "homo_ambig"


@dataclass
class AmbiguityClassification:
    """Classification result for a single locus."""
    locus: str
    primary_reason: AmbiguityReason
    secondary_reasons: list[AmbiguityReason] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    resolution_suggestion: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return self.primary_reason != AmbiguityReason.UNAMBIGUOUS

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "primary_reason": self.primary_reason.value,
            "secondary_reasons": [r.value for r in self.secondary_reasons],
            "evidence": self.evidence,
            "resolution_suggestion": self.resolution_suggestion,
            "is_ambiguous": self.is_ambiguous,
        }


class AmbiguityClassifier:
    """Classifies the reason for ambiguity in an HLA genotype call.

    Integrates evidence from VB posteriors, k-mer validation, phasing,
    ILP genotyping, and allele sequence comparisons to determine the
    most likely cause of ambiguity.
    """

    def __init__(
        self,
        posterior_threshold: float = 0.99,
        ambiguity_gap_threshold: float = 0.10,
        close_allele_snp_threshold: int = 3,
        low_depth_threshold: int = 10,
        coverage_gap_fraction: float = 0.70,
    ) -> None:
        self.posterior_threshold = posterior_threshold
        self.ambiguity_gap_threshold = ambiguity_gap_threshold
        self.close_allele_snp_threshold = close_allele_snp_threshold
        self.low_depth_threshold = low_depth_threshold
        self.coverage_gap_fraction = coverage_gap_fraction

    def classify(
        self,
        locus: str,
        vb_result: ConfidenceResult | None = None,
        kmer_result: KmerValidationResult | None = None,
        phasing_result: PhasingResult | None = None,
        ilp_result: ILPResult | None = None,
        matrix: ReadAlleleMatrix | None = None,
        allele_sequences: dict[str, str] | None = None,
        data_type: str = "short",
    ) -> AmbiguityClassification:
        """Classify the ambiguity reason for a locus call.

        Evaluates multiple evidence sources in priority order:
        1. Check if call is unambiguous (high posterior + large gap)
        2. Check for low depth
        3. Check for close alleles (few SNP differences)
        4. Check for phase breaks
        5. Check for coverage gaps
        6. Check for exon-only evidence limitation
        7. Check for homozygous ambiguity
        """
        reasons: list[AmbiguityReason] = []
        evidence: dict[str, Any] = {}

        posterior = vb_result.posterior_prob if vb_result else 0.5
        evidence["posterior"] = posterior

        # Compute ambiguity gap
        ambiguity_gap = 1.0
        if vb_result and vb_result.alternative_pairs:
            second_best_post = vb_result.alternative_pairs[0][2]
            ambiguity_gap = posterior - second_best_post
        evidence["ambiguity_gap"] = ambiguity_gap

        # 1. Unambiguous check
        if posterior >= self.posterior_threshold and ambiguity_gap >= self.ambiguity_gap_threshold:
            return AmbiguityClassification(
                locus=locus,
                primary_reason=AmbiguityReason.UNAMBIGUOUS,
                evidence=evidence,
                resolution_suggestion="No additional data needed.",
            )

        # 2. Low depth
        total_reads = ilp_result.total_reads if ilp_result else 0
        evidence["total_reads"] = total_reads
        if total_reads < self.low_depth_threshold:
            reasons.append(AmbiguityReason.LOW_DEPTH)
            evidence["low_depth_threshold"] = self.low_depth_threshold

        # 3. Close alleles
        if vb_result and allele_sequences and vb_result.alternative_pairs:
            best_a1 = vb_result.allele1
            best_a2 = vb_result.allele2
            alt_a1, alt_a2 = vb_result.alternative_pairs[0][0], vb_result.alternative_pairs[0][1]

            snp_diff = self._count_snp_differences(
                best_a1, best_a2, alt_a1, alt_a2, allele_sequences,
            )
            evidence["snp_diff_to_alternative"] = snp_diff

            if snp_diff <= self.close_allele_snp_threshold:
                reasons.append(AmbiguityReason.CLOSE_ALLELES)

        # 4. Phase breaks
        if phasing_result:
            evidence["is_phased"] = phasing_result.is_phased
            evidence["n_het_sites"] = phasing_result.n_het_sites
            evidence["phase_confidence"] = phasing_result.phase_confidence

            if (phasing_result.n_het_sites > 0
                    and not phasing_result.is_phased):
                reasons.append(AmbiguityReason.PHASE_BREAK)
            elif (phasing_result.is_phased
                  and phasing_result.phase_confidence < 0.5):
                reasons.append(AmbiguityReason.PHASE_BREAK)

        # 5. Coverage gap
        if kmer_result:
            evidence["kmer_coverage"] = kmer_result.proportion_kmers_covered
            if kmer_result.proportion_kmers_covered < self.coverage_gap_fraction:
                reasons.append(AmbiguityReason.COVERAGE_GAP)

        # 6. Exon-only evidence
        if data_type in ("exome", "rna"):
            if vb_result and vb_result.alternative_pairs:
                # Check if top pairs differ only in non-exonic regions
                best_a1 = vb_result.allele1
                alt_a1 = vb_result.alternative_pairs[0][0]
                if self._differ_only_in_introns(best_a1, alt_a1):
                    reasons.append(AmbiguityReason.EXON_ONLY_EVIDENCE)
                    evidence["exon_only_note"] = (
                        f"Top alleles {best_a1} and {alt_a1} differ only "
                        f"at non-exonic positions (unresolvable from {data_type} data)"
                    )

        # 7. Homozygous ambiguity
        if ilp_result and ilp_result.is_homozygous:
            if ambiguity_gap < self.ambiguity_gap_threshold:
                reasons.append(AmbiguityReason.HOMOZYGOUS_AMBIGUOUS)
                evidence["homo_note"] = (
                    "Homozygous call but alternative heterozygous "
                    "pair has similar posterior"
                )

        # Determine primary reason
        if not reasons:
            # Generic low-confidence — likely a combination of factors
            primary = AmbiguityReason.CLOSE_ALLELES
            reasons = [primary]
        else:
            primary = reasons[0]

        secondary = reasons[1:] if len(reasons) > 1 else []

        suggestion = self._suggest_resolution(primary, data_type, evidence)

        return AmbiguityClassification(
            locus=locus,
            primary_reason=primary,
            secondary_reasons=secondary,
            evidence=evidence,
            resolution_suggestion=suggestion,
        )

    def _count_snp_differences(
        self,
        best_a1: str, best_a2: str,
        alt_a1: str, alt_a2: str,
        sequences: dict[str, str],
    ) -> int:
        """Count SNP differences between the best pair and alternative pair."""
        # Find which allele changed between the two pairs
        changed_best = []
        changed_alt = []
        if best_a1 != alt_a1:
            changed_best.append(best_a1)
            changed_alt.append(alt_a1)
        if best_a2 != alt_a2:
            changed_best.append(best_a2)
            changed_alt.append(alt_a2)

        total_snps = 0
        for b, a in zip(changed_best, changed_alt):
            seq_b = sequences.get(b, "")
            seq_a = sequences.get(a, "")
            if seq_b and seq_a:
                min_len = min(len(seq_b), len(seq_a))
                snps = sum(
                    1 for i in range(min_len)
                    if seq_b[i] != seq_a[i]
                    and seq_b[i] in "ACGT"
                    and seq_a[i] in "ACGT"
                )
                total_snps += snps

        return total_snps

    def _differ_only_in_introns(self, allele1: str, allele2: str) -> bool:
        """Check if two alleles differ only at resolution levels beyond 2-field.

        If the 2-field (protein-level) names match, the difference is in
        synonymous/non-coding positions — unresolvable from exon-only data.
        """
        from ..reference.loci import parse_allele_name
        info1 = parse_allele_name(allele1)
        info2 = parse_allele_name(allele2)
        return (info1.locus == info2.locus
                and info1.field1 == info2.field1
                and info1.field2 == info2.field2)

    def _suggest_resolution(
        self,
        reason: AmbiguityReason,
        data_type: str,
        evidence: dict,
    ) -> str:
        """Suggest what would resolve the ambiguity."""
        suggestions = {
            AmbiguityReason.COVERAGE_GAP: (
                "Increase sequencing depth or use targeted HLA capture. "
                "Current k-mer coverage is {kmer_coverage:.0%}."
            ),
            AmbiguityReason.PHASE_BREAK: (
                "Use long-read sequencing (PacBio HiFi/ONT) to span "
                "heterozygous sites across the full gene."
            ),
            AmbiguityReason.CLOSE_ALLELES: (
                "Top pairs differ by only {snp_diff_to_alternative} SNPs. "
                "Higher coverage or full-gene sequencing may discriminate."
            ),
            AmbiguityReason.EXON_ONLY_EVIDENCE: (
                "Alleles differ only in intronic/non-coding regions. "
                "Full-gene sequencing (WGS or targeted capture) is needed."
            ),
            AmbiguityReason.LOW_DEPTH: (
                "Only {total_reads} reads at this locus (minimum {low_depth_threshold}). "
                "Increase sequencing depth."
            ),
            AmbiguityReason.HOMOZYGOUS_AMBIGUOUS: (
                "Cannot distinguish true homozygosity from close heterozygosity. "
                "Higher depth or family/population priors may help."
            ),
        }

        template = suggestions.get(reason, "No specific suggestion available.")
        try:
            return template.format(**evidence)
        except (KeyError, ValueError):
            return template

    def classify_all_loci(
        self,
        loci: list[str],
        vb_results: dict[str, ConfidenceResult],
        kmer_results: dict[str, KmerValidationResult],
        phasing_results: dict[str, PhasingResult],
        ilp_results: dict[str, ILPResult],
        allele_sequences: dict[str, str] | None = None,
        data_type: str = "short",
    ) -> dict[str, AmbiguityClassification]:
        """Classify ambiguity for all loci."""
        results = {}
        for locus in loci:
            results[locus] = self.classify(
                locus=locus,
                vb_result=vb_results.get(locus),
                kmer_result=kmer_results.get(locus),
                phasing_result=phasing_results.get(locus),
                ilp_result=ilp_results.get(locus),
                allele_sequences=allele_sequences,
                data_type=data_type,
            )
        return results
