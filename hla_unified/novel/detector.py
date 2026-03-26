"""Proactive novel allele detection for HLA-Unified V2.

Unlike the V1 assembly-only approach (which only triggers for low-confidence
calls), this module proactively screens for novel alleles after ILP genotyping
by looking for systematic unexplained read evidence.

Detection signals:
1. High fraction of unexplained reads after ILP genotyping
2. K-mer validation shows high unexpected k-mer fraction
3. Consistent mismatch pattern at specific positions
4. Assembly produces contigs not matching any known allele at >99.5% identity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..assembly.targeted_assembler import (
    TargetedAssembler, AssemblyResult, NovelAlleleReport,
)
from ..genotyper.ilp_solver import ILPResult
from ..kmer.validator import KmerValidationResult

logger = logging.getLogger(__name__)


@dataclass
class NovelAlleleCandidate:
    """A candidate novel allele detected at a locus."""
    locus: str
    detection_reason: str
    closest_known_allele: str
    identity_to_closest: float
    n_mismatches: int
    mismatch_summary: str
    assembly_result: AssemblyResult | None = None
    novel_report: NovelAlleleReport | None = None
    confidence: str = "LOW"  # LOW, MEDIUM, HIGH
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "locus": self.locus,
            "detection_reason": self.detection_reason,
            "closest_known_allele": self.closest_known_allele,
            "identity_to_closest": round(self.identity_to_closest, 6),
            "n_mismatches": self.n_mismatches,
            "mismatch_summary": self.mismatch_summary,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }
        if self.novel_report:
            result["novel_report"] = {
                "closest_allele": self.novel_report.closest_allele,
                "identity": round(self.novel_report.identity, 6),
                "n_snps": self.novel_report.n_snps,
                "n_insertions": self.novel_report.n_insertions,
                "n_deletions": self.novel_report.n_deletions,
                "aligned_length": self.novel_report.aligned_length,
                "summary": self.novel_report.summary,
                "mismatches": [
                    {
                        "position": m.position,
                        "ref_base": m.ref_base,
                        "alt_base": m.alt_base,
                        "variant_type": m.variant_type,
                    }
                    for m in self.novel_report.mismatches
                ],
            }
        return result


class NovelAlleleDetector:
    """Proactively detects potential novel alleles.

    Screens all loci after genotyping, not just low-confidence ones.
    Uses multiple evidence signals to flag candidates.
    """

    def __init__(
        self,
        unexplained_read_threshold: float = 0.15,
        unexpected_kmer_threshold: float = 0.20,
        identity_threshold: float = 0.995,
        min_reads_for_detection: int = 20,
        threads: int = 4,
        data_type: str = "short",
    ) -> None:
        self.unexplained_read_threshold = unexplained_read_threshold
        self.unexpected_kmer_threshold = unexpected_kmer_threshold
        self.identity_threshold = identity_threshold
        self.min_reads_for_detection = min_reads_for_detection
        self.threads = threads
        self.data_type = data_type

    def screen_locus(
        self,
        locus: str,
        ilp_result: ILPResult | None = None,
        kmer_result: KmerValidationResult | None = None,
        assembly_result: AssemblyResult | None = None,
    ) -> NovelAlleleCandidate | None:
        """Screen a single locus for novel allele signals.

        Returns a NovelAlleleCandidate if signals are detected, None otherwise.
        """
        signals: list[str] = []
        evidence: dict[str, Any] = {}

        # Signal 1: High unexplained reads
        if ilp_result and ilp_result.total_reads >= self.min_reads_for_detection:
            explained_frac = ilp_result.reads_explained / max(ilp_result.total_reads, 1)
            unexplained_frac = 1.0 - explained_frac
            evidence["unexplained_read_fraction"] = round(unexplained_frac, 4)

            if unexplained_frac > self.unexplained_read_threshold:
                signals.append(
                    f"High unexplained reads: {unexplained_frac:.1%} "
                    f"(threshold {self.unexplained_read_threshold:.0%})"
                )

        # Signal 2: Unexpected k-mers
        if kmer_result:
            evidence["unexpected_kmer_fraction"] = round(
                kmer_result.unexpected_kmer_fraction, 4,
            )
            evidence["kmer_coverage"] = round(
                kmer_result.proportion_kmers_covered, 4,
            )

            if kmer_result.unexpected_kmer_fraction > self.unexpected_kmer_threshold:
                signals.append(
                    f"High unexpected k-mers: {kmer_result.unexpected_kmer_fraction:.1%}"
                )

            # Low coverage of expected k-mers can also indicate novel alleles
            if kmer_result.proportion_kmers_covered < 0.85:
                signals.append(
                    f"Low allele k-mer coverage: {kmer_result.proportion_kmers_covered:.1%}"
                )

        # Signal 3: Assembly found a novel allele
        if assembly_result and assembly_result.is_novel:
            signals.append(
                f"Assembly detected novel allele (best match "
                f"{assembly_result.best_match_allele} at "
                f"{assembly_result.best_match_identity:.2%})"
            )
            evidence["assembly_identity"] = round(
                assembly_result.best_match_identity, 6,
            )

        if not signals:
            return None

        # Build candidate
        closest_allele = ""
        identity = 0.0
        n_mismatches = 0
        novel_report = None
        mismatch_summary = ""

        if assembly_result and assembly_result.novel_report:
            novel_report = assembly_result.novel_report
            closest_allele = novel_report.closest_allele
            identity = novel_report.identity
            n_mismatches = (
                novel_report.n_snps
                + novel_report.n_insertions
                + novel_report.n_deletions
            )
            mismatch_summary = novel_report.summary
        elif assembly_result:
            closest_allele = assembly_result.best_match_allele
            identity = assembly_result.best_match_identity
            mismatch_summary = f"Identity to {closest_allele}: {identity:.2%}"

        # Confidence of novel allele call
        n_signals = len(signals)
        if n_signals >= 3:
            confidence = "HIGH"
        elif n_signals >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return NovelAlleleCandidate(
            locus=locus,
            detection_reason="; ".join(signals),
            closest_known_allele=closest_allele,
            identity_to_closest=identity,
            n_mismatches=n_mismatches,
            mismatch_summary=mismatch_summary,
            assembly_result=assembly_result,
            novel_report=novel_report,
            confidence=confidence,
            evidence=evidence,
        )

    def screen_all_loci(
        self,
        loci: list[str],
        ilp_results: dict[str, ILPResult],
        kmer_results: dict[str, KmerValidationResult],
        assembly_results: dict[str, AssemblyResult],
    ) -> dict[str, NovelAlleleCandidate]:
        """Screen all loci for novel alleles."""
        candidates = {}
        for locus in loci:
            candidate = self.screen_locus(
                locus=locus,
                ilp_result=ilp_results.get(locus),
                kmer_result=kmer_results.get(locus),
                assembly_result=assembly_results.get(locus),
            )
            if candidate:
                candidates[locus] = candidate
                logger.info(
                    "Novel allele candidate at %s: %s (confidence=%s)",
                    locus, candidate.detection_reason, candidate.confidence,
                )

        if candidates:
            logger.info(
                "Detected %d novel allele candidate(s): %s",
                len(candidates), ", ".join(candidates.keys()),
            )
        return candidates
