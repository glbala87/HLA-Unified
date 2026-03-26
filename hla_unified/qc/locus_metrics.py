"""Detailed per-locus QC metrics for HLA-Unified V2.

Goes beyond simple coverage stats to provide clinically actionable
QC evidence: haplotype balance, informative position coverage,
assembly continuity, and read quality metrics.

These metrics answer the key question: "Can I trust this locus call?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..phasing.haplotype_binner import PhasingResult
from ..assembly.targeted_assembler import AssemblyResult
from ..kmer.validator import KmerValidationResult

logger = logging.getLogger(__name__)


@dataclass
class DetailedLocusQC:
    """Comprehensive QC metrics for a single locus."""
    locus: str

    # === Haplotype balance ===
    haplotype_balance: float = 0.0  # min(bin_reads) / max(bin_reads); 1.0 = perfect
    allele1_depth: float = 0.0      # estimated read depth for allele 1
    allele2_depth: float = 0.0      # estimated read depth for allele 2
    depth_ratio: float = 0.0        # min(depth) / max(depth)

    # === Informative positions ===
    n_informative_positions: int = 0  # het sites distinguishing the two alleles
    informative_position_coverage: float = 0.0  # fraction of informative positions with reads
    exon_coverage_completeness: dict[int, float] = field(default_factory=dict)

    # === Assembly continuity (if assembly was run) ===
    assembly_n50: int = 0
    assembly_total_length: int = 0
    assembly_n_contigs: int = 0
    assembly_gene_coverage: float = 0.0  # fraction of gene length covered

    # === K-mer evidence ===
    kmer_coverage: float = 0.0
    kmer_uniformity: float = 0.0
    kmer_unexpected_fraction: float = 0.0

    # === Read quality ===
    mean_mapping_quality: float = 0.0
    total_reads: int = 0
    reads_explained: int = 0

    # === Phasing ===
    is_phased: bool = False
    n_het_sites: int = 0
    phase_confidence: float = 0.0

    # === Overall verdict ===
    qc_pass: bool = False
    qc_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "locus": self.locus,
            "haplotype_balance": {
                "balance_ratio": round(self.haplotype_balance, 4),
                "allele1_depth": round(self.allele1_depth, 1),
                "allele2_depth": round(self.allele2_depth, 1),
                "depth_ratio": round(self.depth_ratio, 4),
            },
            "informative_positions": {
                "count": self.n_informative_positions,
                "coverage": round(self.informative_position_coverage, 4),
                "exon_completeness": {
                    str(k): round(v, 4)
                    for k, v in self.exon_coverage_completeness.items()
                },
            },
            "assembly": {
                "n50": self.assembly_n50,
                "total_length": self.assembly_total_length,
                "n_contigs": self.assembly_n_contigs,
                "gene_coverage": round(self.assembly_gene_coverage, 4),
            },
            "kmer": {
                "coverage": round(self.kmer_coverage, 4),
                "uniformity": round(self.kmer_uniformity, 4),
                "unexpected_fraction": round(self.kmer_unexpected_fraction, 4),
            },
            "reads": {
                "total": self.total_reads,
                "explained": self.reads_explained,
                "mean_mapping_quality": round(self.mean_mapping_quality, 1),
            },
            "phasing": {
                "is_phased": self.is_phased,
                "n_het_sites": self.n_het_sites,
                "phase_confidence": round(self.phase_confidence, 4),
            },
            "verdict": {
                "qc_pass": self.qc_pass,
                "issues": self.qc_issues,
            },
        }


class LocusMetricsCalculator:
    """Computes detailed per-locus QC metrics from pipeline evidence."""

    def __init__(
        self,
        min_balance_ratio: float = 0.3,
        min_informative_positions: int = 3,
        min_kmer_coverage: float = 0.85,
    ) -> None:
        self.min_balance_ratio = min_balance_ratio
        self.min_informative_positions = min_informative_positions
        self.min_kmer_coverage = min_kmer_coverage

    def calculate(
        self,
        locus: str,
        phasing_result: PhasingResult | None = None,
        kmer_result: KmerValidationResult | None = None,
        assembly_result: AssemblyResult | None = None,
        reads_explained: int = 0,
        total_reads: int = 0,
    ) -> DetailedLocusQC:
        """Compute metrics for a single locus from available evidence."""
        qc = DetailedLocusQC(locus=locus)
        issues: list[str] = []

        # Read evidence
        qc.total_reads = total_reads
        qc.reads_explained = reads_explained

        # Haplotype balance from phasing
        if phasing_result and phasing_result.is_phased:
            qc.is_phased = True
            qc.n_het_sites = phasing_result.n_het_sites
            qc.phase_confidence = phasing_result.phase_confidence
            qc.n_informative_positions = phasing_result.n_het_sites

            bin0_reads = len(phasing_result.bins[0].read_names)
            bin1_reads = len(phasing_result.bins[1].read_names)

            if max(bin0_reads, bin1_reads) > 0:
                qc.haplotype_balance = min(bin0_reads, bin1_reads) / max(bin0_reads, bin1_reads)
                qc.allele1_depth = float(bin0_reads)
                qc.allele2_depth = float(bin1_reads)
                qc.depth_ratio = qc.haplotype_balance

                if qc.haplotype_balance < self.min_balance_ratio:
                    issues.append(
                        f"Haplotype imbalance: {qc.haplotype_balance:.2f} "
                        f"(min {self.min_balance_ratio})"
                    )

            # Informative position coverage
            total_sites = phasing_result.n_het_sites
            if total_sites > 0:
                covered = sum(
                    1 for site in phasing_result.het_sites
                    if site.depth >= 5
                )
                qc.informative_position_coverage = covered / total_sites
            else:
                issues.append("No heterozygous sites detected")

        elif phasing_result:
            qc.is_phased = False
            qc.n_het_sites = phasing_result.n_het_sites
            if phasing_result.n_het_sites == 0:
                pass  # Possibly homozygous — not necessarily an issue
            else:
                issues.append(
                    f"Phasing failed despite {phasing_result.n_het_sites} het sites"
                )

        # K-mer metrics
        if kmer_result:
            qc.kmer_coverage = kmer_result.proportion_kmers_covered
            qc.kmer_uniformity = kmer_result.kmer_coverage_uniformity
            qc.kmer_unexpected_fraction = kmer_result.unexpected_kmer_fraction

            if qc.kmer_coverage < self.min_kmer_coverage:
                issues.append(
                    f"Low k-mer coverage: {qc.kmer_coverage:.2%} "
                    f"(min {self.min_kmer_coverage:.0%})"
                )

        # Assembly metrics
        if assembly_result and assembly_result.contigs:
            qc.assembly_n_contigs = assembly_result.n_contigs
            qc.assembly_total_length = assembly_result.total_length

            contig_lengths = sorted(
                [len(s) for s in assembly_result.contigs.values()],
                reverse=True,
            )
            if contig_lengths:
                cumsum = np.cumsum(contig_lengths)
                half = cumsum[-1] / 2
                qc.assembly_n50 = int(contig_lengths[
                    int(np.searchsorted(cumsum, half))
                ])

            # Gene coverage estimate (typical HLA gene ~3-5kb)
            typical_gene_length = 4000
            qc.assembly_gene_coverage = min(
                1.0, qc.assembly_total_length / typical_gene_length,
            )

        # QC verdict
        qc.qc_issues = issues
        qc.qc_pass = len(issues) == 0

        return qc

    def calculate_all(
        self,
        loci: list[str],
        phasing_results: dict[str, PhasingResult],
        kmer_results: dict[str, KmerValidationResult],
        assembly_results: dict[str, AssemblyResult],
        reads_per_locus: dict[str, tuple[int, int]],
    ) -> dict[str, DetailedLocusQC]:
        """Compute metrics for all loci."""
        results = {}
        for locus in loci:
            explained, total = reads_per_locus.get(locus, (0, 0))
            results[locus] = self.calculate(
                locus=locus,
                phasing_result=phasing_results.get(locus),
                kmer_result=kmer_results.get(locus),
                assembly_result=assembly_results.get(locus),
                reads_explained=explained,
                total_reads=total,
            )
        return results
