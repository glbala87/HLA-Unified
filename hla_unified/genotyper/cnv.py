"""Copy-number estimation for DRB3/DRB4/DRB5 loci.

These loci have variable copy number (0, 1, or 2 per haploid genome).
The standard diploid ILP model (exactly 2 alleles) is wrong for them.
This module estimates copy number from read depth and adjusts genotyping
accordingly.

Copy-number states:
  0 copies: gene absent on both haplotypes (no call)
  1 copy:   gene on one haplotype only (hemizygous — report 1 allele)
  2 copies: gene on both haplotypes (standard diploid — report 2 alleles)

Detection approach:
  1. Compare read depth at the CNV locus vs a stable reference locus (DRB1)
  2. Depth ratio ~ 0 → 0 copies, ~ 0.5 → 1 copy, ~ 1.0 → 2 copies
  3. Use k-mer evidence as secondary signal (low allele k-mer coverage → absent)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)

# Loci with variable copy number
CNV_LOCI = {"DRB3", "DRB4", "DRB5"}

# Stable reference locus (always 2 copies, high coverage)
REFERENCE_LOCUS = "DRB1"


class CopyNumber(IntEnum):
    ABSENT = 0
    HEMIZYGOUS = 1
    DIPLOID = 2


@dataclass
class CNVEstimate:
    """Copy-number estimate for one locus."""
    locus: str
    copy_number: CopyNumber
    depth_ratio: float          # depth(locus) / depth(reference)
    locus_reads: int
    reference_reads: int
    confidence: str             # HIGH, MEDIUM, LOW
    method: str                 # "depth_ratio", "kmer", "fallback"


class CopyNumberEstimator:
    """Estimates copy number for DRB3/4/5 from read depth ratios.

    Uses DRB1 as the reference (always diploid) and compares per-locus
    read counts. Thresholds calibrated for WGS at >=20x.
    """

    def __init__(
        self,
        absent_threshold: float = 0.10,
        hemizygous_range: tuple[float, float] = (0.10, 0.65),
        diploid_threshold: float = 0.65,
        min_reference_reads: int = 10,
    ) -> None:
        self.absent_threshold = absent_threshold
        self.hemizygous_range = hemizygous_range
        self.diploid_threshold = diploid_threshold
        self.min_reference_reads = min_reference_reads

    def estimate(
        self,
        locus: str,
        locus_reads: int,
        reference_reads: int,
        kmer_coverage: float | None = None,
    ) -> CNVEstimate:
        """Estimate copy number for a single locus."""
        if locus not in CNV_LOCI:
            return CNVEstimate(
                locus=locus,
                copy_number=CopyNumber.DIPLOID,
                depth_ratio=1.0,
                locus_reads=locus_reads,
                reference_reads=reference_reads,
                confidence="HIGH",
                method="not_cnv_locus",
            )

        # If reference locus has too few reads, can't estimate reliably
        if reference_reads < self.min_reference_reads:
            logger.warning(
                "%s: too few reference reads (%d) for CNV estimation, "
                "defaulting to diploid",
                locus, reference_reads,
            )
            return CNVEstimate(
                locus=locus,
                copy_number=CopyNumber.DIPLOID,
                depth_ratio=0.0,
                locus_reads=locus_reads,
                reference_reads=reference_reads,
                confidence="LOW",
                method="fallback",
            )

        depth_ratio = locus_reads / reference_reads

        # Primary: depth ratio classification
        if depth_ratio < self.absent_threshold:
            cn = CopyNumber.ABSENT
        elif depth_ratio < self.diploid_threshold:
            cn = CopyNumber.HEMIZYGOUS
        else:
            cn = CopyNumber.DIPLOID

        # Secondary: k-mer coverage corroboration
        method = "depth_ratio"
        if kmer_coverage is not None:
            if kmer_coverage < 0.10 and cn != CopyNumber.ABSENT:
                logger.info(
                    "%s: depth suggests %d copies but k-mer coverage is %.2f, "
                    "overriding to ABSENT",
                    locus, cn, kmer_coverage,
                )
                cn = CopyNumber.ABSENT
                method = "kmer_override"

        # Confidence
        if cn == CopyNumber.ABSENT and depth_ratio < 0.05:
            confidence = "HIGH"
        elif cn == CopyNumber.DIPLOID and depth_ratio > 0.80:
            confidence = "HIGH"
        elif cn == CopyNumber.HEMIZYGOUS:
            # Hemizygous is inherently harder to call
            low, high = self.hemizygous_range
            midpoint = (low + high) / 2
            if abs(depth_ratio - midpoint) < 0.10:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "MEDIUM"

        logger.info(
            "%s: CNV estimate = %d copies (depth_ratio=%.2f, "
            "reads=%d/%d, confidence=%s)",
            locus, cn, depth_ratio, locus_reads, reference_reads, confidence,
        )

        return CNVEstimate(
            locus=locus,
            copy_number=cn,
            depth_ratio=round(depth_ratio, 4),
            locus_reads=locus_reads,
            reference_reads=reference_reads,
            confidence=confidence,
            method=method,
        )

    def estimate_all(
        self,
        loci: list[str],
        reads_per_locus: dict[str, int],
        kmer_coverages: dict[str, float] | None = None,
    ) -> dict[str, CNVEstimate]:
        """Estimate copy number for all CNV loci."""
        reference_reads = reads_per_locus.get(REFERENCE_LOCUS, 0)
        results = {}

        for locus in loci:
            if locus not in CNV_LOCI:
                continue
            locus_reads = reads_per_locus.get(locus, 0)
            kmer_cov = kmer_coverages.get(locus) if kmer_coverages else None
            results[locus] = self.estimate(
                locus, locus_reads, reference_reads, kmer_cov,
            )

        return results


def adjust_ilp_for_cnv(
    cnv_estimate: CNVEstimate,
) -> int:
    """Return the number of alleles the ILP should select for this locus.

    Standard diploid: 2
    Hemizygous (1 copy): 1
    Absent (0 copies): 0
    """
    return int(cnv_estimate.copy_number)
