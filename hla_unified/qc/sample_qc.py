"""Sample-level QC: allele dropout detection and contamination screening.

Allele dropout: distinguish true homozygosity from one allele failing
to sequence. Uses haplotype balance and depth-based statistics.

Contamination: detect mixed samples by screening for evidence of >2
alleles at a locus (indicative of DNA from multiple individuals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..genotyper.read_matrix import ReadAlleleMatrix
from ..genotyper.ilp_solver import ILPResult
from ..confidence.vb_estimator import ConfidenceResult
from ..phasing.haplotype_binner import PhasingResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allele Dropout Detection
# ---------------------------------------------------------------------------

@dataclass
class DropoutAssessment:
    """Assessment of whether a homozygous call is real or due to dropout."""
    locus: str
    is_homozygous_call: bool
    dropout_risk: str           # NONE, LOW, MEDIUM, HIGH
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class AlleleDropoutDetector:
    """Detects potential allele dropout at homozygous loci.

    A true homozygous locus should have:
    - Uniform read depth (no strand/position bias)
    - Dosage ~1.0 for the called allele
    - No het sites in phasing

    Allele dropout should show:
    - Lower than expected depth (one allele missing)
    - Possibly some het sites with very low minor AF
    - Dosage ~1.0 but with low overall depth
    """

    def __init__(
        self,
        min_depth_ratio: float = 0.6,
        expected_het_rate: float = 0.15,
    ) -> None:
        self.min_depth_ratio = min_depth_ratio
        self.expected_het_rate = expected_het_rate

    def assess(
        self,
        locus: str,
        ilp_result: ILPResult | None = None,
        vb_result: ConfidenceResult | None = None,
        phasing_result: PhasingResult | None = None,
        locus_reads: int = 0,
        reference_locus_reads: int = 0,
    ) -> DropoutAssessment:
        """Assess dropout risk for a single locus."""
        evidence: dict[str, Any] = {}

        if not ilp_result or not ilp_result.is_homozygous:
            return DropoutAssessment(
                locus=locus,
                is_homozygous_call=False,
                dropout_risk="NONE",
                evidence=evidence,
                recommendation="Heterozygous call — dropout not applicable.",
            )

        evidence["is_homozygous_call"] = True
        risk_signals = 0

        # Signal 1: Depth ratio vs reference locus
        if reference_locus_reads > 0:
            depth_ratio = locus_reads / reference_locus_reads
            evidence["depth_ratio_vs_reference"] = round(depth_ratio, 3)
            # For true homo, depth should be ~same as reference
            # For dropout, depth ~50% of reference (one allele missing)
            if depth_ratio < self.min_depth_ratio:
                risk_signals += 1
                evidence["low_depth_flag"] = True

        # Signal 2: Het sites present despite homozygous call
        if phasing_result and phasing_result.n_het_sites > 0:
            evidence["het_sites_in_homo_call"] = phasing_result.n_het_sites
            # Het sites in a "homozygous" call is suspicious
            if phasing_result.n_het_sites >= 2:
                risk_signals += 2  # strong signal
            elif phasing_result.n_het_sites == 1:
                risk_signals += 1

        # Signal 3: VB dosage should be ~1.0 for true homo
        if vb_result:
            dosage_imbalance = abs(vb_result.allele1_dosage - vb_result.allele2_dosage)
            evidence["dosage_imbalance"] = round(dosage_imbalance, 3)
            # For true homo, both dosages should be ~0.5 (same allele)
            # This signal is less informative for dropout

        # Signal 4: Low posterior despite homozygous call
        if vb_result and vb_result.posterior_prob < 0.90:
            risk_signals += 1
            evidence["low_posterior"] = round(vb_result.posterior_prob, 3)

        # Classify risk
        if risk_signals >= 3:
            risk = "HIGH"
            rec = (
                "High dropout risk: consider re-sequencing or checking "
                "for library preparation issues. Het sites detected in "
                "a homozygous call with low depth ratio."
            )
        elif risk_signals >= 2:
            risk = "MEDIUM"
            rec = (
                "Moderate dropout risk: homozygous call may reflect "
                "allele dropout rather than true homozygosity. "
                "Higher depth or family data could resolve."
            )
        elif risk_signals >= 1:
            risk = "LOW"
            rec = (
                "Minor dropout concern: one weak signal detected. "
                "Likely true homozygous but monitor."
            )
        else:
            risk = "NONE"
            rec = "No dropout signals — likely true homozygous."

        return DropoutAssessment(
            locus=locus,
            is_homozygous_call=True,
            dropout_risk=risk,
            evidence=evidence,
            recommendation=rec,
        )


# ---------------------------------------------------------------------------
# Contamination Detection
# ---------------------------------------------------------------------------

@dataclass
class ContaminationResult:
    """Sample-level contamination screening result."""
    is_contaminated: bool
    contamination_score: float  # 0.0 = clean, 1.0 = heavily contaminated
    loci_with_extra_alleles: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class ContaminationDetector:
    """Detects sample contamination by screening for >2 alleles at loci.

    In a clean diploid sample, each locus should have at most 2 distinct
    alleles. If 3+ alleles have substantial read support, the sample may
    contain DNA from multiple individuals.
    """

    def __init__(
        self,
        min_third_allele_fraction: float = 0.10,
        min_reads_for_detection: int = 20,
        min_loci_flagged: int = 2,
    ) -> None:
        self.min_third_allele_fraction = min_third_allele_fraction
        self.min_reads_for_detection = min_reads_for_detection
        self.min_loci_flagged = min_loci_flagged

    def screen(
        self,
        loci: list[str],
        matrices: dict[str, ReadAlleleMatrix],
        ilp_results: dict[str, ILPResult],
    ) -> ContaminationResult:
        """Screen for contamination across all loci."""
        flagged_loci: list[str] = []
        evidence: dict[str, Any] = {}

        for locus in loci:
            matrix = matrices.get(locus)
            ilp = ilp_results.get(locus)

            if matrix is None or ilp is None:
                continue
            if matrix.n_reads < self.min_reads_for_detection:
                continue

            # Count reads per allele
            reads_per_allele = (matrix.matrix > 0).sum(axis=0)
            total_reads = matrix.n_reads

            if total_reads == 0:
                continue

            # Sort alleles by read count (descending)
            sorted_idx = np.argsort(-reads_per_allele)
            fractions = reads_per_allele[sorted_idx] / total_reads

            # Check if 3rd allele has substantial support
            if len(fractions) >= 3 and fractions[2] >= self.min_third_allele_fraction:
                flagged_loci.append(locus)
                evidence[locus] = {
                    "top_3_fractions": [
                        round(float(fractions[i]), 3) for i in range(min(3, len(fractions)))
                    ],
                    "top_3_alleles": [
                        matrix.allele_names[sorted_idx[i]]
                        for i in range(min(3, len(sorted_idx)))
                    ],
                    "third_allele_fraction": round(float(fractions[2]), 3),
                }

        is_contaminated = len(flagged_loci) >= self.min_loci_flagged
        score = len(flagged_loci) / max(len(loci), 1)

        if is_contaminated:
            rec = (
                f"CONTAMINATION WARNING: {len(flagged_loci)} loci show "
                f"evidence of >2 alleles ({', '.join(flagged_loci)}). "
                f"Sample may contain mixed DNA. Verify sample identity "
                f"and consider re-extraction."
            )
        else:
            rec = "No contamination signals detected."

        return ContaminationResult(
            is_contaminated=is_contaminated,
            contamination_score=round(score, 3),
            loci_with_extra_alleles=flagged_loci,
            evidence=evidence,
            recommendation=rec,
        )
