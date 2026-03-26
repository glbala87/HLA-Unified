"""Formal configuration schema for HLA-Unified V2.

Centralizes all pipeline parameters into validated, composable dataclasses.
Every hardcoded threshold from V1 is now a configurable field with documented
defaults and assay-aware overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Locus configuration
# ---------------------------------------------------------------------------

@dataclass
class LocusConfig:
    """Configuration for a single HLA locus."""
    name: str
    gene_class: Literal["I", "II"]
    typing_exons: list[int]
    supported_resolutions: dict[str, int] = field(default_factory=dict)
    # Per-build MHC region coordinates: build -> (chr, start, end)
    mhc_region: dict[str, tuple[str, int, int]] = field(default_factory=dict)


# Default locus definitions (mirrors reference/loci.py but validated)
DEFAULT_LOCI: dict[str, LocusConfig] = {
    "A": LocusConfig("A", "I", [2, 3]),
    "B": LocusConfig("B", "I", [2, 3]),
    "C": LocusConfig("C", "I", [2, 3]),
    "E": LocusConfig("E", "I", [2, 3]),
    "F": LocusConfig("F", "I", [2, 3]),
    "G": LocusConfig("G", "I", [2, 3]),
    "DRB1": LocusConfig("DRB1", "II", [2]),
    "DRB3": LocusConfig("DRB3", "II", [2]),
    "DRB4": LocusConfig("DRB4", "II", [2]),
    "DRB5": LocusConfig("DRB5", "II", [2]),
    "DQA1": LocusConfig("DQA1", "II", [2]),
    "DQB1": LocusConfig("DQB1", "II", [2]),
    "DPA1": LocusConfig("DPA1", "II", [2]),
    "DPB1": LocusConfig("DPB1", "II", [2]),
    "DRA": LocusConfig("DRA", "II", [2]),
}


# ---------------------------------------------------------------------------
# Assay preset
# ---------------------------------------------------------------------------

@dataclass
class AssayPreset:
    """Assay-specific parameter preset.

    Encapsulates all parameters that vary by sequencing technology:
    alignment, filtering, QC thresholds, and assembly settings.
    """
    name: str
    description: str
    # Alignment
    minimap2_preset: str = "sr"
    bowtie2_params: list[str] = field(default_factory=lambda: ["--very-sensitive"])
    # Coverage
    expected_coverage: int = 30
    min_coverage: int = 5
    # Read extraction
    extract_unmapped: bool = True
    focus_exons_only: bool = False
    # Candidate filtering
    max_prefilter_candidates: int = 80
    max_refined_candidates: int = 20
    # Confidence
    confidence_threshold: float = 0.90
    assembly_confidence_trigger: float = 0.80
    # Phasing
    phasing_min_het_depth: int = 5
    phasing_min_minor_af: float = 0.15
    phasing_min_het_sites: int = 2
    # K-mer validation
    kmer_k: int = 31
    kmer_min_coverage: float = 0.90
    kmer_max_unexpected: float = 0.30
    # Assembly
    assembler: str = "megahit"
    assembly_timeout: int = 300
    assembly_genome_size: str = "15k"
    # Informative positions
    min_informative_positions: int = 5


# ---------------------------------------------------------------------------
# Use-case profile
# ---------------------------------------------------------------------------

@dataclass
class UseCaseProfile:
    """Use-case-specific profile that overrides defaults for a workflow context.

    Profiles encode domain-specific requirements: transplant typing demands
    conservative HIGH-confidence calls at classical loci; research prioritizes
    full-resolution discovery across all loci; immuno-oncology needs fast
    Class I calls for neoantigen prediction.
    """
    name: str
    description: str
    loci: list[str]
    min_confidence: str = "HIGH"       # minimum acceptable confidence tier
    require_phasing: bool = False
    output_resolution: int | str = 2   # field count or "max" / "G"
    report_novel_alleles: bool = True
    clinical_summary: bool = False
    strict_reproducibility: bool = False


# ---------------------------------------------------------------------------
# Pipeline configuration (top-level)
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Complete pipeline configuration — the single source of truth.

    Constructed from CLI flags, profile overrides, and assay presets.
    Passed to UnifiedPipeline instead of individual keyword arguments.
    """
    # Input
    imgt_db_path: str | Path = ""
    work_dir: str | Path = ""
    threads: int = 4
    # Loci
    loci: list[str] = field(default_factory=lambda: [
        "A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1",
    ])
    loci_config: dict[str, LocusConfig] = field(default_factory=lambda: dict(DEFAULT_LOCI))
    # Assay
    data_type: str = "short"
    assay: AssayPreset | None = None
    # Output
    output_resolution: int | str = "max"
    # IMGT version control
    imgt_release: str | None = None
    strict_reproducibility: bool = False
    # Phase toggles
    skip_refinement: bool = False
    skip_confidence: bool = False
    skip_kmer: bool = False
    skip_assembly: bool = False
    # Use-case profile
    profile: UseCaseProfile | None = None
    # Clinical
    clinical_summary: bool = False
    # Novel allele
    report_novel_alleles: bool = True

    def effective_assay(self) -> AssayPreset:
        """Return the assay preset, falling back to a built-in default."""
        if self.assay is not None:
            return self.assay
        from .presets import ASSAY_PRESETS
        return ASSAY_PRESETS.get(self.data_type, ASSAY_PRESETS["short"])

    def effective_loci(self) -> list[str]:
        """Return the effective locus list (profile may override)."""
        if self.profile and self.profile.loci:
            return self.profile.loci
        return self.loci

    def effective_resolution(self) -> int | str:
        """Return effective output resolution (profile may override)."""
        if self.profile and self.profile.output_resolution:
            return self.profile.output_resolution
        return self.output_resolution

    @classmethod
    def from_cli(
        cls,
        *,
        imgt_db: str,
        out: str,
        threads: int = 4,
        loci: list[str] | None = None,
        data_type: str = "short",
        output_resolution: int | str = "max",
        imgt_release: str | None = None,
        strict_reproducibility: bool = False,
        skip_refinement: bool = False,
        skip_confidence: bool = False,
        skip_kmer: bool = False,
        skip_assembly: bool = False,
        profile_name: str | None = None,
        clinical: bool = False,
        max_candidates: int = 80,
    ) -> PipelineConfig:
        """Factory: build a PipelineConfig from CLI flags."""
        from .presets import ASSAY_PRESETS, USE_CASE_PROFILES

        assay = ASSAY_PRESETS.get(data_type, ASSAY_PRESETS["short"])
        # Override max candidates if set explicitly
        if max_candidates != 80:
            assay.max_prefilter_candidates = max_candidates

        profile = USE_CASE_PROFILES.get(profile_name) if profile_name else None

        # Profile can force clinical mode and strict reproducibility
        if profile:
            clinical = clinical or profile.clinical_summary
            strict_reproducibility = strict_reproducibility or profile.strict_reproducibility

        return cls(
            imgt_db_path=imgt_db,
            work_dir=out,
            threads=threads,
            loci=loci or ["A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1"],
            data_type=data_type,
            assay=assay,
            output_resolution=output_resolution,
            imgt_release=imgt_release,
            strict_reproducibility=strict_reproducibility,
            skip_refinement=skip_refinement,
            skip_confidence=skip_confidence,
            skip_kmer=skip_kmer,
            skip_assembly=skip_assembly,
            profile=profile,
            clinical_summary=clinical,
        )
