"""Fast pre-filter: minimap2/bowtie2-based candidate reduction (xHLA-style).

Maps reads against the full IMGT/HLA reference to quickly identify
which alleles have read support. Reduces candidate space from
thousands to ~50-100 per locus before expensive genotyping.

Supports short reads (Illumina), long reads (PacBio/ONT), and RNA-seq.
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pysam

from ..reference.loci import ALL_TYPING_LOCI, parse_allele_name
from ..utils.external import run_cmd, run_pipeline, check_tool, ToolError

logger = logging.getLogger(__name__)

# minimap2 presets for different data types
MINIMAP2_PRESETS = {
    "short": "sr",              # Illumina WGS short reads
    "exome": "sr",              # Illumina WES (same aligner, different extraction)
    "targeted_capture": "sr",   # Targeted HLA capture panels
    "pacbio": "map-pb",         # PacBio CLR
    "hifi": "map-hifi",         # PacBio HiFi (CCS)
    "ont": "map-ont",           # Oxford Nanopore
    "rna": "splice",            # RNA-seq (splice-aware)
}

# Assay-specific configurations
ASSAY_CONFIGS = {
    "short": {
        "description": "Illumina WGS (whole-genome short reads)",
        "extract_unmapped": True,
        "min_coverage": 10,
        "expected_coverage": 30,
    },
    "exome": {
        "description": "Illumina WES (exome capture)",
        "extract_unmapped": True,
        "min_coverage": 20,
        "expected_coverage": 100,
        "focus_exons_only": True,  # exome may only cover coding regions
    },
    "targeted_capture": {
        "description": "Targeted HLA capture panel (deep coverage)",
        "extract_unmapped": False,  # capture panels have minimal off-target
        "min_coverage": 50,
        "expected_coverage": 500,
        "focus_exons_only": False,  # capture panels cover full HLA genes
    },
    "pacbio": {
        "description": "PacBio CLR long reads",
        "extract_unmapped": True,
        "min_coverage": 5,
        "expected_coverage": 15,
    },
    "hifi": {
        "description": "PacBio HiFi (CCS) long reads",
        "extract_unmapped": True,
        "min_coverage": 5,
        "expected_coverage": 20,
    },
    "ont": {
        "description": "Oxford Nanopore long reads",
        "extract_unmapped": True,
        "min_coverage": 5,
        "expected_coverage": 20,
    },
    "rna": {
        "description": "RNA-seq (splice-aware alignment)",
        "extract_unmapped": True,
        "min_coverage": 5,
        "expected_coverage": 50,
        "focus_exons_only": True,
    },
}


@dataclass
class PrefilterResult:
    """Result of fast pre-filtering for one locus."""
    locus: str
    candidate_alleles: list[str]
    read_counts: dict[str, int]
    total_reads_mapped: int = 0


@dataclass
class PrefilterResults:
    """Aggregated pre-filter results across all loci."""
    per_locus: dict[str, PrefilterResult] = field(default_factory=dict)
    total_input_reads: int = 0
    total_mapped_reads: int = 0


class FastPrefilter:
    """xHLA-style fast candidate reduction using minimap2.

    Strategy:
    1. Build minimap2 index of IMGT/HLA allele sequences
    2. Map extracted reads with permissive parameters
    3. Count reads per allele, keep top candidates per locus
    """

    def __init__(
        self,
        mapper: str = "minimap2",
        threads: int = 4,
        max_candidates_per_locus: int = 80,
        min_read_fraction: float = 0.01,
        data_type: str = "short",
    ) -> None:
        self.mapper = check_tool(mapper)
        self.threads = threads
        self.max_candidates = max_candidates_per_locus
        self.min_read_fraction = min_read_fraction

        if data_type not in MINIMAP2_PRESETS:
            raise ValueError(
                f"Unknown data_type '{data_type}'. "
                f"Supported: {list(MINIMAP2_PRESETS)}"
            )
        self.data_type = data_type
        self.preset = MINIMAP2_PRESETS[data_type]

    def build_index(self, reference_fasta: Path, output_idx: Path) -> Path:
        """Build minimap2 index for the HLA reference panel."""
        logger.info("Building minimap2 index: %s", output_idx)
        run_cmd(
            [self.mapper, "-d", str(output_idx), str(reference_fasta)],
            description="build minimap2 index",
        )
        return output_idx

    def run(
        self,
        r1_fastq: Path,
        r2_fastq: Path | None,
        reference: Path,
        work_dir: Path,
        loci: list[str] | None = None,
    ) -> PrefilterResults:
        """Run fast pre-filtering.

        Maps reads to IMGT/HLA reference, counts hits per allele,
        returns top candidates per locus.
        """
        if loci is None:
            loci = list(ALL_TYPING_LOCI)

        work_dir.mkdir(parents=True, exist_ok=True)
        output_bam = work_dir / "prefilter.bam"

        # Map reads with minimap2
        cmd = [
            self.mapper,
            "-a",
            "-x", self.preset,
            "-t", str(self.threads),
            "--secondary=yes",
            "-N", "50",
            str(reference),
        ]

        # Paired-end only for short reads
        if r2_fastq and r2_fastq.exists() and self.data_type == "short":
            cmd.extend([str(r1_fastq), str(r2_fastq)])
        else:
            cmd.append(str(r1_fastq))

        logger.info("Pre-filter alignment (%s mode): %s",
                     self.data_type, " ".join(cmd[:6]))

        # Pipe minimap2 | samtools sort -> BAM
        sort_cmd = ["samtools", "sort", "-@", str(self.threads), "-"]
        run_pipeline(
            [cmd, sort_cmd],
            output_path=output_bam,
            description="prefilter alignment",
        )

        # Index the BAM
        run_cmd(
            ["samtools", "index", str(output_bam)],
            description="index prefilter BAM",
        )

        # Count reads per allele
        return self._count_and_filter(output_bam, loci)

    def _count_and_filter(
        self, bam_path: Path, loci: list[str]
    ) -> PrefilterResults:
        """Count reads per allele and select top candidates per locus."""
        allele_counts: Counter[str] = Counter()
        total_reads = 0

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped or read.mapping_quality < 1:
                    continue
                total_reads += 1
                ref_name = read.reference_name
                if ref_name:
                    allele_counts[ref_name] += 1

        # Group alleles by locus
        locus_alleles: dict[str, Counter[str]] = defaultdict(Counter)
        for allele, count in allele_counts.items():
            info = parse_allele_name(allele)
            if info.locus in loci:
                locus_alleles[info.locus][allele] = count

        results = PrefilterResults(
            total_input_reads=total_reads,
            total_mapped_reads=total_reads,
        )

        for locus in loci:
            counts = locus_alleles.get(locus, Counter())
            if not counts:
                results.per_locus[locus] = PrefilterResult(
                    locus=locus, candidate_alleles=[], read_counts={},
                )
                continue

            total_locus = sum(counts.values())
            min_reads = max(2, int(total_locus * self.min_read_fraction))

            filtered = {a: c for a, c in counts.items() if c >= min_reads}
            top = sorted(filtered.items(), key=lambda x: -x[1])
            top = top[:self.max_candidates]

            results.per_locus[locus] = PrefilterResult(
                locus=locus,
                candidate_alleles=[a for a, _ in top],
                read_counts=dict(top),
                total_reads_mapped=total_locus,
            )

        logger.info(
            "Pre-filter: %d total mapped reads across %d loci",
            total_reads, len(loci),
        )
        for locus in loci:
            r = results.per_locus.get(locus)
            if r:
                logger.info(
                    "  %s: %d candidates (%d reads)",
                    locus, len(r.candidate_alleles), r.total_reads_mapped,
                )

        return results
