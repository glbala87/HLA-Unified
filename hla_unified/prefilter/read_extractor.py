"""Extract HLA-relevant reads from BAM/CRAM/FASTQ input.

Supports multiple input types:
- WGS BAM/CRAM: extract reads from MHC region + unmapped reads
- WES BAM: extract reads from captured HLA regions
- RNA-seq BAM: extract reads mapping to HLA genes (splice-aware)
- FASTQ: pass through directly (no extraction needed)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..utils.external import run_cmd, run_pipeline, check_tool, ToolError

logger = logging.getLogger(__name__)


class ReadExtractor:
    """Extracts HLA-relevant reads from alignment files."""

    def __init__(self, samtools_bin: str = "samtools",
                 threads: int = 4) -> None:
        self.samtools = check_tool(samtools_bin)
        self.threads = threads

    def detect_reference(self, bam_path: str) -> str | None:
        """Auto-detect reference genome build from BAM header."""
        result = run_cmd(
            [self.samtools, "idxstats", bam_path],
            description="detect reference genome",
        )
        contigs = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2:
                contigs[parts[0]] = int(parts[1])

        # Detect by chr6 length
        for name in ["chr6", "6"]:
            length = contigs.get(name, 0)
            if 170_000_000 < length < 172_000_000:
                if name == "chr6":
                    return "GRCh38"
                return "GRCh37"

        logger.warning(
            "Could not auto-detect reference build. "
            "Found contigs: %s", list(contigs.keys())[:10]
        )
        return None

    def extract_mhc_reads(
        self,
        bam_path: str,
        output_dir: Path,
        region: str = "chr6:28510120-33480577",
        include_unmapped: bool = True,
    ) -> tuple[Path, Path]:
        """Extract reads from MHC region, return (R1.fastq.gz, R2.fastq.gz).

        Two-step extraction:
        1. samtools view to pull MHC reads (+ unmapped if requested)
        2. samtools fastq to split into paired FASTQ
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_bam = output_dir / "mhc_extracted.bam"

        # Step 1a: Extract MHC-mapping reads
        run_cmd(
            [self.samtools, "view", "-b",
             "-@", str(self.threads),
             "-o", str(extracted_bam),
             bam_path, region],
            description=f"extract MHC region ({region})",
        )

        # Step 1b: Also grab unmapped reads and merge
        if include_unmapped:
            unmapped_bam = output_dir / "unmapped.bam"
            run_cmd(
                [self.samtools, "view", "-b", "-f", "4",
                 "-@", str(self.threads),
                 "-o", str(unmapped_bam),
                 bam_path],
                description="extract unmapped reads",
            )

            merged_bam = output_dir / "mhc_plus_unmapped.bam"
            run_cmd(
                [self.samtools, "merge", "-f",
                 "-@", str(self.threads),
                 str(merged_bam),
                 str(extracted_bam),
                 str(unmapped_bam)],
                description="merge MHC + unmapped reads",
            )
            extracted_bam = merged_bam

        # Step 2: Sort by name for paired FASTQ extraction
        namesorted = output_dir / "namesorted.bam"
        run_cmd(
            [self.samtools, "sort", "-n",
             "-@", str(self.threads),
             "-o", str(namesorted),
             str(extracted_bam)],
            description="name-sort extracted BAM",
        )

        # Step 3: Convert to FASTQ
        r1 = output_dir / "R1.fastq.gz"
        r2 = output_dir / "R2.fastq.gz"
        singleton = output_dir / "singleton.fastq.gz"

        run_cmd(
            [self.samtools, "fastq",
             "-@", str(self.threads),
             "-1", str(r1),
             "-2", str(r2),
             "-s", str(singleton),
             "-0", "/dev/null",
             str(namesorted)],
            description="convert BAM to FASTQ",
        )

        logger.info("Extracted FASTQ: %s, %s", r1, r2)
        return r1, r2

    def count_reads(self, fastq_path: Path) -> int:
        """Quick line-count based read count for FASTQ."""
        if not fastq_path.exists():
            return 0
        if str(fastq_path).endswith(".gz"):
            result = run_cmd(
                ["zcat", str(fastq_path)],
                description="count reads",
                capture=True,
            )
            return result.stdout.count("\n") // 4
        with open(fastq_path) as fh:
            return sum(1 for _ in fh) // 4
