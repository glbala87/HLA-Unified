"""BWA/minimap2 wrapper for read mapping.

Maps extracted FASTQ reads against the extended reference genome
or PRG reference using BWA-MEM or minimap2.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class BWAMapper:
    """Wrapper for BWA-MEM and minimap2 read mapping."""

    def __init__(self, bwa_bin: str = "bwa", samtools_bin: str = "samtools",
                 minimap2_bin: str = "minimap2") -> None:
        self.bwa_bin = bwa_bin
        self.samtools_bin = samtools_bin
        self.minimap2_bin = minimap2_bin

    def index(self, reference: str | Path) -> None:
        """Index a reference genome with BWA."""
        ref = str(reference)
        cmd = [self.bwa_bin, "index", ref]
        logger.info(f"Indexing reference: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def map_paired(self, reference: str | Path, fastq1: str | Path,
                   fastq2: str | Path, output_bam: str | Path,
                   threads: int = 1) -> Path:
        """Map paired-end reads with BWA-MEM.

        Returns path to sorted BAM file.
        """
        ref = str(reference)
        out = Path(output_bam)

        bwa_cmd = [self.bwa_bin, "mem", "-t", str(threads), ref,
                   str(fastq1), str(fastq2)]
        sort_cmd = [self.samtools_bin, "sort", "-@", str(threads),
                    "-o", str(out)]

        logger.info(f"Mapping paired reads: {fastq1}, {fastq2}")
        bwa_proc = subprocess.Popen(bwa_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sort_proc = subprocess.Popen(sort_cmd, stdin=bwa_proc.stdout,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bwa_proc.stdout.close()
        sort_out, sort_err = sort_proc.communicate()

        if sort_proc.returncode != 0:
            raise RuntimeError(f"BWA/samtools sort failed: {sort_err.decode()}")

        # Index the BAM
        subprocess.run([self.samtools_bin, "index", str(out)], check=True)
        return out

    def map_long_reads(self, reference: str | Path, fastq: str | Path,
                       output_bam: str | Path, technology: str = "ont2d",
                       threads: int = 1) -> Path:
        """Map long reads with minimap2.

        Args:
            technology: 'ont2d' for Oxford Nanopore, 'pacbio' for PacBio
        """
        ref = str(reference)
        out = Path(output_bam)
        preset = "map-ont" if technology == "ont2d" else "map-pb"

        mm2_cmd = [self.minimap2_bin, "-a", "-x", preset,
                   "-t", str(threads), ref, str(fastq)]
        sort_cmd = [self.samtools_bin, "sort", "-@", str(threads),
                    "-o", str(out)]

        logger.info(f"Mapping long reads ({technology}): {fastq}")
        mm2_proc = subprocess.Popen(mm2_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sort_proc = subprocess.Popen(sort_cmd, stdin=mm2_proc.stdout,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mm2_proc.stdout.close()
        sort_out, sort_err = sort_proc.communicate()

        if sort_proc.returncode != 0:
            raise RuntimeError(f"minimap2/samtools failed: {sort_err.decode()}")

        subprocess.run([self.samtools_bin, "index", str(out)], check=True)
        return out

    def map_unpaired(self, reference: str | Path, fastq: str | Path,
                     output_bam: str | Path, threads: int = 1) -> Path:
        """Map unpaired reads with BWA-MEM."""
        ref = str(reference)
        out = Path(output_bam)

        bwa_cmd = [self.bwa_bin, "mem", "-t", str(threads), ref, str(fastq)]
        sort_cmd = [self.samtools_bin, "sort", "-@", str(threads),
                    "-o", str(out)]

        logger.info(f"Mapping unpaired reads: {fastq}")
        bwa_proc = subprocess.Popen(bwa_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sort_proc = subprocess.Popen(sort_cmd, stdin=bwa_proc.stdout,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bwa_proc.stdout.close()
        sort_out, sort_err = sort_proc.communicate()

        if sort_proc.returncode != 0:
            raise RuntimeError(f"BWA/samtools sort failed: {sort_err.decode()}")

        subprocess.run([self.samtools_bin, "index", str(out)], check=True)
        return out
