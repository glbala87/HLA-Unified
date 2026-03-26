"""CLI: Extract MHC reads from BAM file."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click


@click.command("extract-reads")
@click.option("--bam", required=True, type=click.Path(exists=True),
              help="Input BAM/CRAM file")
@click.option("--ref-info", required=True, type=click.Path(exists=True),
              help="Reference info JSON from detect-reference")
@click.option("--output", "-o", required=True, help="Output BAM path")
@click.option("--samtools", default="samtools", help="Path to samtools binary")
@click.option("--threads", default=1, type=int, help="Number of threads")
@click.option("--samtools-t", default=None, help="Reference FASTA for CRAM")
def extract_reads(bam: str, ref_info: str, output: str, samtools: str,
                  threads: int, samtools_t: str | None) -> None:
    """Extract reads from MHC and related regions.

    Uses samtools to extract mapped reads from specified regions
    and optionally unmapped reads. Merges and indexes the result.
    """
    info = json.loads(Path(ref_info).read_text())
    regions = info["regions"]
    extract_unmapped = info["extract_unmapped"]

    out_path = Path(output)
    mapped_bam = str(out_path.with_suffix(".mapped.bam"))

    t_switch = f"-T {samtools_t}" if samtools_t else ""
    threads_arg = threads - 1

    # Extract mapped reads from regions
    cmd = f"{samtools} view -@ {threads_arg} {t_switch} -bo {mapped_bam} {bam} {' '.join(regions)}"
    click.echo(f"Extracting reads from {len(regions)} regions...")
    subprocess.run(cmd, shell=True, check=True)

    if extract_unmapped:
        unmapped_bam = str(out_path.with_suffix(".unmapped.bam"))
        cmd_unmapped = (
            f"{samtools} view -@ {threads_arg} {t_switch} -f 4 -bo {unmapped_bam} {bam}"
        )
        click.echo("Extracting unmapped reads...")
        try:
            subprocess.run(cmd_unmapped, shell=True, check=True)
        except subprocess.CalledProcessError:
            click.echo("Warning: unmapped read extraction had issues, creating empty BAM")
            subprocess.run(
                f"{samtools} view -bo {unmapped_bam} -@ {threads_arg} {t_switch} {bam} '*'",
                shell=True, check=False,
            )

        # Merge
        cmd_merge = f"{samtools} merge {output} {mapped_bam} {unmapped_bam}"
        click.echo("Merging BAMs...")
        subprocess.run(cmd_merge, shell=True, check=True)
    else:
        import shutil
        shutil.move(mapped_bam, output)

    # Index
    subprocess.run(f"{samtools} index {output}", shell=True, check=True)
    click.echo(f"Output: {output}")
