"""CLI: Detect reference genome from BAM file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command("detect-reference")
@click.option("--bam", required=True, type=click.Path(exists=True),
              help="Input BAM/CRAM file")
@click.option("--graph-dir", required=True, type=click.Path(exists=True),
              help="Graph directory with knownReferences/")
@click.option("--additional-refs", type=click.Path(exists=True), default=None,
              help="Additional references directory")
@click.option("--output", "-o", default="-", help="Output file (default: stdout)")
def detect_reference(bam: str, graph_dir: str, additional_refs: str | None,
                     output: str) -> None:
    """Detect which reference genome was used for the input BAM.

    Compares BAM contig names and lengths against known reference
    specifications to identify the reference genome.
    """
    from ..reference.known_references import detect_reference as _detect, get_extraction_regions

    ref_file = _detect(bam, graph_dir, additional_refs)
    if ref_file is None:
        click.echo("ERROR: No compatible reference found", err=True)
        sys.exit(1)

    regions, has_unmapped = get_extraction_regions(ref_file)
    result = {
        "reference_file": str(ref_file),
        "regions": regions,
        "extract_unmapped": has_unmapped,
    }

    if output == "-":
        click.echo(json.dumps(result, indent=2))
    else:
        Path(output).write_text(json.dumps(result, indent=2))
