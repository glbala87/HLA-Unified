"""Known reference detection: match BAM contigs to reference specifications.

Translates the reference matching logic from HLA-LA.pl.
Reads knownReferences/*.txt files and matches against BAM idxstats
to auto-detect which reference genome was used for alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_reference_file(filepath: Path) -> tuple[list[dict], bool]:
    """Parse a knownReferences/*.txt file.

    Each file specifies contigs and extraction coordinates for one
    reference genome.

    Returns:
        (contig_specs, has_unmapped): list of contig specifications
        and whether unmapped reads should be extracted.
    """
    contigs: list[dict] = []
    has_unmapped = False

    with open(filepath) as fh:
        header = fh.readline().strip().split("\t")
        expected = ["contigID", "contigLength", "ExtractCompleteContig",
                    "PartialExtraction_Start", "PartialExtraction_Stop"]
        if header != expected:
            raise ValueError(f"Unexpected header in {filepath}: {header}")

        for line in fh:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                fields.extend([""] * (5 - len(fields)))

            contig = {
                "contig_id": fields[0],
                "contig_length": int(fields[1]) if fields[1] else 0,
                "extract_complete": bool(fields[2]),
                "partial_start": int(fields[3]) if fields[3] else 0,
                "partial_stop": int(fields[4]) if fields[4] else 0,
            }

            if contig["contig_id"] == "*" and contig["extract_complete"]:
                has_unmapped = True
            else:
                contigs.append(contig)

    return contigs, has_unmapped


def detect_reference(bam_path: str | Path, graph_dir: str | Path,
                     additional_refs_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Auto-detect which reference specification matches the BAM file.

    Reads BAM idxstats and compares against all known reference files
    in the graph's knownReferences directory.

    Args:
        bam_path: path to the BAM file
        graph_dir: path to the graph directory
        additional_refs_dir: optional additional references directory

    Returns:
        Path to the matching reference file, or None.
    """
    import pysam

    bam_path = Path(bam_path)
    graph_dir = Path(graph_dir)

    # Get BAM idxstats
    bam_contigs: dict[str, int] = {}
    contig_order: list[str] = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for entry in bam.get_index_statistics():
            bam_contigs[entry.contig] = bam.get_reference_length(entry.contig)
            contig_order.append(entry.contig)

    # Collect reference files
    ref_dirs = [graph_dir / "knownReferences"]
    if additional_refs_dir:
        ref_dirs.append(Path(additional_refs_dir))

    ref_files: list[Path] = []
    for d in ref_dirs:
        if d.exists():
            ref_files.extend(d.glob("*.txt"))

    # Find matching reference
    compatible: list[Path] = []
    for ref_file in ref_files:
        try:
            contigs, _ = parse_reference_file(ref_file)
        except (ValueError, OSError) as e:
            logger.debug(f"Skipping {ref_file}: {e}")
            continue

        n_total = len(contigs)
        n_matching = 0
        for spec in contigs:
            cid = spec["contig_id"]
            if cid in bam_contigs and bam_contigs[cid] == spec["contig_length"]:
                n_matching += 1

        if n_matching == n_total and n_total == len(bam_contigs):
            compatible.append(ref_file)

    if len(compatible) == 0:
        logger.error("No compatible reference specification found")
        return None
    if len(compatible) > 1:
        logger.warning(f"Multiple compatible references found: {compatible}")

    return compatible[0]


def get_extraction_regions(ref_file: Path) -> tuple[list[str], bool]:
    """Get the samtools-compatible extraction regions from a reference file.

    Returns:
        (regions, extract_unmapped): list of region strings like
        "chr6:28510120-33480577" and whether to extract unmapped reads.
    """
    contigs, has_unmapped = parse_reference_file(ref_file)
    regions: list[str] = []

    for spec in contigs:
        cid = spec["contig_id"]
        if spec["extract_complete"]:
            regions.append(cid)
        elif spec["partial_start"] and spec["partial_stop"]:
            regions.append(f"{cid}:{spec['partial_start']}-{spec['partial_stop']}")

    return regions, has_unmapped
