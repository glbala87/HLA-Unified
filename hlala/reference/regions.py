"""MHC extraction coordinates and region definitions."""

from __future__ import annotations

# Default MHC region coordinates for common reference genomes
MHC_REGIONS = {
    "GRCh38": {
        "contig": "chr6",
        "start": 28510120,
        "end": 33480577,
    },
    "GRCh37": {
        "contig": "6",
        "start": 28477797,
        "end": 33448354,
    },
    "hg19": {
        "contig": "chr6",
        "start": 28477797,
        "end": 33448354,
    },
    "hg38": {
        "contig": "chr6",
        "start": 28510120,
        "end": 33480577,
    },
}


def get_mhc_region(reference_name: str) -> dict[str, str | int] | None:
    """Get MHC region coordinates for a known reference genome."""
    for key, region in MHC_REGIONS.items():
        if key.lower() in reference_name.lower():
            return region
    return None
