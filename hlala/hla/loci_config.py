"""HLA loci configuration: maps loci to their typing exons.

Defines which exons are used for typing each HLA locus,
and provides locus-specific parameters.
"""

from __future__ import annotations

# HLA loci to type and their key exons
# Class I (A, B, C): exons 2 and 3 are most informative
# Class II (DRA, DRB, DQA, DQB, DPA, DPB): exon 2 is most informative
LOCI_FOR_TYPING: list[str] = [
    "A", "B", "C", "DQA1", "DQB1", "DRB1", "DPA1", "DPB1",
]

# Mapping from locus to the exons used for typing
LOCI_TO_EXONS: dict[str, list[str]] = {
    "A": ["exon2", "exon3"],
    "B": ["exon2", "exon3"],
    "C": ["exon2", "exon3"],
    "DQA1": ["exon2"],
    "DQB1": ["exon2"],
    "DRB1": ["exon2"],
    "DRB3": ["exon2"],
    "DRB4": ["exon2"],
    "DPA1": ["exon2"],
    "DPB1": ["exon2"],
    "DRA": ["exon2"],
    "E": ["exon2", "exon3"],
    "F": ["exon2", "exon3"],
    "G": ["exon2", "exon3"],
    "H": ["exon2", "exon3"],
    "K": ["exon2", "exon3"],
    "V": ["exon2", "exon3"],
}

# Extended locus list (includes KIR)
ALL_LOCI: list[str] = LOCI_FOR_TYPING + [
    "DRB3", "DRB4", "DRA", "E", "F", "G", "H", "K", "V",
]

# Threshold parameters for HLA typing
THRESHOLD_UNACCOUNTED_ALLELES_MIN_COVERAGE = 30
THRESHOLD_UNACCOUNTED_ALLELES_MIN_ALLELE_FRACTION = 0.2

# High coverage filtering
HIGH_COVERAGE_MIN_COVERAGE = 50
HIGH_COVERAGE_MIN_ALLELE_FREQ = 0.2

# First-20 filter
FILTER_FIRST_20_N = 20
FILTER_FIRST_20_MIN_PROP = 0.3
FILTER_FIRST_20_LIMIT_KICK_OUT_PER_READ = 2

# Long reads strand filter
LONG_READS_MIN_ALLELE_COVERAGE = 5
LONG_READS_MIN_STRAND_FREQ = 0.2
