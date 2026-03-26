"""HLA locus definitions, resolution registry, and MHC region coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

# Classical HLA loci for typing
CLASS_I_LOCI = ["A", "B", "C"]
CLASS_II_LOCI = ["DRB1", "DQA1", "DQB1", "DPA1", "DPB1"]
EXTENDED_LOCI = ["DRB3", "DRB4", "DRB5", "DRA", "E", "F", "G"]
ALL_TYPING_LOCI = CLASS_I_LOCI + CLASS_II_LOCI

# Valid resolution levels (N-field)
RESOLUTION_LEVELS = {
    1: "1-field (allele group, e.g. A*02)",
    2: "2-field (specific protein, e.g. A*02:01)",
    3: "3-field (synonymous, e.g. A*02:01:01)",
    4: "4-field (non-coding, e.g. A*02:01:01:01)",
    "G": "G-group (antigen-binding domain equivalence)",
}

# Per-locus resolution registry: max achievable resolution per assay type
# Key: (locus, assay) -> max reliable field resolution
LOCUS_RESOLUTION_LIMITS: dict[tuple[str, str], int] = {
    # Short-read WGS: 2-3 field resolution (exons + partial introns)
    **{(l, "short"): 3 for l in CLASS_I_LOCI},
    **{(l, "short"): 2 for l in CLASS_II_LOCI},
    # Exome: 2-field only (only exons captured)
    **{(l, "exome"): 2 for l in CLASS_I_LOCI + CLASS_II_LOCI},
    # Targeted capture: up to 4-field (full gene coverage)
    **{(l, "targeted_capture"): 4 for l in CLASS_I_LOCI + CLASS_II_LOCI},
    # Long reads: up to 4-field (full gene phased)
    **{(l, "pacbio"): 4 for l in CLASS_I_LOCI + CLASS_II_LOCI},
    **{(l, "hifi"): 4 for l in CLASS_I_LOCI + CLASS_II_LOCI},
    **{(l, "ont"): 3 for l in CLASS_I_LOCI + CLASS_II_LOCI},
    # RNA-seq: 2-field only (coding sequence)
    **{(l, "rna"): 2 for l in CLASS_I_LOCI + CLASS_II_LOCI},
}


def get_max_resolution(locus: str, assay: str) -> int:
    """Get the maximum reliable field resolution for a locus+assay combination."""
    return LOCUS_RESOLUTION_LIMITS.get((locus, assay), 2)


def truncate_to_resolution(allele_name: str, fields: int) -> str:
    """Truncate an allele name to the specified field resolution.

    fields=1: A*02, fields=2: A*02:01, fields=3: A*02:01:01, etc.
    """
    info = parse_allele_name(allele_name)
    all_fields = [info.field1, info.field2, info.field3, info.field4]
    kept = [f for f in all_fields[:fields] if f]
    if not kept:
        return allele_name
    return f"{info.locus}*{':'.join(kept)}{info.suffix}"


# Exons used for typing at each resolution
TYPING_EXONS: dict[str, list[int]] = {
    # Class I: exons 2+3 define the peptide-binding groove
    "A": [2, 3],
    "B": [2, 3],
    "C": [2, 3],
    "E": [2, 3],
    "F": [2, 3],
    "G": [2, 3],
    # Class II: exon 2 defines the peptide-binding groove
    "DRB1": [2],
    "DRB3": [2],
    "DRB4": [2],
    "DRB5": [2],
    "DQA1": [2],
    "DQB1": [2],
    "DPA1": [2],
    "DPB1": [2],
    "DRA": [2],
}

# MHC region coordinates by reference build
MHC_REGIONS: dict[str, tuple[str, int, int]] = {
    "GRCh38": ("chr6", 28510120, 33480577),
    "GRCh37": ("6", 28477797, 33448354),
    "hg38": ("chr6", 28510120, 33480577),
    "hg19": ("chr6", 28477797, 33448354),
    "T2T": ("chr6", 28510120, 33480577),
}


@dataclass
class AlleleInfo:
    """Parsed HLA allele name fields."""
    locus: str
    field1: str  # allele group (2-digit)
    field2: str = ""  # specific protein (4-digit)
    field3: str = ""  # synonymous substitution (6-digit)
    field4: str = ""  # non-coding (8-digit)
    suffix: str = ""  # expression (N, L, S, C, A, Q)

    @property
    def two_digit(self) -> str:
        return f"{self.locus}*{self.field1}"

    @property
    def four_digit(self) -> str:
        if self.field2:
            return f"{self.locus}*{self.field1}:{self.field2}"
        return self.two_digit

    @property
    def full_name(self) -> str:
        parts = [self.field1]
        for f in [self.field2, self.field3, self.field4]:
            if f:
                parts.append(f)
            else:
                break
        return f"{self.locus}*{':'.join(parts)}{self.suffix}"


def parse_allele_name(name: str) -> AlleleInfo:
    """Parse an HLA allele name like 'A*02:01:01:01' into components."""
    # Strip HLA- prefix if present
    if name.startswith("HLA-"):
        name = name[4:]

    # Extract suffix
    suffix = ""
    if name and name[-1] in "NLSCAQ":
        suffix = name[-1]
        name = name[:-1]

    # Split locus from fields
    if "*" in name:
        locus, fields_str = name.split("*", 1)
    else:
        # Try to find locus boundary
        for i, ch in enumerate(name):
            if ch.isdigit():
                locus = name[:i]
                fields_str = name[i:]
                break
        else:
            return AlleleInfo(locus=name, field1="")

    fields = fields_str.split(":")
    return AlleleInfo(
        locus=locus,
        field1=fields[0] if len(fields) > 0 else "",
        field2=fields[1] if len(fields) > 1 else "",
        field3=fields[2] if len(fields) > 2 else "",
        field4=fields[3] if len(fields) > 3 else "",
        suffix=suffix,
    )


def group_alleles_by_resolution(
    alleles: list[str], level: int = 2
) -> dict[str, list[str]]:
    """Group allele names by N-field resolution.

    level=1: group by 2-digit (allele group)
    level=2: group by 4-digit (specific protein)
    """
    groups: dict[str, list[str]] = {}
    for allele in alleles:
        info = parse_allele_name(allele)
        if level == 1:
            key = info.two_digit
        else:
            key = info.four_digit
        groups.setdefault(key, []).append(allele)
    return groups
