"""G-group allele translation.

Parses IMGT hla_nom_g.txt nomenclature file and translates
four-field HLA alleles to G-group designations.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GGroupTranslator:
    """Translates HLA alleles to G-group designations.

    G-groups cluster alleles that have identical sequences in the
    antigen-binding domain (exons 2+3 for class I, exon 2 for class II).
    """

    def __init__(self) -> None:
        self.allele_to_g: dict[str, str] = {}
        self.g_loci: set[str] = set()

    def load(self, filepath: str | Path) -> None:
        """Parse an IMGT hla_nom_g.txt file.

        Format per line:
            LOCUS*;allele1/allele2/.../alleleN;G_GROUP_NAME

        Lines starting with # are comments.
        """
        filepath = Path(filepath)
        logger.info(f"Loading G-group definitions from {filepath}")

        with open(filepath) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(";")
                if len(parts) < 3:
                    continue

                locus_prefix = parts[0]  # e.g. "A*"
                allele_list_str = parts[1]
                g_group = parts[2]

                if not g_group:
                    continue

                locus = locus_prefix.rstrip("*")
                self.g_loci.add(locus)

                # Full G-group name
                g_name = f"{locus_prefix}{g_group}"

                # Parse allele list
                alleles = allele_list_str.split("/")
                for allele in alleles:
                    if allele:
                        full_allele = f"{locus_prefix}{allele}"
                        self.allele_to_g[full_allele] = g_name

        logger.info(f"Loaded {len(self.allele_to_g)} allele-to-G mappings for {len(self.g_loci)} loci")

    def can_translate(self, locus: str) -> bool:
        """Check if G-group translation is available for a locus."""
        return locus in self.g_loci

    def translate(self, allele: str) -> tuple[str, bool]:
        """Translate an allele to its G-group.

        Returns (g_group_name, is_perfect_match).
        If no G-group mapping exists, returns the allele itself with False.
        """
        if allele in self.allele_to_g:
            return self.allele_to_g[allele], True

        # Try truncating to find a match
        # e.g., A*01:01:01:01 -> A*01:01:01 -> A*01:01
        parts = allele.split(":")
        for n_fields in range(len(parts) - 1, 1, -1):
            truncated = ":".join(parts[:n_fields])
            if truncated in self.allele_to_g:
                return self.allele_to_g[truncated], False

        return allele, False

    def translate_allele_list(self, alleles: list[str]) -> tuple[str, bool]:
        """Translate a list of equivalent alleles to a G-group.

        All alleles should map to the same G-group. Returns the consensus
        G-group and whether the mapping is perfect.
        """
        g_groups: set[str] = set()
        all_perfect = True
        for allele in alleles:
            g, perfect = self.translate(allele)
            g_groups.add(g)
            if not perfect:
                all_perfect = False

        if len(g_groups) == 1:
            return g_groups.pop(), all_perfect
        elif len(g_groups) == 0:
            return "", False
        else:
            # Multiple G-groups — return most common or first
            return sorted(g_groups)[0], False
