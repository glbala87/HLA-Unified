"""G-group translation: maps full allele names to G-group designations.

G-groups cluster alleles that have identical nucleotide sequences across
the antigen-binding domain exons (exons 2+3 for class I, exon 2 for class II).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GGroupTranslator:
    """Translates HLA allele names to G-group resolution."""

    def __init__(self) -> None:
        # allele -> G-group name
        self._mapping: dict[str, str] = {}
        self._loaded = False

    def load(self, path: str | Path) -> None:
        """Load G-group definitions from IMGT hla_nom_g.txt."""
        path = Path(path)
        if not path.exists():
            logger.warning("G-group file not found: %s", path)
            return

        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: locus;allele_list;g_group_name
                parts = line.split(";")
                if len(parts) < 3:
                    continue
                locus = parts[0].replace("*", "")
                allele_list = parts[1]
                g_name = parts[2]

                if not g_name:
                    continue

                # Parse allele list (separated by /)
                for allele_str in allele_list.split("/"):
                    allele_str = allele_str.strip()
                    if allele_str:
                        full = f"{locus}*{allele_str}"
                        self._mapping[full] = f"{locus}*{g_name}"

        self._loaded = True
        logger.info("Loaded %d G-group mappings", len(self._mapping))

    def translate(self, allele: str) -> tuple[str, bool]:
        """Translate an allele to G-group.

        Returns (g_group_name, is_perfect_translation).
        """
        if not self._loaded:
            return allele, False

        # Strip HLA- prefix
        clean = allele.replace("HLA-", "")

        # Try exact match
        if clean in self._mapping:
            return self._mapping[clean], True

        # Try progressive truncation
        parts = clean.split(":")
        for n in range(len(parts), 0, -1):
            truncated = ":".join(parts[:n])
            if truncated in self._mapping:
                return self._mapping[truncated], n == len(parts)

        return allele, False
