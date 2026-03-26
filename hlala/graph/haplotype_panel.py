"""Haplotype panel: a collection of known HLA haplotype sequences.

Used as input for building the population reference graph.
"""

from __future__ import annotations

from pathlib import Path


class HaplotypePanel:
    """A panel of haplotype sequences for graph construction.

    Stores haplotype IDs and their sequences organized by locus.
    """

    def __init__(self) -> None:
        self.haplotype_ids: list[str] = []
        self.haplotypes_by_loci: dict[str, bytes] = {}  # locus -> concatenated alleles
        self.locus_positions: dict[str, int] = {}
        self.locus_strands: dict[str, str] = {}
        self._id_cache: dict[str, int] = {}

    def read_from_file(self, filename: str | Path, positions_file: str | Path) -> None:
        """Read haplotype panel from file.

        The main file contains haplotype data, and positions_file contains
        locus position information.
        """
        with open(positions_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.locus_positions[parts[0]] = int(parts[1])
                    if len(parts) >= 3:
                        self.locus_strands[parts[0]] = parts[2]

        with open(filename) as fh:
            header = fh.readline().strip().split("\t")
            loci = header[1:]  # first column is haplotype ID

            for line in fh:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                hap_id = parts[0]
                self.haplotype_ids.append(hap_id)
                self._id_cache[hap_id] = len(self.haplotype_ids) - 1

                for i, locus in enumerate(loci):
                    allele = parts[i + 1] if i + 1 < len(parts) else ""
                    if locus not in self.haplotypes_by_loci:
                        self.haplotypes_by_loci[locus] = b""
                    self.haplotypes_by_loci[locus] += allele.encode()

    def get_unordered_loci(self) -> list[str]:
        return list(self.haplotypes_by_loci.keys())

    def get_ordered_loci(self) -> list[str]:
        """Return loci sorted by their position."""
        return sorted(self.locus_positions.keys(), key=lambda x: self.locus_positions.get(x, 0))

    def get_all_haplotypes(self) -> dict[str, str]:
        """Return dict mapping haplotype_id -> full concatenated sequence."""
        result: dict[str, str] = {}
        loci = self.get_ordered_loci()
        for idx, hap_id in enumerate(self.haplotype_ids):
            parts = []
            for locus in loci:
                data = self.haplotypes_by_loci.get(locus, b"")
                if data:
                    parts.append(chr(data[idx]))
            result[hap_id] = "".join(parts)
        return result

    def add_string(self, loci: list[str], hap_id: str, haplotype: str) -> None:
        """Add a haplotype sequence to the panel."""
        self.haplotype_ids.append(hap_id)
        self._id_cache[hap_id] = len(self.haplotype_ids) - 1
        for i, locus in enumerate(loci):
            char = haplotype[i] if i < len(haplotype) else ""
            if locus not in self.haplotypes_by_loci:
                self.haplotypes_by_loci[locus] = b""
            self.haplotypes_by_loci[locus] += char.encode()
