"""Allele encoding: maps locus+allele strings to compact integer codes."""

from __future__ import annotations


class LocusCodeAllocation:
    """Bidirectional mapping between (locus, allele_string) and numeric codes."""

    def __init__(self) -> None:
        self._coded: dict[str, dict[str, int]] = {}
        self._decoded: dict[str, dict[int, str]] = {}

    def encode(self, locus: str, value: str) -> int:
        """Encode an allele string for a given locus. Auto-assigns if new."""
        if locus not in self._coded:
            self._coded[locus] = {}
            self._decoded[locus] = {}
        if value not in self._coded[locus]:
            code = len(self._coded[locus])
            self._coded[locus][value] = code
            self._decoded[locus][code] = value
        return self._coded[locus][value]

    def decode(self, locus: str, code: int) -> str:
        """Decode a numeric code back to an allele string."""
        return self._decoded[locus][code]

    def knows_code(self, locus: str, code: int) -> bool:
        return locus in self._decoded and code in self._decoded[locus]

    def knows_allele(self, locus: str, allele: str) -> bool:
        return locus in self._coded and allele in self._coded[locus]

    def get_alleles(self, locus: str) -> list[int]:
        """Return all allocated codes for a locus."""
        if locus not in self._decoded:
            return []
        return list(self._decoded[locus].keys())

    def get_loci(self) -> list[str]:
        return list(self._coded.keys())

    def invert(self, locus: str, emission: int) -> int:
        """Get the complement code for a nucleotide emission."""
        allele = self.decode(locus, emission)
        complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
        inverted = complement_map.get(allele, allele)
        return self.encode(locus, inverted)

    def remove_locus(self, locus: str) -> None:
        self._coded.pop(locus, None)
        self._decoded.pop(locus, None)

    def serialize(self) -> list[str]:
        """Serialize to a list of strings for file storage."""
        lines = []
        for locus in sorted(self._coded.keys()):
            for allele, code in sorted(self._coded[locus].items(), key=lambda x: x[1]):
                lines.append(f"{locus}\t{allele}\t{code}")
        return lines

    @classmethod
    def deserialize(cls, lines: list[str]) -> LocusCodeAllocation:
        """Reconstruct from serialized lines."""
        alloc = cls()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                locus, allele, code = parts
                code_int = int(code)
                if locus not in alloc._coded:
                    alloc._coded[locus] = {}
                    alloc._decoded[locus] = {}
                alloc._coded[locus][allele] = code_int
                alloc._decoded[locus][code_int] = allele
        return alloc
