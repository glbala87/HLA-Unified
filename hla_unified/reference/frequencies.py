"""Allele frequency database for population-informed priors.

Provides population-level allele frequencies for the VB estimator's
Dirichlet prior. Common alleles get a stronger prior, reducing false
positives on rare types while still allowing rare calls when data supports.

Sources: approximate global frequencies from Allele Frequency Net Database
(AFND) and 1000 Genomes Project. These are rough estimates; users can
supply their own TSV for population-specific priors.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .loci import parse_allele_name

logger = logging.getLogger(__name__)

# Floor frequency for alleles not in the database
FLOOR_FREQUENCY = 1e-5


# Built-in global approximate 2-field frequencies for classical loci.
# Sourced from AFND global averages (Gonzalez-Galarza et al. 2020).
# Values are approximate and intended as priors, not exact estimates.
_BUILTIN_GLOBAL: dict[str, float] = {
    # HLA-A
    "A*02:01": 0.170, "A*01:01": 0.100, "A*03:01": 0.080, "A*24:02": 0.075,
    "A*11:01": 0.065, "A*68:01": 0.035, "A*23:01": 0.030, "A*30:01": 0.028,
    "A*26:01": 0.025, "A*31:01": 0.025, "A*32:01": 0.022, "A*33:03": 0.020,
    "A*29:02": 0.018, "A*30:02": 0.015, "A*02:06": 0.015, "A*68:02": 0.014,
    "A*33:01": 0.012, "A*34:01": 0.010, "A*36:01": 0.008, "A*74:01": 0.008,
    # HLA-B
    "B*07:02": 0.080, "B*08:01": 0.060, "B*35:01": 0.050, "B*44:02": 0.045,
    "B*15:01": 0.040, "B*40:01": 0.038, "B*51:01": 0.035, "B*44:03": 0.030,
    "B*18:01": 0.028, "B*27:05": 0.025, "B*57:01": 0.022, "B*14:02": 0.020,
    "B*53:01": 0.018, "B*58:01": 0.016, "B*13:02": 0.015, "B*38:01": 0.014,
    "B*49:01": 0.012, "B*52:01": 0.012, "B*55:01": 0.010, "B*56:01": 0.008,
    # HLA-C
    "C*07:01": 0.120, "C*07:02": 0.100, "C*04:01": 0.090, "C*06:02": 0.060,
    "C*03:04": 0.055, "C*05:01": 0.050, "C*01:02": 0.045, "C*03:03": 0.040,
    "C*08:02": 0.035, "C*12:03": 0.032, "C*02:02": 0.028, "C*15:02": 0.025,
    "C*16:01": 0.022, "C*14:02": 0.018, "C*08:01": 0.015, "C*17:01": 0.012,
    # HLA-DRB1
    "DRB1*07:01": 0.100, "DRB1*15:01": 0.080, "DRB1*03:01": 0.075,
    "DRB1*04:01": 0.065, "DRB1*01:01": 0.060, "DRB1*13:01": 0.050,
    "DRB1*11:01": 0.045, "DRB1*04:04": 0.035, "DRB1*08:01": 0.030,
    "DRB1*12:01": 0.028, "DRB1*14:01": 0.025, "DRB1*09:01": 0.022,
    "DRB1*13:02": 0.020, "DRB1*11:04": 0.018, "DRB1*04:05": 0.016,
    "DRB1*10:01": 0.015, "DRB1*16:01": 0.014, "DRB1*15:03": 0.012,
    # HLA-DQB1
    "DQB1*03:01": 0.130, "DQB1*02:01": 0.110, "DQB1*06:02": 0.080,
    "DQB1*05:01": 0.075, "DQB1*03:02": 0.060, "DQB1*06:03": 0.045,
    "DQB1*03:03": 0.040, "DQB1*04:02": 0.035, "DQB1*05:03": 0.025,
    "DQB1*06:04": 0.020, "DQB1*02:02": 0.018,
    # HLA-DQA1
    "DQA1*01:01": 0.110, "DQA1*05:01": 0.100, "DQA1*01:02": 0.090,
    "DQA1*03:01": 0.080, "DQA1*02:01": 0.070, "DQA1*01:03": 0.050,
    "DQA1*04:01": 0.035, "DQA1*05:05": 0.025, "DQA1*06:01": 0.015,
    # HLA-DPA1
    "DPA1*01:03": 0.300, "DPA1*02:01": 0.200, "DPA1*01:01": 0.150,
    "DPA1*02:02": 0.080, "DPA1*03:01": 0.050,
    # HLA-DPB1
    "DPB1*04:01": 0.200, "DPB1*04:02": 0.120, "DPB1*02:01": 0.100,
    "DPB1*01:01": 0.080, "DPB1*03:01": 0.060, "DPB1*05:01": 0.040,
    "DPB1*13:01": 0.030, "DPB1*06:01": 0.025, "DPB1*14:01": 0.020,
}


class AlleleFrequencyDatabase:
    """Manages allele frequency data for population-informed priors.

    Supports:
    - Built-in global frequencies (default)
    - User-supplied TSV files with population-specific data
    - Allele-group fallback (if A*02:01:01:01 not found, try A*02:01)
    """

    def __init__(self) -> None:
        self._frequencies: dict[str, float] = {}
        self._population: str = "global"

    @classmethod
    def load_builtin(cls, population: str = "global") -> AlleleFrequencyDatabase:
        """Load built-in frequency database."""
        db = cls()
        db._population = population
        db._frequencies = dict(_BUILTIN_GLOBAL)
        logger.info(
            "Loaded built-in %s frequencies: %d alleles",
            population, len(db._frequencies),
        )
        return db

    @classmethod
    def load_tsv(cls, path: Path, population: str | None = None) -> AlleleFrequencyDatabase:
        """Load frequencies from a TSV file.

        Expected columns: Locus, Allele, Frequency
        Optional column: Population (filtered if population arg is set)
        """
        db = cls()
        db._population = population or "custom"

        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if population and "Population" in row:
                    if row["Population"] != population:
                        continue
                allele = row["Allele"]
                freq = float(row["Frequency"])
                # Normalize allele name
                if not allele.startswith("HLA-"):
                    info = parse_allele_name(allele)
                    allele = info.full_name
                db._frequencies[allele] = freq

        logger.info(
            "Loaded %d frequencies from %s (population=%s)",
            len(db._frequencies), path, db._population,
        )
        return db

    def get_frequency(self, allele: str) -> float:
        """Get frequency for an allele, with fallback to allele group.

        Lookup order:
        1. Exact match (e.g., A*02:01:01:01)
        2. Truncated to 2-field (e.g., A*02:01)
        3. Truncated to 1-field (e.g., A*02)
        4. Floor frequency
        """
        # Exact match
        if allele in self._frequencies:
            return self._frequencies[allele]

        # Strip HLA- prefix
        clean = allele.removeprefix("HLA-")
        if clean in self._frequencies:
            return self._frequencies[clean]

        # Truncate to 2-field
        info = parse_allele_name(allele)
        two_field = info.four_digit
        if two_field in self._frequencies:
            return self._frequencies[two_field]

        # Truncate to 1-field
        one_field = info.two_digit
        if one_field in self._frequencies:
            return self._frequencies[one_field]

        return FLOOR_FREQUENCY

    def get_locus_frequencies(self, locus: str, allele_names: list[str]) -> dict[str, float]:
        """Get frequencies for all alleles at a locus."""
        return {name: self.get_frequency(name) for name in allele_names}

    @property
    def population(self) -> str:
        return self._population

    @property
    def n_alleles(self) -> int:
        return len(self._frequencies)


def load_default_frequencies() -> AlleleFrequencyDatabase:
    """Load the default global frequency database."""
    return AlleleFrequencyDatabase.load_builtin("global")
