"""Cross-caller consensus comparison for HLA-Unified V2.

Compares HLA typing results across multiple callers to identify
concordant calls, discordant loci, and systematic disagreements.
Supports common caller output formats.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import compare_diploid

logger = logging.getLogger(__name__)


@dataclass
class ConsensusLocusResult:
    """Consensus result for a single locus."""
    locus: str
    consensus_allele1: str
    consensus_allele2: str
    n_callers_agree: int
    n_callers_total: int
    concordance_rate: float
    per_caller: dict[str, tuple[str, str]]  # caller -> (a1, a2)
    is_concordant: bool

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "consensus": [self.consensus_allele1, self.consensus_allele2],
            "concordance_rate": round(self.concordance_rate, 4),
            "n_agree": self.n_callers_agree,
            "n_total": self.n_callers_total,
            "is_concordant": self.is_concordant,
            "per_caller": {
                caller: list(alleles)
                for caller, alleles in self.per_caller.items()
            },
        }


@dataclass
class ConsensusReport:
    """Complete cross-caller consensus report."""
    sample_id: str
    callers: list[str]
    resolution: int
    per_locus: dict[str, ConsensusLocusResult]
    overall_concordance: float
    fully_concordant_loci: int
    discordant_loci: int

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "callers": self.callers,
            "resolution": self.resolution,
            "overall_concordance": round(self.overall_concordance, 4),
            "fully_concordant_loci": self.fully_concordant_loci,
            "discordant_loci": self.discordant_loci,
            "per_locus": {
                locus: result.to_dict()
                for locus, result in self.per_locus.items()
            },
        }


class CrossCallerConsensus:
    """Compare and reconcile HLA calls across multiple callers."""

    SUPPORTED_FORMATS = [
        "hla_unified", "hla_la", "optitype", "xhla",
        "hla_hd", "arcas_hla", "generic_tsv",
    ]

    def parse_results(
        self, caller: str, results_path: Path,
    ) -> dict[str, tuple[str, str]]:
        """Parse results from various caller output formats.

        Returns: locus -> (allele1, allele2)
        """
        if not results_path.exists():
            logger.warning("Results file not found: %s", results_path)
            return {}

        parser = {
            "hla_unified": self._parse_hla_unified,
            "hla_la": self._parse_hla_la,
            "optitype": self._parse_optitype,
            "xhla": self._parse_xhla,
            "arcas_hla": self._parse_arcas,
            "generic_tsv": self._parse_generic_tsv,
        }.get(caller, self._parse_generic_tsv)

        return parser(results_path)

    def compute_consensus(
        self,
        caller_results: dict[str, dict[str, tuple[str, str]]],
        resolution: int = 2,
        sample_id: str = "",
    ) -> ConsensusReport:
        """Compute per-locus consensus across callers."""
        callers = sorted(caller_results.keys())
        all_loci = set()
        for results in caller_results.values():
            all_loci.update(results.keys())

        per_locus: dict[str, ConsensusLocusResult] = {}
        n_concordant = 0

        for locus in sorted(all_loci):
            # Collect calls from all callers
            locus_calls: dict[str, tuple[str, str]] = {}
            for caller in callers:
                call = caller_results[caller].get(locus)
                if call and (call[0] or call[1]):
                    locus_calls[caller] = call

            if not locus_calls:
                continue

            # Find consensus by majority vote
            consensus, n_agree = self._majority_vote(
                locus_calls, resolution,
            )

            concordance = n_agree / max(len(locus_calls), 1)
            is_concordant = concordance >= 0.5 and n_agree >= 2

            if is_concordant:
                n_concordant += 1

            per_locus[locus] = ConsensusLocusResult(
                locus=locus,
                consensus_allele1=consensus[0],
                consensus_allele2=consensus[1],
                n_callers_agree=n_agree,
                n_callers_total=len(locus_calls),
                concordance_rate=concordance,
                per_caller=locus_calls,
                is_concordant=is_concordant,
            )

        overall_concordance = n_concordant / max(len(per_locus), 1)

        return ConsensusReport(
            sample_id=sample_id,
            callers=callers,
            resolution=resolution,
            per_locus=per_locus,
            overall_concordance=overall_concordance,
            fully_concordant_loci=n_concordant,
            discordant_loci=len(per_locus) - n_concordant,
        )

    def _majority_vote(
        self,
        calls: dict[str, tuple[str, str]],
        resolution: int,
    ) -> tuple[tuple[str, str], int]:
        """Find the most common diploid call via majority vote."""
        from ..reference.loci import truncate_to_resolution

        normalized: list[tuple[str, str]] = []
        for caller, (a1, a2) in calls.items():
            t1 = truncate_to_resolution(a1, resolution) if a1 else ""
            t2 = truncate_to_resolution(a2, resolution) if a2 else ""
            # Normalize order
            pair = tuple(sorted([t1, t2]))
            normalized.append(pair)

        # Count occurrences
        from collections import Counter
        counts = Counter(normalized)
        best_pair, best_count = counts.most_common(1)[0]

        return best_pair, best_count

    # --- Format-specific parsers ---

    def _parse_hla_unified(self, path: Path) -> dict[str, tuple[str, str]]:
        calls: dict[str, dict[int, str]] = {}
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3 or parts[0] == "Locus":
                    continue
                locus = parts[0].replace("HLA-", "")
                chrom = int(parts[1])
                allele = parts[2]
                calls.setdefault(locus, {})[chrom] = allele
        return {l: (c.get(1, ""), c.get(2, "")) for l, c in calls.items()}

    def _parse_hla_la(self, path: Path) -> dict[str, tuple[str, str]]:
        """Parse HLA*LA bestguess_G.txt format."""
        calls: dict[str, dict[int, str]] = {}
        with open(path) as fh:
            for line in fh:
                if line.startswith("Locus"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                locus = parts[0].replace("HLA-", "")
                chrom = int(parts[1])
                allele = parts[2]
                calls.setdefault(locus, {})[chrom] = allele
        return {l: (c.get(1, ""), c.get(2, "")) for l, c in calls.items()}

    def _parse_optitype(self, path: Path) -> dict[str, tuple[str, str]]:
        """Parse OptiType result TSV."""
        calls = {}
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                for locus in ["A", "B", "C"]:
                    a1 = row.get(f"{locus}1", "")
                    a2 = row.get(f"{locus}2", "")
                    if a1 or a2:
                        calls[locus] = (a1, a2)
        return calls

    def _parse_xhla(self, path: Path) -> dict[str, tuple[str, str]]:
        """Parse xHLA JSON output."""
        try:
            data = json.loads(path.read_text())
            hla = data.get("hla", {}).get("alleles", [])
            calls: dict[str, list[str]] = {}
            for allele in hla:
                from ..reference.loci import parse_allele_name
                info = parse_allele_name(allele)
                calls.setdefault(info.locus, []).append(allele)
            return {
                l: (alleles[0] if len(alleles) > 0 else "",
                    alleles[1] if len(alleles) > 1 else "")
                for l, alleles in calls.items()
            }
        except Exception as e:
            logger.warning("Failed to parse xHLA output %s: %s", path, e)
            return {}

    def _parse_arcas(self, path: Path) -> dict[str, tuple[str, str]]:
        """Parse arcasHLA genotype JSON."""
        try:
            data = json.loads(path.read_text())
            calls = {}
            for locus, alleles in data.items():
                locus = locus.replace("HLA-", "").replace("hla_", "").upper()
                if isinstance(alleles, list) and len(alleles) >= 2:
                    calls[locus] = (alleles[0], alleles[1])
                elif isinstance(alleles, list) and len(alleles) == 1:
                    calls[locus] = (alleles[0], alleles[0])
            return calls
        except Exception as e:
            logger.warning("Failed to parse arcasHLA output %s: %s", path, e)
            return {}

    def _parse_generic_tsv(self, path: Path) -> dict[str, tuple[str, str]]:
        """Parse generic TSV with Locus/Chromosome/Allele columns."""
        return self._parse_hla_unified(path)
