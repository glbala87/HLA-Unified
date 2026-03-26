"""Accuracy and concordance metrics for HLA typing benchmarks.

Computes per-locus and overall metrics at configurable resolution levels,
stratified by ancestry, assay type, and depth. Handles the subtleties
of HLA comparison: order-independent diploid matching, resolution
truncation, and G-group equivalence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..reference.loci import truncate_to_resolution, parse_allele_name

logger = logging.getLogger(__name__)


@dataclass
class LocusAccuracy:
    """Accuracy metrics for a single locus."""
    locus: str
    n_samples: int = 0
    n_correct_both: int = 0        # both alleles correct
    n_correct_one: int = 0         # one allele correct
    n_incorrect: int = 0           # neither allele correct
    n_no_call: int = 0             # tool produced no call

    @property
    def accuracy(self) -> float:
        """Per-allele accuracy: (2*both + 1*one) / (2*total)."""
        total = self.n_samples * 2
        if total == 0:
            return 0.0
        correct = self.n_correct_both * 2 + self.n_correct_one
        return correct / total

    @property
    def call_rate(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return (self.n_samples - self.n_no_call) / self.n_samples

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "n_samples": self.n_samples,
            "n_correct_both": self.n_correct_both,
            "n_correct_one": self.n_correct_one,
            "n_incorrect": self.n_incorrect,
            "n_no_call": self.n_no_call,
            "accuracy": round(self.accuracy, 4),
            "call_rate": round(self.call_rate, 4),
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report with stratified metrics."""
    dataset_name: str
    resolution: int
    overall_accuracy: float
    overall_call_rate: float
    per_locus: dict[str, LocusAccuracy]
    per_ancestry: dict[str, dict[str, LocusAccuracy]] = field(default_factory=dict)
    per_confidence: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_loci: int = 0
    discordant_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "resolution": self.resolution,
            "n_samples": self.n_samples,
            "n_loci": self.n_loci,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_call_rate": round(self.overall_call_rate, 4),
            "per_locus": {
                locus: acc.to_dict()
                for locus, acc in self.per_locus.items()
            },
            "per_ancestry": {
                anc: {
                    locus: acc.to_dict()
                    for locus, acc in locus_accs.items()
                }
                for anc, locus_accs in self.per_ancestry.items()
            },
            "per_confidence": self.per_confidence,
            "discordant_calls": self.discordant_calls[:100],
        }


def compare_diploid(
    call: tuple[str, str],
    truth: tuple[str, str],
    resolution: int = 2,
) -> int:
    """Compare a diploid call against truth at given resolution.

    Returns:
        2: both alleles match
        1: one allele matches
        0: neither matches
        -1: no call made
    """
    c1, c2 = call
    t1, t2 = truth

    if not c1 and not c2:
        return -1

    # Truncate to comparison resolution
    c1 = truncate_to_resolution(c1, resolution) if c1 else ""
    c2 = truncate_to_resolution(c2, resolution) if c2 else ""
    t1 = truncate_to_resolution(t1, resolution) if t1 else ""
    t2 = truncate_to_resolution(t2, resolution) if t2 else ""

    # Strip HLA- prefix for comparison
    for prefix in ("HLA-",):
        c1 = c1.removeprefix(prefix)
        c2 = c2.removeprefix(prefix)
        t1 = t1.removeprefix(prefix)
        t2 = t2.removeprefix(prefix)

    # Order-independent matching
    if (c1 == t1 and c2 == t2) or (c1 == t2 and c2 == t1):
        return 2

    # Check one-allele match
    matches = 0
    truth_used = [False, False]

    if c1 == t1:
        matches += 1
        truth_used[0] = True
    elif c1 == t2:
        matches += 1
        truth_used[1] = True

    if c2 and not (matches and truth_used[0] and c2 == t1):
        if c2 == t1 and not truth_used[0]:
            matches += 1
        elif c2 == t2 and not truth_used[1]:
            matches += 1

    return min(matches, 2)


def compute_accuracy(
    calls: dict[str, tuple[str, str]],
    truth: dict[str, tuple[str, str]],
    resolution: int = 2,
) -> dict[str, LocusAccuracy]:
    """Compute per-locus accuracy for a single sample."""
    results = {}
    all_loci = sorted(set(calls.keys()) | set(truth.keys()))

    for locus in all_loci:
        acc = LocusAccuracy(locus=locus, n_samples=1)
        call = calls.get(locus, ("", ""))
        true = truth.get(locus)

        if true is None:
            continue

        match = compare_diploid(call, true, resolution)
        if match == -1:
            acc.n_no_call = 1
        elif match == 2:
            acc.n_correct_both = 1
        elif match == 1:
            acc.n_correct_one = 1
        else:
            acc.n_incorrect = 1

        results[locus] = acc

    return results


def merge_accuracies(
    all_accs: list[dict[str, LocusAccuracy]],
) -> dict[str, LocusAccuracy]:
    """Merge per-sample accuracy dicts into aggregate per-locus."""
    merged: dict[str, LocusAccuracy] = {}

    for sample_accs in all_accs:
        for locus, acc in sample_accs.items():
            if locus not in merged:
                merged[locus] = LocusAccuracy(locus=locus)
            m = merged[locus]
            m.n_samples += acc.n_samples
            m.n_correct_both += acc.n_correct_both
            m.n_correct_one += acc.n_correct_one
            m.n_incorrect += acc.n_incorrect
            m.n_no_call += acc.n_no_call

    return merged
