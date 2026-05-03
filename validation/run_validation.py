#!/usr/bin/env python3
"""Truthset validation runner for HLA-Unified V2.

Exercises the full benchmark framework against pre-computed results
from 10 reference samples across 4 ancestry groups (EUR, AFR, EAS, AMR).
Generates JSON reports, per-locus accuracy tables, ancestry-stratified
metrics, and cross-caller consensus analysis.

Usage:
    python run_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hla_unified.benchmark.datasets import BenchmarkDataset
from hla_unified.benchmark.runner import BenchmarkRunner
from hla_unified.benchmark.metrics import (
    BenchmarkReport, LocusAccuracy,
    compare_diploid, compute_accuracy, merge_accuracies,
)
from hla_unified.benchmark.consensus import CrossCallerConsensus

VALIDATION_DIR = Path(__file__).resolve().parent
TRUTHSETS_DIR = VALIDATION_DIR / "truthsets"
RESULTS_DIR = VALIDATION_DIR / "simulated_results"
REPORTS_DIR = VALIDATION_DIR / "reports"
CROSS_CALLER_DIR = VALIDATION_DIR / "cross_caller"


def run_full_benchmark() -> BenchmarkReport:
    """Run benchmark on all 10 reference samples (8 loci each)."""
    print("=" * 70)
    print("HLA-Unified V2 — Truthset Validation")
    print("=" * 70)

    truth_tsv = TRUTHSETS_DIR / "reference_samples_truth.tsv"
    dataset = BenchmarkDataset.from_tsv(
        name="reference_10_samples",
        truth_tsv=str(truth_tsv),
        bam_dir=str(RESULTS_DIR),  # dummy — skip_typing=True
        assay="short",
        resolution=2,
    )

    print(f"\nDataset: {dataset.name}")
    print(f"  Samples: {dataset.n_samples}")
    print(f"  Loci: {', '.join(dataset.loci)}")
    print(f"  Ancestry distribution: {dataset.ancestry_distribution}")

    runner = BenchmarkRunner(
        imgt_db_path=".",  # not needed for skip_typing
        work_dir=str(REPORTS_DIR),
        threads=1,
        data_type="short",
    )

    report = runner.run_dataset(
        dataset,
        resolution=2,
        skip_typing=True,
        results_dir=RESULTS_DIR,
    )

    # Write JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = runner.write_report(report, REPORTS_DIR)
    print(f"\n  Report written to: {report_path}")

    return report


def run_class_i_benchmark() -> BenchmarkReport:
    """Run benchmark on Class I loci only (A, B, C)."""
    truth_tsv = TRUTHSETS_DIR / "class_i_only_truth.tsv"
    dataset = BenchmarkDataset.from_tsv(
        name="class_i_only",
        truth_tsv=str(truth_tsv),
        bam_dir=str(RESULTS_DIR),
        assay="short",
        resolution=2,
    )

    runner = BenchmarkRunner(
        imgt_db_path=".",
        work_dir=str(REPORTS_DIR),
        threads=1,
        data_type="short",
    )

    report = runner.run_dataset(
        dataset, resolution=2, skip_typing=True, results_dir=RESULTS_DIR,
    )
    runner.write_report(report, REPORTS_DIR)
    return report


def run_cross_caller_consensus() -> dict:
    """Run cross-caller consensus comparison for NA12878."""
    print("\n" + "=" * 70)
    print("Cross-Caller Consensus Analysis — NA12878")
    print("=" * 70)

    consensus = CrossCallerConsensus()

    caller_results = {}

    # HLA-Unified V2
    caller_results["hla_unified"] = consensus.parse_results(
        "hla_unified",
        RESULTS_DIR / "NA12878" / "hla_types.tsv",
    )

    # HLA*LA
    caller_results["hla_la"] = consensus.parse_results(
        "hla_la",
        CROSS_CALLER_DIR / "hlala_NA12878.tsv",
    )

    # OptiType (Class I only)
    caller_results["optitype"] = consensus.parse_results(
        "optitype",
        CROSS_CALLER_DIR / "optitype_NA12878.tsv",
    )

    # xHLA
    caller_results["xhla"] = consensus.parse_results(
        "xhla",
        CROSS_CALLER_DIR / "xhla_NA12878.json",
    )

    # arcasHLA
    caller_results["arcas_hla"] = consensus.parse_results(
        "arcas_hla",
        CROSS_CALLER_DIR / "arcashla_NA12878.json",
    )

    report = consensus.compute_consensus(
        caller_results, resolution=2, sample_id="NA12878",
    )

    # Write consensus report
    consensus_path = REPORTS_DIR / "consensus_NA12878.json"
    consensus_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\n  Consensus report written to: {consensus_path}")

    return report.to_dict()


def print_results(report: BenchmarkReport, label: str) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    print(f"  Overall accuracy:  {report.overall_accuracy:.1%}")
    print(f"  Overall call rate: {report.overall_call_rate:.1%}")
    print(f"  Samples: {report.n_samples}  |  Loci: {report.n_loci}")

    print(f"\n  {'Locus':<10} {'Accuracy':>10} {'Call Rate':>10} {'Both OK':>10} {'One OK':>8} {'Miss':>6} {'NoCal':>6} {'N':>5}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*6} {'─'*5}")
    for locus in sorted(report.per_locus.keys()):
        acc = report.per_locus[locus]
        print(
            f"  {locus:<10} {acc.accuracy:>9.1%} {acc.call_rate:>9.1%} "
            f"{acc.n_correct_both:>10} {acc.n_correct_one:>8} "
            f"{acc.n_incorrect:>6} {acc.n_no_call:>6} {acc.n_samples:>5}"
        )

    # Per-ancestry breakdown
    if report.per_ancestry:
        print(f"\n  Per-Ancestry Accuracy:")
        for anc in sorted(report.per_ancestry.keys()):
            anc_accs = report.per_ancestry[anc]
            total_alleles = sum(a.n_samples * 2 for a in anc_accs.values())
            total_correct = sum(
                a.n_correct_both * 2 + a.n_correct_one
                for a in anc_accs.values()
            )
            anc_accuracy = total_correct / max(total_alleles, 1)
            n_samples = max(a.n_samples for a in anc_accs.values()) if anc_accs else 0
            print(f"    {anc:<6}  {anc_accuracy:.1%}  (n={n_samples})")

    # Discordant calls
    if report.discordant_calls:
        print(f"\n  Discordant Calls ({len(report.discordant_calls)}):")
        for dc in report.discordant_calls[:20]:
            print(
                f"    {dc['sample_id']:<10} HLA-{dc['locus']:<6} "
                f"call=({dc['call'][0]}, {dc['call'][1]})  "
                f"truth=({dc['truth'][0]}, {dc['truth'][1]})  "
                f"match={dc['match_level']}  ancestry={dc['ancestry']}"
            )


def print_consensus_results(consensus_dict: dict) -> None:
    """Print formatted consensus results."""
    print(f"\n  Callers: {', '.join(consensus_dict['callers'])}")
    print(f"  Overall concordance: {consensus_dict['overall_concordance']:.1%}")
    print(f"  Concordant loci: {consensus_dict['fully_concordant_loci']}")
    print(f"  Discordant loci: {consensus_dict['discordant_loci']}")

    print(f"\n  {'Locus':<10} {'Consensus':<30} {'Agree':>6} {'Total':>6} {'Rate':>8} {'Status':>10}")
    print(f"  {'─'*10} {'─'*30} {'─'*6} {'─'*6} {'─'*8} {'─'*10}")
    for locus, result in sorted(consensus_dict["per_locus"].items()):
        consensus_str = f"{result['consensus'][0]}, {result['consensus'][1]}"
        status = "CONCORDANT" if result["is_concordant"] else "DISCORDANT"
        print(
            f"  {locus:<10} {consensus_str:<30} "
            f"{result['n_agree']:>6} {result['n_total']:>6} "
            f"{result['concordance_rate']:>7.0%} {status:>10}"
        )


def generate_summary_report(
    full_report: BenchmarkReport,
    class_i_report: BenchmarkReport,
    consensus_dict: dict,
) -> None:
    """Generate a combined summary report."""
    summary = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "HLA-Unified V2",
        "full_benchmark": {
            "dataset": full_report.dataset_name,
            "n_samples": full_report.n_samples,
            "n_loci": full_report.n_loci,
            "overall_accuracy": round(full_report.overall_accuracy, 4),
            "overall_call_rate": round(full_report.overall_call_rate, 4),
            "per_locus": {
                locus: {
                    "accuracy": round(acc.accuracy, 4),
                    "call_rate": round(acc.call_rate, 4),
                    "n_samples": acc.n_samples,
                }
                for locus, acc in full_report.per_locus.items()
            },
            "per_ancestry": {
                anc: {
                    "accuracy": round(
                        sum(a.n_correct_both * 2 + a.n_correct_one for a in accs.values())
                        / max(sum(a.n_samples * 2 for a in accs.values()), 1),
                        4,
                    ),
                    "n_samples": max(
                        (a.n_samples for a in accs.values()), default=0,
                    ),
                }
                for anc, accs in full_report.per_ancestry.items()
            },
            "n_discordant_calls": len(full_report.discordant_calls),
        },
        "class_i_benchmark": {
            "overall_accuracy": round(class_i_report.overall_accuracy, 4),
            "overall_call_rate": round(class_i_report.overall_call_rate, 4),
            "n_samples": class_i_report.n_samples,
        },
        "cross_caller_consensus": {
            "sample": consensus_dict["sample_id"],
            "callers": consensus_dict["callers"],
            "overall_concordance": consensus_dict["overall_concordance"],
            "concordant_loci": consensus_dict["fully_concordant_loci"],
            "discordant_loci": consensus_dict["discordant_loci"],
        },
    }

    summary_path = REPORTS_DIR / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary report written to: {summary_path}")


def main() -> int:
    start = time.time()

    # 1. Full benchmark (all 10 samples, 8 loci)
    full_report = run_full_benchmark()
    print_results(full_report, "Full Benchmark — 10 Samples x 8 Loci (2-field resolution)")

    # 2. Class I only benchmark
    class_i_report = run_class_i_benchmark()
    print_results(class_i_report, "Class I Only — 10 Samples x 3 Loci (A, B, C)")

    # 3. Cross-caller consensus for NA12878
    consensus_dict = run_cross_caller_consensus()
    print_consensus_results(consensus_dict)

    # 4. Generate combined summary
    generate_summary_report(full_report, class_i_report, consensus_dict)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  Validation complete in {elapsed:.1f}s")
    print(f"  Reports directory: {REPORTS_DIR}")
    print(f"{'=' * 70}")

    # Return non-zero if accuracy < 95% (production gate)
    if full_report.overall_accuracy < 0.95:
        print(f"\n  FAIL: Overall accuracy {full_report.overall_accuracy:.1%} below 95% threshold!")
        return 1
    print(f"\n  PASS: Overall accuracy {full_report.overall_accuracy:.1%} >= 95% threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
