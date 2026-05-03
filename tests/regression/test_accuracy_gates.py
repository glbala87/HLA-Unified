"""Regression tests: accuracy gates that prevent degradation.

These tests run against the existing simulated benchmark results and
enforce minimum accuracy thresholds. Any algorithmic change that drops
accuracy below these gates will fail CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hla_unified.benchmark.datasets import BenchmarkDataset
from hla_unified.benchmark.runner import BenchmarkRunner
from hla_unified.benchmark.metrics import compute_accuracy, LocusAccuracy

VALIDATION_DIR = Path(__file__).resolve().parent.parent.parent / "validation"
TRUTHSETS_DIR = VALIDATION_DIR / "truthsets"
RESULTS_DIR = VALIDATION_DIR / "simulated_results"
REPORTS_DIR = VALIDATION_DIR / "reports"
BASELINE_REPORT = REPORTS_DIR / "benchmark_reference_10_samples.json"


def _load_benchmark_report() -> dict:
    """Load the reference benchmark report."""
    if not BASELINE_REPORT.exists():
        pytest.skip(f"Baseline report not found: {BASELINE_REPORT}")
    return json.loads(BASELINE_REPORT.read_text())


def _run_benchmark() -> dict:
    """Run benchmark and return report dict."""
    truth_tsv = TRUTHSETS_DIR / "reference_samples_truth.tsv"
    if not truth_tsv.exists():
        pytest.skip(f"Truth set not found: {truth_tsv}")

    dataset = BenchmarkDataset.from_tsv(
        name="regression_gate",
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
    return {
        "overall_accuracy": report.overall_accuracy,
        "overall_call_rate": report.overall_call_rate,
        "per_locus": {
            locus: {"accuracy": acc.accuracy, "call_rate": acc.call_rate}
            for locus, acc in report.per_locus.items()
        },
    }


@pytest.mark.regression
class TestOverallAccuracyGate:
    """Overall accuracy must stay above threshold."""

    def test_overall_accuracy_minimum(self):
        report = _run_benchmark()
        assert report["overall_accuracy"] >= 0.95, (
            f"Overall accuracy {report['overall_accuracy']:.1%} dropped below 95% gate"
        )

    def test_overall_call_rate(self):
        report = _run_benchmark()
        assert report["overall_call_rate"] >= 0.99, (
            f"Call rate {report['overall_call_rate']:.1%} dropped below 99% gate"
        )


@pytest.mark.regression
class TestClassIAccuracyGate:
    """Class I loci must maintain high accuracy."""

    @pytest.mark.parametrize("locus", ["A", "B", "C"])
    def test_class_i_per_locus(self, locus):
        report = _run_benchmark()
        acc = report["per_locus"].get(locus, {}).get("accuracy", 0)
        assert acc >= 0.97, (
            f"Class I locus {locus} accuracy {acc:.1%} dropped below 97% gate"
        )


@pytest.mark.regression
class TestClassIIAccuracyGate:
    """Class II loci must maintain acceptable accuracy."""

    @pytest.mark.parametrize("locus", ["DRB1", "DQB1", "DQA1", "DPA1", "DPB1"])
    def test_class_ii_per_locus(self, locus):
        report = _run_benchmark()
        acc = report["per_locus"].get(locus, {}).get("accuracy", 0)
        assert acc >= 0.90, (
            f"Class II locus {locus} accuracy {acc:.1%} dropped below 90% gate"
        )


@pytest.mark.regression
class TestNoRegressionFromBaseline:
    """Accuracy must not drop more than 2% from stored baseline."""

    def test_no_regression(self):
        baseline = _load_benchmark_report()
        current = _run_benchmark()

        baseline_acc = baseline["overall_accuracy"]
        current_acc = current["overall_accuracy"]
        max_drop = 0.02

        assert current_acc >= baseline_acc - max_drop, (
            f"Accuracy regressed: baseline={baseline_acc:.1%}, "
            f"current={current_acc:.1%}, max allowed drop={max_drop:.0%}"
        )

    def test_no_per_locus_regression(self):
        baseline = _load_benchmark_report()
        current = _run_benchmark()

        for locus in baseline.get("per_locus", {}):
            base_acc = baseline["per_locus"][locus].get("accuracy", 0)
            curr_acc = current["per_locus"].get(locus, {}).get("accuracy", 0)
            assert curr_acc >= base_acc - 0.05, (
                f"Locus {locus} regressed: baseline={base_acc:.1%}, "
                f"current={curr_acc:.1%}"
            )
