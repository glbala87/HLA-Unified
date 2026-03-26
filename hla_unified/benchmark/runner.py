"""Benchmark suite orchestrator for HLA-Unified V2.

Runs typing on benchmark datasets, computes multi-axis metrics,
and generates comparison reports. Supports stratification by
locus, ancestry, depth, and assay type.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from .datasets import BenchmarkDataset, BenchmarkSample
from .metrics import (
    BenchmarkReport, LocusAccuracy,
    compare_diploid, compute_accuracy, merge_accuracies,
)

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark evaluation runs."""

    def __init__(
        self,
        imgt_db_path: str | Path,
        work_dir: str | Path,
        threads: int = 4,
        data_type: str = "short",
    ) -> None:
        self.imgt_db_path = imgt_db_path
        self.work_dir = Path(work_dir)
        self.threads = threads
        self.data_type = data_type

    def run_dataset(
        self,
        dataset: BenchmarkDataset,
        resolution: int = 2,
        skip_typing: bool = False,
        results_dir: Path | None = None,
    ) -> BenchmarkReport:
        """Run benchmark on a dataset.

        Args:
            dataset: Benchmark dataset with truth
            resolution: Comparison resolution (fields)
            skip_typing: If True, only evaluate (assumes results exist)
            results_dir: Directory containing pre-computed results TSVs
        """
        logger.info(
            "Benchmarking %s: %d samples, %d loci, resolution=%d",
            dataset.name, dataset.n_samples, len(dataset.loci), resolution,
        )

        all_sample_accs: list[dict[str, LocusAccuracy]] = []
        all_sample_accs_by_ancestry: dict[str, list[dict[str, LocusAccuracy]]] = {}
        discordant_calls: list[dict[str, Any]] = []

        for sample in dataset.samples:
            # Get calls for this sample
            if skip_typing and results_dir:
                calls = self._load_results(results_dir / f"{sample.sample_id}" / "hla_types.tsv")
            elif skip_typing:
                logger.warning("skip_typing=True but no results_dir; skipping %s", sample.sample_id)
                continue
            else:
                calls = self._run_typing(sample, dataset.assay)

            if not calls:
                continue

            # Compare to truth
            sample_acc = compute_accuracy(calls, sample.truth, resolution)
            all_sample_accs.append(sample_acc)

            # Track by ancestry
            anc = sample.ancestry or "unknown"
            all_sample_accs_by_ancestry.setdefault(anc, []).append(sample_acc)

            # Record discordant calls
            for locus, acc in sample_acc.items():
                if acc.n_incorrect > 0 or acc.n_correct_one > 0:
                    call = calls.get(locus, ("", ""))
                    truth = sample.truth.get(locus, ("", ""))
                    discordant_calls.append({
                        "sample_id": sample.sample_id,
                        "locus": locus,
                        "call": list(call),
                        "truth": list(truth),
                        "match_level": compare_diploid(call, truth, resolution),
                        "ancestry": anc,
                    })

        # Aggregate results
        per_locus = merge_accuracies(all_sample_accs)

        per_ancestry = {}
        for anc, accs in all_sample_accs_by_ancestry.items():
            per_ancestry[anc] = merge_accuracies(accs)

        # Overall accuracy
        total_alleles = sum(a.n_samples * 2 for a in per_locus.values())
        total_correct = sum(
            a.n_correct_both * 2 + a.n_correct_one
            for a in per_locus.values()
        )
        overall_accuracy = total_correct / max(total_alleles, 1)

        total_called = sum(a.n_samples - a.n_no_call for a in per_locus.values())
        total_samples = sum(a.n_samples for a in per_locus.values())
        overall_call_rate = total_called / max(total_samples, 1)

        report = BenchmarkReport(
            dataset_name=dataset.name,
            resolution=resolution,
            overall_accuracy=overall_accuracy,
            overall_call_rate=overall_call_rate,
            per_locus=per_locus,
            per_ancestry=per_ancestry,
            n_samples=dataset.n_samples,
            n_loci=len(dataset.loci),
            discordant_calls=discordant_calls,
        )

        logger.info(
            "Benchmark complete: accuracy=%.2f%%, call_rate=%.2f%%",
            overall_accuracy * 100, overall_call_rate * 100,
        )

        return report

    def _run_typing(
        self, sample: BenchmarkSample, assay: str,
    ) -> dict[str, tuple[str, str]]:
        """Run HLA-Unified typing on a sample."""
        from ..pipeline.runner import UnifiedPipeline

        sample_dir = self.work_dir / sample.sample_id
        pipeline = UnifiedPipeline(
            imgt_db_path=self.imgt_db_path,
            work_dir=str(sample_dir),
            threads=self.threads,
            data_type=assay,
        )

        try:
            result = pipeline.run(sample.bam_path, input_type="bam")
            calls = {}
            for locus, call in result.calls.items():
                calls[locus] = (call.allele1, call.allele2)
            return calls
        except Exception as e:
            logger.error("Typing failed for %s: %s", sample.sample_id, e)
            return {}

    def _load_results(self, tsv_path: Path) -> dict[str, tuple[str, str]]:
        """Load pre-computed results from TSV."""
        calls: dict[str, dict[int, str]] = {}

        if not tsv_path.exists():
            return {}

        with open(tsv_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                break  # skip to DictReader

        with open(tsv_path) as fh:
            # Skip comment lines
            lines = [l for l in fh if not l.startswith("#")]

        if not lines:
            return {}

        import io
        reader = csv.DictReader(io.StringIO("".join(lines)), delimiter="\t")
        for row in reader:
            locus = row.get("Locus", "").replace("HLA-", "")
            chrom = int(row.get("Chromosome", 0))
            allele = row.get("Allele", "")
            calls.setdefault(locus, {})[chrom] = allele

        return {
            locus: (chroms.get(1, ""), chroms.get(2, ""))
            for locus, chroms in calls.items()
        }

    def write_report(self, report: BenchmarkReport, output_dir: Path) -> Path:
        """Write benchmark report to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"benchmark_{report.dataset_name}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Benchmark report written to %s", path)
        return path
