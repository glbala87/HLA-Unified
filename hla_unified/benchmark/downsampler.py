"""Synthetic downsampling for depth-curve evaluation.

Creates downsampled versions of BAM files at multiple target depths
and runs typing at each level to generate accuracy-vs-depth curves.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from ..utils.external import run_cmd, ToolError

logger = logging.getLogger(__name__)


@dataclass
class DepthCurvePoint:
    """One point on the accuracy-vs-depth curve."""
    target_depth: int
    actual_fraction: float
    accuracy: float
    call_rate: float
    n_high_confidence: int
    n_loci: int


class SyntheticDownsampler:
    """Downsample BAMs and evaluate typing at multiple depth levels."""

    def __init__(self, threads: int = 4) -> None:
        self.threads = threads

    def downsample(
        self,
        bam_path: Path,
        fraction: float,
        output_path: Path,
    ) -> Path:
        """Downsample a BAM file to a target fraction.

        Uses samtools view -s for reproducible subsampling.
        The seed is embedded in the fraction (e.g., 42.30 = seed 42, 30%).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # samtools -s INT.FRAC format: seed.fraction
        seed = 42
        subsample_arg = f"{seed}.{int(fraction * 100):02d}"

        try:
            run_cmd(
                [
                    "samtools", "view", "-b",
                    "-s", subsample_arg,
                    "-@", str(self.threads),
                    "-o", str(output_path),
                    str(bam_path),
                ],
                description=f"downsample to {fraction:.0%}",
            )
            run_cmd(
                ["samtools", "index", str(output_path)],
                description="index downsampled BAM",
            )
        except ToolError as e:
            logger.error("Downsampling failed: %s", e)
            raise

        return output_path

    def estimate_depth(self, bam_path: Path) -> float:
        """Estimate mean coverage depth from a BAM file."""
        try:
            result = run_cmd(
                ["samtools", "idxstats", str(bam_path)],
                description="get BAM index stats",
            )
            total_reads = 0
            total_length = 0
            for line in result.stdout.strip().split("\n"):
                fields = line.split("\t")
                if len(fields) >= 4 and fields[0] != "*":
                    total_length += int(fields[1])
                    total_reads += int(fields[2])
            if total_length > 0:
                return (total_reads * 150) / total_length  # assume 150bp reads
        except Exception:
            pass
        return 30.0  # default guess

    def run_depth_curve(
        self,
        bam_path: Path,
        truth: dict[str, tuple[str, str]],
        depths: list[int] | None = None,
        work_dir: Path | None = None,
        imgt_db_path: str = "",
        data_type: str = "short",
        resolution: int = 2,
    ) -> list[DepthCurvePoint]:
        """Run typing at multiple depths and compute accuracy curve.

        Args:
            bam_path: Full-depth BAM file
            truth: Ground truth per locus
            depths: Target depth levels (default: [5, 10, 20, 30, 50, 100])
            work_dir: Working directory
            imgt_db_path: Path to IMGT database
            data_type: Sequencing assay type
            resolution: Comparison resolution
        """
        if depths is None:
            depths = [5, 10, 20, 30, 50, 100]

        if work_dir is None:
            work_dir = Path(bam_path).parent / "depth_curve"

        work_dir.mkdir(parents=True, exist_ok=True)

        # Estimate current depth
        current_depth = self.estimate_depth(bam_path)
        logger.info("Estimated current depth: %.1fx", current_depth)

        from .metrics import compute_accuracy, merge_accuracies

        points: list[DepthCurvePoint] = []

        for target in sorted(depths):
            if target >= current_depth:
                fraction = 1.0
            else:
                fraction = target / current_depth

            logger.info("Depth %dx (fraction %.2f)", target, fraction)

            # Downsample
            ds_bam = work_dir / f"depth_{target}x.bam"
            if fraction < 1.0:
                self.downsample(bam_path, fraction, ds_bam)
            else:
                ds_bam = bam_path

            # Run typing
            from ..pipeline.runner import UnifiedPipeline
            sample_dir = work_dir / f"results_{target}x"
            pipeline = UnifiedPipeline(
                imgt_db_path=imgt_db_path,
                work_dir=str(sample_dir),
                threads=self.threads,
                data_type=data_type,
            )

            try:
                result = pipeline.run(str(ds_bam), input_type="bam")
                calls = {
                    locus: (call.allele1, call.allele2)
                    for locus, call in result.calls.items()
                }
            except Exception as e:
                logger.error("Typing failed at %dx: %s", target, e)
                calls = {}

            # Evaluate
            accs = compute_accuracy(calls, truth, resolution)

            total_alleles = sum(a.n_samples * 2 for a in accs.values())
            total_correct = sum(
                a.n_correct_both * 2 + a.n_correct_one
                for a in accs.values()
            )
            accuracy = total_correct / max(total_alleles, 1)

            n_called = sum(a.n_samples - a.n_no_call for a in accs.values())
            call_rate = n_called / max(len(accs), 1)

            n_high = sum(
                1 for locus, call in (result.calls if hasattr(result, "calls") else {}).items()
                if hasattr(call, "confidence") and call.confidence == "HIGH"
            )

            points.append(DepthCurvePoint(
                target_depth=target,
                actual_fraction=fraction,
                accuracy=accuracy,
                call_rate=call_rate,
                n_high_confidence=n_high,
                n_loci=len(accs),
            ))

            logger.info(
                "  %dx: accuracy=%.2f%%, call_rate=%.2f%%",
                target, accuracy * 100, call_rate * 100,
            )

        return points
