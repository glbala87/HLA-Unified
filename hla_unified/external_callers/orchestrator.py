"""Multi-caller orchestration: run available external callers in parallel
and merge their results into a unified consensus call.

This allows HLA-Unified to be used as an orchestration layer that runs
several validated callers and combines their output, leveraging
HLA-Unified's downstream infrastructure (validation, ambiguity, reporting).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .base import ExternalCaller, ExternalCallResult
from .optitype import OptiTypeWrapper
from .hlala import HLALAWrapper
from .t1k import T1KWrapper
from .arcas import ArcasHLAWrapper

logger = logging.getLogger(__name__)


@dataclass
class ConsensusCall:
    """Consensus call for a single locus across multiple callers."""
    locus: str
    allele1: str
    allele2: str
    n_callers_total: int
    n_callers_agreeing: int
    confidence: str  # HIGH (all agree), MEDIUM (majority), LOW (split)
    per_caller: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result from running multiple external callers."""
    sample_id: str
    callers_attempted: list[str]
    callers_succeeded: list[str]
    consensus: dict[str, ConsensusCall]
    individual_results: dict[str, ExternalCallResult]
    total_runtime_seconds: float

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "callers_attempted": self.callers_attempted,
            "callers_succeeded": self.callers_succeeded,
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
            "consensus": {
                locus: {
                    "allele1": c.allele1,
                    "allele2": c.allele2,
                    "n_callers_total": c.n_callers_total,
                    "n_callers_agreeing": c.n_callers_agreeing,
                    "confidence": c.confidence,
                    "per_caller": {
                        caller: list(alleles)
                        for caller, alleles in c.per_caller.items()
                    },
                }
                for locus, c in self.consensus.items()
            },
            "individual_results": {
                caller: r.to_dict()
                for caller, r in self.individual_results.items()
            },
        }


class MultiCallerOrchestrator:
    """Run multiple external HLA callers and merge results into consensus."""

    def __init__(
        self,
        work_dir: str | Path,
        threads: int = 4,
        callers: list[str] | None = None,
        graph_dir: str | None = None,
        timeout: int = 14400,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.threads = threads
        self.timeout = timeout

        # Build available caller wrappers
        all_wrappers: dict[str, ExternalCaller] = {
            "optitype": OptiTypeWrapper(work_dir, threads, timeout),
            "hlala": HLALAWrapper(work_dir, threads, timeout, graph_dir=graph_dir),
            "t1k": T1KWrapper(work_dir, threads, timeout),
            "arcashla": ArcasHLAWrapper(work_dir, threads, timeout),
        }

        if callers:
            self.wrappers = {k: v for k, v in all_wrappers.items() if k in callers}
        else:
            self.wrappers = all_wrappers

    def list_available(self) -> dict[str, bool]:
        """Check which callers are installed and ready to run."""
        return {
            name: wrapper.is_available()
            for name, wrapper in self.wrappers.items()
        }

    def run(
        self,
        bam_path: str | Path | None = None,
        r1_fastq: str | Path | None = None,
        r2_fastq: str | Path | None = None,
        sample_id: str = "sample",
        parallel: bool = True,
    ) -> OrchestrationResult:
        """Run all available callers and merge results.

        Args:
            bam_path: Input BAM/CRAM file
            r1_fastq, r2_fastq: Input FASTQ files (alternative to BAM)
            sample_id: Sample identifier
            parallel: Run callers in parallel (default True)
        """
        start = time.time()

        available = {
            name: wrapper for name, wrapper in self.wrappers.items()
            if wrapper.is_available()
        }

        if not available:
            logger.warning("No external callers available")
            return OrchestrationResult(
                sample_id=sample_id,
                callers_attempted=list(self.wrappers.keys()),
                callers_succeeded=[],
                consensus={},
                individual_results={},
                total_runtime_seconds=0.0,
            )

        logger.info(
            "Running %d external callers on %s: %s",
            len(available), sample_id, list(available.keys()),
        )

        # Run callers (in parallel by default)
        results: dict[str, ExternalCallResult] = {}

        def _run_one(name: str, wrapper: ExternalCaller) -> tuple[str, ExternalCallResult]:
            try:
                r = wrapper.run(
                    bam_path=bam_path,
                    r1_fastq=r1_fastq,
                    r2_fastq=r2_fastq,
                    sample_id=sample_id,
                )
                logger.info(
                    "  [%s] %s in %.0fs",
                    "OK" if r.success else "FAIL",
                    name, r.runtime_seconds,
                )
                return name, r
            except Exception as e:
                logger.error("  [%s] crashed: %s", name, e)
                return name, ExternalCallResult(
                    caller_name=name,
                    caller_version="unknown",
                    sample_id=sample_id,
                    calls={},
                    success=False,
                    error_message=str(e),
                )

        if parallel and len(available) > 1:
            with ThreadPoolExecutor(max_workers=len(available)) as pool:
                futures = {
                    pool.submit(_run_one, name, wrapper): name
                    for name, wrapper in available.items()
                }
                for future in as_completed(futures):
                    name, r = future.result()
                    results[name] = r
        else:
            for name, wrapper in available.items():
                name, r = _run_one(name, wrapper)
                results[name] = r

        # Compute consensus across successful callers
        succeeded = [name for name, r in results.items() if r.success]
        consensus = self._compute_consensus(results, succeeded)

        return OrchestrationResult(
            sample_id=sample_id,
            callers_attempted=list(self.wrappers.keys()),
            callers_succeeded=succeeded,
            consensus=consensus,
            individual_results=results,
            total_runtime_seconds=time.time() - start,
        )

    @staticmethod
    def _compute_consensus(
        results: dict[str, ExternalCallResult],
        succeeded: list[str],
    ) -> dict[str, ConsensusCall]:
        """Merge per-caller calls into consensus via majority vote at 2-field."""
        from ..reference.loci import truncate_to_resolution

        # Collect all loci across all successful callers
        all_loci = set()
        for name in succeeded:
            all_loci.update(results[name].calls.keys())

        consensus: dict[str, ConsensusCall] = {}

        for locus in sorted(all_loci):
            # Collect (a1, a2) pairs from each caller, normalized to 2-field
            per_caller: dict[str, tuple[str, str]] = {}
            normalized_pairs: list[tuple[str, str]] = []

            for name in succeeded:
                call = results[name].calls.get(locus)
                if not call or not (call[0] or call[1]):
                    continue
                a1 = truncate_to_resolution(call[0], 2) if call[0] else ""
                a2 = truncate_to_resolution(call[1], 2) if call[1] else ""
                # Strip HLA- prefix
                a1 = a1.removeprefix("HLA-")
                a2 = a2.removeprefix("HLA-")
                pair = tuple(sorted([a1, a2]))
                per_caller[name] = call
                normalized_pairs.append(pair)

            if not normalized_pairs:
                continue

            # Majority vote
            pair_counts = Counter(normalized_pairs)
            best_pair, best_count = pair_counts.most_common(1)[0]
            n_total = len(normalized_pairs)

            agreement = best_count / n_total
            if agreement == 1.0:
                conf = "HIGH"
            elif agreement >= 0.5:
                conf = "MEDIUM"
            else:
                conf = "LOW"

            consensus[locus] = ConsensusCall(
                locus=locus,
                allele1=best_pair[0],
                allele2=best_pair[1],
                n_callers_total=n_total,
                n_callers_agreeing=best_count,
                confidence=conf,
                per_caller=per_caller,
            )

        return consensus
