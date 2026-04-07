"""HLA*LA wrapper.

HLA*LA (Dilthey et al. 2019) uses graph alignment against a population-
augmented HLA reference. Reference: https://github.com/DiltheyLab/HLA-LA
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .base import ExternalCaller, ExternalCallResult

logger = logging.getLogger(__name__)


class HLALAWrapper(ExternalCaller):
    """Wrapper for HLA*LA (Class I + II HLA caller)."""

    name = "HLA*LA"
    supported_loci = ["A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1"]
    supported_input_types = ["bam", "cram"]

    def __init__(
        self,
        work_dir: str | Path,
        threads: int = 4,
        timeout: int = 14400,  # 4 hours
        graph_dir: str | None = None,
    ) -> None:
        super().__init__(work_dir, threads, timeout)
        self.graph_dir = graph_dir

    def is_available(self) -> bool:
        return self._check_tool("HLA-LA.pl") or self._check_tool("hla-la")

    def get_version(self) -> str:
        for tool in ("HLA-LA.pl", "hla-la"):
            if self._check_tool(tool):
                result = self._run_subprocess(
                    [tool, "--version"], description="get version",
                )
                if result.stdout:
                    return result.stdout.strip().split()[-1]
        return "unknown"

    def run(
        self,
        bam_path: str | Path | None = None,
        r1_fastq: str | Path | None = None,
        r2_fastq: str | Path | None = None,
        sample_id: str = "sample",
        **kwargs,
    ) -> ExternalCallResult:
        start = time.time()
        result = ExternalCallResult(
            caller_name=self.name,
            caller_version=self.get_version(),
            sample_id=sample_id,
            calls={},
        )

        if not self.is_available():
            result.success = False
            result.error_message = "HLA*LA not installed"
            return result

        if not bam_path:
            result.success = False
            result.error_message = "HLA*LA requires BAM/CRAM input"
            return result

        out_dir = self.work_dir / sample_id / "hlala"
        out_dir.mkdir(parents=True, exist_ok=True)

        tool = "HLA-LA.pl" if self._check_tool("HLA-LA.pl") else "hla-la"
        cmd = [
            tool,
            "--BAM", str(bam_path),
            "--graph", self.graph_dir or "PRG_MHC_GRCh38_withIMGT",
            "--sampleID", sample_id,
            "--maxThreads", str(self.threads),
            "--workingDir", str(out_dir),
        ]

        try:
            proc = self._run_subprocess(cmd, description=f"HLA*LA on {sample_id}")
            if proc.returncode != 0:
                result.success = False
                result.error_message = proc.stderr[-500:]
                return result
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

        # Parse R1_bestguess_G.txt or hla_summary.txt
        bestguess = out_dir / sample_id / "hla" / "R1_bestguess_G.txt"
        if not bestguess.exists():
            # Try alternate paths
            for alt in out_dir.rglob("R1_bestguess_G.txt"):
                bestguess = alt
                break
            else:
                for alt in out_dir.rglob("*bestguess*.txt"):
                    bestguess = alt
                    break

        if not bestguess.exists():
            result.success = False
            result.error_message = f"HLA*LA result file not found in {out_dir}"
            return result

        result.calls, result.confidence = self._parse_bestguess(bestguess)
        result.output_files = [bestguess]
        result.runtime_seconds = time.time() - start
        return result

    @staticmethod
    def _parse_bestguess(
        path: Path,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
        """Parse HLA*LA R1_bestguess_G.txt format."""
        calls: dict[str, dict[int, str]] = {}
        confidence: dict[str, str] = {}

        with open(path) as fh:
            header = fh.readline().strip().split("\t")
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                row = dict(zip(header, parts))
                locus = row.get("Locus", "").replace("HLA-", "")
                chrom = int(row.get("Chromosome", 0))
                allele = row.get("Allele", "")
                q1 = float(row.get("Q1", 0))

                calls.setdefault(locus, {})[chrom] = allele
                # HLA*LA Q1 is a confidence quality score
                if q1 >= 0.99:
                    confidence[locus] = "HIGH"
                elif q1 >= 0.90:
                    confidence[locus] = "MEDIUM"
                else:
                    confidence[locus] = "LOW"

        return (
            {l: (c.get(1, ""), c.get(2, "")) for l, c in calls.items()},
            confidence,
        )
