"""OptiType wrapper.

OptiType (Szolek et al. 2014) is a Class I HLA caller using ILP-based
inference on read alignments. Reference implementation:
https://github.com/FRED-2/OptiType
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .base import ExternalCaller, ExternalCallResult

logger = logging.getLogger(__name__)


class OptiTypeWrapper(ExternalCaller):
    """Wrapper for OptiType (Class I HLA caller)."""

    name = "OptiType"
    supported_loci = ["A", "B", "C"]
    supported_input_types = ["bam", "fastq"]

    def is_available(self) -> bool:
        return self._check_tool("OptiTypePipeline.py") or self._check_tool("optitype")

    def get_version(self) -> str:
        for tool in ("OptiTypePipeline.py", "optitype"):
            if self._check_tool(tool):
                result = self._run_subprocess(
                    [tool, "--version"], description="get version",
                )
                if result.returncode == 0:
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
            result.error_message = "OptiType not installed"
            return result

        out_dir = self.work_dir / sample_id / "optitype"
        out_dir.mkdir(parents=True, exist_ok=True)

        # OptiType requires FASTQ input — convert BAM if needed
        if bam_path and not r1_fastq:
            r1_fastq, r2_fastq = self._bam_to_fastq(Path(bam_path), out_dir)

        if not r1_fastq:
            result.success = False
            result.error_message = "No input reads provided"
            return result

        # Run OptiType
        tool = "OptiTypePipeline.py" if self._check_tool("OptiTypePipeline.py") else "optitype"
        cmd = [
            tool,
            "-i", str(r1_fastq),
        ]
        if r2_fastq:
            cmd.append(str(r2_fastq))
        cmd.extend([
            "--dna",
            "-o", str(out_dir),
            "-p", sample_id,
        ])

        try:
            proc = self._run_subprocess(cmd, description=f"OptiType on {sample_id}")
            if proc.returncode != 0:
                result.success = False
                result.error_message = proc.stderr[-500:]
                return result
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

        # Parse OptiType result TSV
        result_tsvs = list(out_dir.glob("**/*_result.tsv"))
        if not result_tsvs:
            result.success = False
            result.error_message = f"No result TSV found in {out_dir}"
            return result

        result.calls = self._parse_result_tsv(result_tsvs[0])
        result.confidence = {locus: "HIGH" for locus in result.calls}
        result.output_files = result_tsvs
        result.runtime_seconds = time.time() - start
        return result

    @staticmethod
    def _parse_result_tsv(path: Path) -> dict[str, tuple[str, str]]:
        """Parse OptiType result TSV format.

        Format:
            \tA1\tA2\tB1\tB2\tC1\tC2\tReads\tObjective
            0\tA*01:01\tA*11:01\tB*08:01\tB*56:01\tC*01:02\tC*07:01\t1250\t98.5
        """
        calls: dict[str, tuple[str, str]] = {}
        with open(path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                for locus in ["A", "B", "C"]:
                    a1 = row.get(f"{locus}1", "").strip()
                    a2 = row.get(f"{locus}2", "").strip()
                    if a1 and a2:
                        calls[locus] = (a1, a2)
                break  # OptiType outputs one result row
        return calls

    def _bam_to_fastq(
        self, bam_path: Path, out_dir: Path,
    ) -> tuple[Path, Path]:
        """Convert BAM to paired FASTQ for OptiType input."""
        r1 = out_dir / "reads_R1.fastq.gz"
        r2 = out_dir / "reads_R2.fastq.gz"

        if r1.exists() and r2.exists():
            return r1, r2

        # Name-sort then convert
        namesorted = out_dir / "namesorted.bam"
        self._run_subprocess(
            ["samtools", "sort", "-n", "-@", str(self.threads),
             "-o", str(namesorted), str(bam_path)],
            description="name-sort BAM",
        )
        self._run_subprocess(
            ["samtools", "fastq",
             "-1", str(r1), "-2", str(r2),
             "-0", "/dev/null", "-s", "/dev/null",
             "-n", str(namesorted)],
            description="convert to FASTQ",
        )
        namesorted.unlink(missing_ok=True)
        return r1, r2
