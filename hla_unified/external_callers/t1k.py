"""T1K wrapper.

T1K (Song et al. 2023) is a fast HLA/KIR caller using k-mer based
allele inference. Reference: https://github.com/mourisl/T1K
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .base import ExternalCaller, ExternalCallResult

logger = logging.getLogger(__name__)


class T1KWrapper(ExternalCaller):
    """Wrapper for T1K (k-mer based HLA/KIR caller)."""

    name = "T1K"
    supported_loci = ["A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1"]
    supported_input_types = ["bam", "fastq"]

    def __init__(
        self,
        work_dir: str | Path,
        threads: int = 4,
        timeout: int = 7200,
        reference_fa: str | None = None,
    ) -> None:
        super().__init__(work_dir, threads, timeout)
        self.reference_fa = reference_fa

    def is_available(self) -> bool:
        return self._check_tool("run-t1k") or self._check_tool("t1k")

    def get_version(self) -> str:
        for tool in ("run-t1k", "t1k"):
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
            result.error_message = "T1K not installed"
            return result

        out_dir = self.work_dir / sample_id / "t1k"
        out_dir.mkdir(parents=True, exist_ok=True)

        tool = "run-t1k" if self._check_tool("run-t1k") else "t1k"
        cmd = [tool, "-t", str(self.threads), "-o", sample_id, "--od", str(out_dir)]

        if bam_path:
            cmd.extend(["-b", str(bam_path)])
        elif r1_fastq:
            cmd.extend(["-1", str(r1_fastq)])
            if r2_fastq:
                cmd.extend(["-2", str(r2_fastq)])
        else:
            result.success = False
            result.error_message = "No input provided"
            return result

        if self.reference_fa:
            cmd.extend(["-f", self.reference_fa])

        cmd.extend(["--preset", "hla-wgs"])

        try:
            proc = self._run_subprocess(cmd, description=f"T1K on {sample_id}")
            if proc.returncode != 0:
                result.success = False
                result.error_message = proc.stderr[-500:]
                return result
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

        # T1K writes <sample_id>_genotype.tsv
        genotype_tsv = out_dir / f"{sample_id}_genotype.tsv"
        if not genotype_tsv.exists():
            for alt in out_dir.rglob("*genotype.tsv"):
                genotype_tsv = alt
                break

        if not genotype_tsv.exists():
            result.success = False
            result.error_message = f"T1K output not found in {out_dir}"
            return result

        result.calls = self._parse_genotype_tsv(genotype_tsv)
        result.confidence = {locus: "HIGH" for locus in result.calls}
        result.output_files = [genotype_tsv]
        result.runtime_seconds = time.time() - start
        return result

    @staticmethod
    def _parse_genotype_tsv(path: Path) -> dict[str, tuple[str, str]]:
        """Parse T1K genotype TSV format.

        Format (one row per gene):
            HLA-A  2  HLA-A*01:01:01  100.0  ...  HLA-A*11:01:01  100.0  ...
        """
        calls: dict[str, tuple[str, str]] = {}
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                gene = parts[0].replace("HLA-", "")
                # Allele 1 in column 2, allele 2 in column 6 (varies by version)
                alleles = []
                for p in parts[2:]:
                    if "*" in p and p.startswith("HLA-"):
                        alleles.append(p.replace("HLA-", ""))
                if len(alleles) >= 2:
                    calls[gene] = (alleles[0], alleles[1])
                elif len(alleles) == 1:
                    calls[gene] = (alleles[0], alleles[0])
        return calls
