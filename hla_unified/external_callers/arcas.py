"""arcasHLA wrapper.

arcasHLA (Orenbuch et al. 2020) is an RNA-seq HLA caller using
expectation-maximization on transcript alignments.
Reference: https://github.com/RabadanLab/arcasHLA
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .base import ExternalCaller, ExternalCallResult

logger = logging.getLogger(__name__)


class ArcasHLAWrapper(ExternalCaller):
    """Wrapper for arcasHLA (RNA-seq HLA caller)."""

    name = "arcasHLA"
    supported_loci = ["A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1"]
    supported_input_types = ["bam", "fastq"]

    def is_available(self) -> bool:
        return self._check_tool("arcasHLA")

    def get_version(self) -> str:
        if not self._check_tool("arcasHLA"):
            return "unknown"
        result = self._run_subprocess(
            ["arcasHLA", "--version"], description="get version",
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
            result.error_message = "arcasHLA not installed"
            return result

        out_dir = self.work_dir / sample_id / "arcas"
        out_dir.mkdir(parents=True, exist_ok=True)

        if bam_path:
            # arcasHLA needs FASTQ from BAM first
            extract_cmd = [
                "arcasHLA", "extract", str(bam_path),
                "-o", str(out_dir),
                "-t", str(self.threads),
                "-v",
            ]
            try:
                proc = self._run_subprocess(extract_cmd, description="extract HLA reads")
                if proc.returncode != 0:
                    result.success = False
                    result.error_message = f"extract failed: {proc.stderr[-300:]}"
                    return result
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                return result

            # Find extracted FASTQs
            r1_files = list(out_dir.glob("*.extracted.1.fq.gz"))
            r2_files = list(out_dir.glob("*.extracted.2.fq.gz"))
            if r1_files:
                r1_fastq = r1_files[0]
            if r2_files:
                r2_fastq = r2_files[0]

        if not r1_fastq:
            result.success = False
            result.error_message = "No FASTQ available"
            return result

        # Run genotype
        genotype_cmd = [
            "arcasHLA", "genotype", str(r1_fastq),
        ]
        if r2_fastq:
            genotype_cmd.append(str(r2_fastq))
        genotype_cmd.extend([
            "-g", "A,B,C,DRB1,DQA1,DQB1,DPA1,DPB1",
            "-o", str(out_dir),
            "-t", str(self.threads),
            "-v",
        ])

        try:
            proc = self._run_subprocess(genotype_cmd, description="arcasHLA genotype")
            if proc.returncode != 0:
                result.success = False
                result.error_message = proc.stderr[-500:]
                return result
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

        # Parse genotype JSON
        genotype_json = None
        for f in out_dir.glob("*.genotype.json"):
            genotype_json = f
            break

        if not genotype_json:
            result.success = False
            result.error_message = f"arcasHLA JSON not found in {out_dir}"
            return result

        result.calls = self._parse_genotype_json(genotype_json)
        result.confidence = {locus: "HIGH" for locus in result.calls}
        result.output_files = [genotype_json]
        result.runtime_seconds = time.time() - start
        return result

    @staticmethod
    def _parse_genotype_json(path: Path) -> dict[str, tuple[str, str]]:
        """Parse arcasHLA genotype JSON format.

        Format: {"A": ["A*01:01:01", "A*11:01:01"], "B": [...], ...}
        """
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}

        calls = {}
        for locus, alleles in data.items():
            clean_locus = locus.replace("HLA-", "").replace("hla_", "").upper()
            if isinstance(alleles, list):
                if len(alleles) >= 2:
                    calls[clean_locus] = (alleles[0], alleles[1])
                elif len(alleles) == 1:
                    calls[clean_locus] = (alleles[0], alleles[0])
        return calls
