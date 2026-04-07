"""Base class for external HLA caller wrappers."""

from __future__ import annotations

import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExternalCallResult:
    """Result from running an external HLA caller."""
    caller_name: str
    caller_version: str
    sample_id: str
    calls: dict[str, tuple[str, str]]  # locus -> (allele1, allele2)
    confidence: dict[str, str] = field(default_factory=dict)  # locus -> HIGH/MED/LOW
    runtime_seconds: float = 0.0
    success: bool = True
    error_message: str = ""
    output_files: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "caller_name": self.caller_name,
            "caller_version": self.caller_version,
            "sample_id": self.sample_id,
            "success": self.success,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "calls": {
                locus: list(alleles)
                for locus, alleles in self.calls.items()
            },
            "confidence": self.confidence,
            "error_message": self.error_message,
        }


class ExternalCaller(ABC):
    """Abstract base class for external HLA caller wrappers."""

    name: str = "external"
    supported_loci: list[str] = []
    supported_input_types: list[str] = ["bam"]  # bam, fastq, cram

    def __init__(
        self,
        work_dir: str | Path,
        threads: int = 4,
        timeout: int = 7200,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.threads = threads
        self.timeout = timeout

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the external caller is installed and runnable."""
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Get the installed caller version, or 'unknown'."""
        ...

    @abstractmethod
    def run(
        self,
        bam_path: str | Path | None = None,
        r1_fastq: str | Path | None = None,
        r2_fastq: str | Path | None = None,
        sample_id: str = "sample",
        **kwargs,
    ) -> ExternalCallResult:
        """Run the caller and return parsed results."""
        ...

    def _check_tool(self, tool: str) -> bool:
        """Check if a CLI tool is on PATH."""
        return shutil.which(tool) is not None

    def _run_subprocess(
        self,
        cmd: list[str],
        description: str = "",
        cwd: str | Path | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess with logging and timeout."""
        logger.info("[%s] %s", self.name, description or " ".join(cmd[:3]))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=cwd,
            check=False,
        )
