"""Runtime reproducibility manifest for HLA-Unified V2.

Captures complete environment state at the start of each pipeline run:
Python version, package versions, external tool versions, IMGT DB
provenance, and OS/platform info. Written to manifest.json alongside
pipeline outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_package_version(package: str) -> str:
    """Get installed version of a Python package."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return "not installed"


def _get_tool_version(tool: str) -> str:
    """Get version string from an external CLI tool."""
    if not shutil.which(tool):
        return "not found"
    try:
        result = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=10,
        )
        output = (result.stdout or result.stderr).strip()
        # Take first line only
        return output.split("\n")[0][:200]
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


def generate_manifest(
    imgt_provenance: dict,
    config_summary: dict | None = None,
) -> dict:
    """Generate a complete reproducibility manifest.

    Args:
        imgt_provenance: Database provenance dict from IMGTDatabase.
        config_summary: Optional pipeline config snapshot.

    Returns:
        Dict suitable for JSON serialization.
    """
    manifest = {
        "manifest_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Platform
        "platform": {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": _get_python_version(),
            "python_executable": sys.executable,
        },
        # Python packages
        "python_packages": {
            pkg: _get_package_version(pkg)
            for pkg in [
                "numpy", "scipy", "pysam", "click", "pulp", "msgpack",
            ]
        },
        # External tools
        "external_tools": {
            tool: _get_tool_version(tool)
            for tool in [
                "samtools", "minimap2", "bowtie2", "bwa",
                "megahit", "spades.py", "flye", "hifiasm",
            ]
        },
        # IMGT database
        "imgt_database": {
            "release": imgt_provenance.get("release", "unknown"),
            "git_commit": imgt_provenance.get("git_commit", "unknown"),
            "download_date": imgt_provenance.get("download_date", "unknown"),
            "source": imgt_provenance.get("source", "unknown"),
            "db_path": imgt_provenance.get("db_path", "unknown"),
        },
        # HLA-Unified version
        "hla_unified": {
            "version": "2.0.0",
        },
    }

    if config_summary:
        manifest["pipeline_config"] = config_summary

    return manifest


def write_manifest(manifest: dict, output_dir: Path) -> Path:
    """Write manifest to JSON file."""
    path = output_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("Reproducibility manifest written to %s", path)
    return path


def write_imgt_lockfile(
    imgt_provenance: dict,
    allele_counts: dict[str, int],
    output_dir: Path,
    allele_list_path: Path | None = None,
) -> Path:
    """Write an IMGT lock file for exact database reproducibility.

    The lockfile contains the release version, git commit, per-locus
    allele counts, and an optional hash of the Allelelist.txt file.
    """
    lockfile = {
        "lockfile_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imgt_release": imgt_provenance.get("release", "unknown"),
        "imgt_git_commit": imgt_provenance.get("git_commit", "unknown"),
        "allele_counts_per_locus": allele_counts,
        "total_alleles": sum(allele_counts.values()),
    }

    if allele_list_path and allele_list_path.exists():
        lockfile["allelelist_sha256"] = _hash_file(allele_list_path)

    path = output_dir / "imgt_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lockfile, indent=2))
    logger.info("IMGT lockfile written to %s", path)
    return path


def verify_imgt_lockfile(lockfile_path: Path, imgt_provenance: dict) -> bool:
    """Verify that the current IMGT database matches a lockfile.

    Returns True if the database matches, False otherwise.
    """
    if not lockfile_path.exists():
        logger.warning("Lockfile not found: %s", lockfile_path)
        return False

    lockfile = json.loads(lockfile_path.read_text())

    expected_release = lockfile.get("imgt_release", "")
    actual_release = imgt_provenance.get("release", "")

    if expected_release != actual_release:
        logger.error(
            "IMGT release mismatch: lockfile=%s, actual=%s",
            expected_release, actual_release,
        )
        return False

    expected_commit = lockfile.get("imgt_git_commit", "")
    actual_commit = imgt_provenance.get("git_commit", "")

    if expected_commit and actual_commit and expected_commit != actual_commit:
        logger.error(
            "IMGT commit mismatch: lockfile=%s, actual=%s",
            expected_commit[:12], actual_commit[:12],
        )
        return False

    logger.info("IMGT lockfile verification passed: %s", expected_release)
    return True
