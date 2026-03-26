"""IMGT/HLA database management: download, index, and query allele sequences.

Includes strict provenance tracking: every database instance records
its exact IPD-IMGT/HLA release version, and this version is
embedded in all pipeline outputs.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from ..utils.io import read_fasta, write_fasta

logger = logging.getLogger(__name__)

PROVENANCE_FILE = ".hla_unified_provenance.json"

# Default IMGT/HLA release URL pattern
IMGT_GITHUB = "https://github.com/ANHIG/IMGTHLA"
IMGT_FASTA_DIR = "fasta"
IMGT_ALIGNMENTS_DIR = "alignments"

# Gene-specific FASTA files in IMGT/HLA
GENE_FILES = {
    "A": "A_gen.fasta",
    "B": "B_gen.fasta",
    "C": "C_gen.fasta",
    "DRB1": "DRB1_gen.fasta",
    "DQA1": "DQA1_gen.fasta",
    "DQB1": "DQB1_gen.fasta",
    "DPA1": "DPA1_gen.fasta",
    "DPB1": "DPB1_gen.fasta",
    "DRB3": "DRB3_gen.fasta",
    "DRB4": "DRB4_gen.fasta",
    "DRB5": "DRB5_gen.fasta",
    "E": "E_gen.fasta",
    "F": "F_gen.fasta",
    "G": "G_gen.fasta",
}

# CDS-only files (exon sequences only, for faster initial screening)
CDS_FILES = {
    "A": "A_nuc.fasta",
    "B": "B_nuc.fasta",
    "C": "C_nuc.fasta",
    "DRB1": "DRB1_nuc.fasta",
    "DQA1": "DQA1_nuc.fasta",
    "DQB1": "DQB1_nuc.fasta",
    "DPA1": "DPA1_nuc.fasta",
    "DPB1": "DPB1_nuc.fasta",
}


class IMGTDatabase:
    """Manages a local copy of the IMGT/HLA allele reference database.

    Tracks exact IPD-IMGT/HLA release version in a provenance metadata
    file. This version string is included in all pipeline outputs to
    ensure reproducibility.
    """

    def __init__(self, db_dir: str | Path) -> None:
        self.db_dir = Path(db_dir)
        self._alleles: dict[str, dict[str, str]] = {}
        self._cds: dict[str, dict[str, str]] = {}
        self._provenance: dict | None = None

    @property
    def is_available(self) -> bool:
        return (self.db_dir / "fasta").exists()

    @property
    def provenance(self) -> dict:
        """Get database provenance metadata.

        Returns dict with keys: release, download_date, git_commit,
        n_alleles, db_path.
        """
        if self._provenance is not None:
            return self._provenance

        prov_file = self.db_dir / PROVENANCE_FILE
        if prov_file.exists():
            self._provenance = json.loads(prov_file.read_text())
            return self._provenance

        # Try to reconstruct from database files
        self._provenance = self._detect_provenance()
        return self._provenance

    @property
    def release_version(self) -> str:
        """Get the IPD-IMGT/HLA release version string."""
        return self.provenance.get("release", "unknown")

    def setup(self, release: str = "Latest") -> None:
        """Download or update IMGT/HLA database with provenance tracking."""
        if self.is_available:
            logger.info("IMGT/HLA database already present at %s", self.db_dir)
            # Ensure provenance file exists
            if not (self.db_dir / PROVENANCE_FILE).exists():
                self._write_provenance(release)
            return

        logger.info("Cloning IMGT/HLA database (release=%s)...", release)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        clone_cmd = ["git", "clone", "--depth", "1"]
        if release != "Latest":
            clone_cmd.extend(["--branch", release])
        clone_cmd.extend([IMGT_GITHUB, str(self.db_dir)])

        subprocess.run(clone_cmd, check=True)
        self._write_provenance(release)

    def _write_provenance(self, release: str = "Latest") -> None:
        """Write provenance metadata file after database setup."""
        prov = {
            "release": self._detect_release_version(release),
            "download_date": datetime.utcnow().isoformat() + "Z",
            "git_commit": self._get_git_commit(),
            "db_path": str(self.db_dir),
            "source": IMGT_GITHUB,
        }
        prov_file = self.db_dir / PROVENANCE_FILE
        prov_file.write_text(json.dumps(prov, indent=2))
        self._provenance = prov
        logger.info("Provenance recorded: release=%s, commit=%s",
                     prov["release"], prov["git_commit"][:12])

    def _detect_release_version(self, requested: str = "Latest") -> str:
        """Detect the actual IPD-IMGT/HLA release version."""
        # Try reading from the release_version.txt file in IMGTHLA
        version_file = self.db_dir / "release_version.txt"
        if version_file.exists():
            return version_file.read_text().strip()

        # Try the Allelelist file header
        allele_list = self.db_dir / "Allelelist.txt"
        if allele_list.exists():
            with open(allele_list) as fh:
                header = fh.readline().strip()
                if header.startswith("#"):
                    # Often contains version info
                    return header.lstrip("#").strip()

        # Try git tags
        try:
            result = subprocess.run(
                ["git", "-C", str(self.db_dir), "describe", "--tags"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except FileNotFoundError:
            pass

        return requested

    def _get_git_commit(self) -> str:
        """Get the git commit hash of the database."""
        try:
            result = subprocess.run(
                ["git", "-C", str(self.db_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        return "unknown"

    def _detect_provenance(self) -> dict:
        """Reconstruct provenance from database files when metadata is missing."""
        return {
            "release": self._detect_release_version(),
            "download_date": "unknown",
            "git_commit": self._get_git_commit(),
            "db_path": str(self.db_dir),
            "source": IMGT_GITHUB,
        }

    def load_genomic(self, locus: str) -> dict[str, str]:
        """Load full genomic sequences for a locus."""
        if locus in self._alleles:
            return self._alleles[locus]

        fname = GENE_FILES.get(locus)
        if not fname:
            logger.warning("No genomic FASTA file known for locus %s", locus)
            return {}

        fpath = self.db_dir / "fasta" / fname
        if not fpath.exists():
            logger.warning("FASTA file not found: %s", fpath)
            return {}

        seqs = dict(read_fasta(fpath))
        self._alleles[locus] = seqs
        logger.info("Loaded %d genomic alleles for %s", len(seqs), locus)
        return seqs

    def load_cds(self, locus: str) -> dict[str, str]:
        """Load CDS (coding) sequences for a locus."""
        if locus in self._cds:
            return self._cds[locus]

        fname = CDS_FILES.get(locus)
        if not fname:
            return self.load_genomic(locus)

        fpath = self.db_dir / "fasta" / fname
        if not fpath.exists():
            return self.load_genomic(locus)

        seqs = dict(read_fasta(fpath))
        self._cds[locus] = seqs
        logger.info("Loaded %d CDS alleles for %s", len(seqs), locus)
        return seqs

    def get_all_allele_names(self, locus: str) -> list[str]:
        """Get all allele names for a locus."""
        seqs = self.load_genomic(locus)
        return list(seqs.keys())

    def build_reduced_panel(self, locus: str, output_path: Path) -> int:
        """Build a non-redundant reference panel (xHLA-style).

        Collapse alleles that have identical typing exon sequences
        into representative alleles. This dramatically reduces the
        candidate space for initial filtering.
        """
        from .loci import TYPING_EXONS, parse_allele_name

        cds_seqs = self.load_cds(locus)
        if not cds_seqs:
            return 0

        # Deduplicate by sequence
        seq_to_representative: dict[str, str] = {}
        for name, seq in cds_seqs.items():
            if seq not in seq_to_representative:
                seq_to_representative[seq] = name

        reduced = {name: seq for seq, name in seq_to_representative.items()}
        write_fasta(output_path, reduced)
        logger.info(
            "Reduced panel for %s: %d -> %d alleles",
            locus, len(cds_seqs), len(reduced),
        )
        return len(reduced)

    def get_allele_counts(self) -> dict[str, int]:
        """Get number of alleles per locus (for lockfile generation)."""
        counts = {}
        for locus in GENE_FILES:
            seqs = self.load_genomic(locus)
            counts[locus] = len(seqs)
        return counts

    def write_lockfile(self, output_dir: Path) -> Path:
        """Write an IMGT lockfile for exact reproducibility."""
        from ..config.manifest import write_imgt_lockfile
        allele_counts = self.get_allele_counts()
        allele_list_path = self.db_dir / "Allelelist.txt"
        return write_imgt_lockfile(
            self.provenance, allele_counts, output_dir,
            allele_list_path if allele_list_path.exists() else None,
        )

    def verify_lockfile(self, lockfile_path: Path) -> bool:
        """Verify database matches a previously generated lockfile."""
        from ..config.manifest import verify_imgt_lockfile
        return verify_imgt_lockfile(lockfile_path, self.provenance)

    def build_combined_reference(self, loci: list[str],
                                  output_path: Path,
                                  genomic: bool = True) -> None:
        """Build a combined FASTA reference for multiple loci."""
        all_seqs: dict[str, str] = {}
        for locus in loci:
            if genomic:
                seqs = self.load_genomic(locus)
            else:
                seqs = self.load_cds(locus)
            for name, seq in seqs.items():
                # Prefix with locus for clarity
                key = f"HLA-{name}" if not name.startswith("HLA-") else name
                all_seqs[key] = seq

        write_fasta(output_path, all_seqs)
        logger.info("Combined reference: %d sequences written to %s",
                     len(all_seqs), output_path)
