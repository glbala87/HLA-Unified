"""Standard benchmark dataset definitions.

Defines the structure for benchmark datasets with truth sets,
ancestry annotations, and assay metadata. Users register their
own datasets; this module provides the schema and built-in
dataset definitions for common evaluation sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkSample:
    """A single sample with known HLA truth."""
    sample_id: str
    bam_path: str
    truth: dict[str, tuple[str, str]]  # locus -> (allele1, allele2)
    ancestry: str = "unknown"
    depth: int = 0
    notes: str = ""


@dataclass
class BenchmarkDataset:
    """A collection of samples for benchmarking."""
    name: str
    description: str
    assay: str
    samples: list[BenchmarkSample]
    truth_source: str = "unknown"
    resolution: int = 2
    ancestry_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def loci(self) -> list[str]:
        all_loci = set()
        for s in self.samples:
            all_loci.update(s.truth.keys())
        return sorted(all_loci)

    @classmethod
    def from_tsv(
        cls,
        name: str,
        truth_tsv: str | Path,
        bam_dir: str | Path,
        assay: str = "short",
        resolution: int = 2,
    ) -> BenchmarkDataset:
        """Load dataset from a truth TSV file.

        Expected TSV columns: SampleID, Locus, Allele1, Allele2
        Optional columns: Ancestry, Depth
        """
        import csv

        samples_dict: dict[str, BenchmarkSample] = {}
        bam_dir = Path(bam_dir)

        with open(truth_tsv) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                sid = row["SampleID"]
                locus = row["Locus"].replace("HLA-", "")
                a1 = row["Allele1"]
                a2 = row["Allele2"]

                if sid not in samples_dict:
                    bam_path = str(bam_dir / f"{sid}.bam")
                    samples_dict[sid] = BenchmarkSample(
                        sample_id=sid,
                        bam_path=bam_path,
                        truth={},
                        ancestry=row.get("Ancestry", "unknown"),
                        depth=int(row.get("Depth", 0)),
                    )

                samples_dict[sid].truth[locus] = (a1, a2)

        samples = list(samples_dict.values())

        # Compute ancestry distribution
        ancestry_dist: dict[str, int] = {}
        for s in samples:
            ancestry_dist[s.ancestry] = ancestry_dist.get(s.ancestry, 0) + 1

        return cls(
            name=name,
            description=f"Dataset loaded from {truth_tsv}",
            assay=assay,
            samples=samples,
            truth_source=str(truth_tsv),
            resolution=resolution,
            ancestry_distribution=ancestry_dist,
        )


# Built-in dataset templates (paths must be configured by user)
BUILTIN_DATASETS: dict[str, dict] = {
    "1000g_30x": {
        "description": "1000 Genomes Project 30x WGS samples with consensus HLA calls",
        "assay": "short",
        "resolution": 2,
        "ancestry_groups": ["AFR", "AMR", "EAS", "EUR", "SAS"],
    },
    "ihiw_wgs": {
        "description": "International HLA and Immunogenetics Workshop WGS samples",
        "assay": "short",
        "resolution": 4,
    },
    "platinum_exome": {
        "description": "Platinum Genomes exome samples with SBT truth",
        "assay": "exome",
        "resolution": 2,
    },
}
