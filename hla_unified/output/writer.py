"""Consolidated output writer for HLA-Unified V2.

Produces all output formats from a single PipelineResult:
- TSV with provenance header (backward-compatible)
- Full JSON with complete evidence trail
- Ambiguity TSV with ranked alternatives
- QC JSON + HTML dashboard
- Clinical summary (when enabled)
- Reproducibility manifest + IMGT lockfile
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..pipeline.runner import PipelineResult, LocusCall
from ..confidence.ambiguity_classifier import AmbiguityClassification

logger = logging.getLogger(__name__)


class OutputWriter:
    """Writes all pipeline outputs to a directory."""

    def __init__(
        self,
        out_dir: Path,
        data_type: str = "short",
        output_resolution: int | str = "max",
        version: str = "2.0.0",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.data_type = data_type
        self.output_resolution = output_resolution
        self.version = version
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        result: PipelineResult,
        ambiguity: dict[str, AmbiguityClassification] | None = None,
        locus_metrics: dict[str, Any] | None = None,
        manifest: dict | None = None,
    ) -> dict[str, Path]:
        """Write all output files and return their paths."""
        paths = {}

        paths["tsv"] = self.write_tsv(result)
        paths["json"] = self.write_json(result, ambiguity, locus_metrics)
        paths["ambiguity"] = self.write_ambiguity_tsv(result)

        if manifest:
            paths["manifest"] = self._write_json_file(
                manifest, "manifest.json",
            )

        return paths

    def write_tsv(self, result: PipelineResult) -> Path:
        """Write main TSV output with provenance header."""
        path = self.out_dir / "hla_types.tsv"

        fieldnames = [
            "Locus", "Chromosome", "Allele", "G_Group", "GL_String",
            "Confidence", "Posterior", "EvidenceScore", "AmbiguityGap",
            "AmbiguityReason", "ReadsExplained", "TotalReads",
            "KmerCovered", "KmerConcordant", "IsNovel", "Flags",
        ]

        with open(path, "w", newline="") as fh:
            fh.write(f"## IPD-IMGT/HLA Release: {result.imgt_release}\n")
            fh.write(f"## IMGT Commit: {result.imgt_commit}\n")
            fh.write(f"## HLA-Unified Version: {self.version}\n")
            fh.write(f"## Data Type: {self.data_type}\n")
            fh.write(f"## Output Resolution: {self.output_resolution}\n")

            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for locus in sorted(result.calls.keys()):
                call = result.calls[locus]
                ev = call.score.evidence_score if call.score else 0.0
                ag = call.score.ambiguity_gap if call.score else 0.0
                ambig_reason = ""
                if hasattr(call, "ambiguity") and call.ambiguity:
                    ambig_reason = call.ambiguity.primary_reason.value

                for chrom, allele, ggroup in [
                    (1, call.allele1, call.g_group1),
                    (2, call.allele2, call.g_group2),
                ]:
                    writer.writerow({
                        "Locus": f"HLA-{locus}",
                        "Chromosome": chrom,
                        "Allele": allele,
                        "G_Group": ggroup,
                        "GL_String": call.gl_string,
                        "Confidence": call.confidence,
                        "Posterior": f"{call.posterior:.4f}",
                        "EvidenceScore": f"{ev:.4f}",
                        "AmbiguityGap": f"{ag:.4f}",
                        "AmbiguityReason": ambig_reason,
                        "ReadsExplained": call.reads_explained,
                        "TotalReads": call.total_reads,
                        "KmerCovered": f"{call.kmer_covered:.4f}",
                        "KmerConcordant": call.kmer_concordant,
                        "IsNovel": call.is_novel,
                        "Flags": ";".join(call.flags) if call.flags else "",
                    })

        logger.info("Results written to %s", path)
        return path

    def write_json(
        self,
        result: PipelineResult,
        ambiguity: dict[str, AmbiguityClassification] | None = None,
        locus_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Write full JSON output with complete evidence trail.

        Structure:
        {
          "metadata": { imgt_release, version, config, runtime },
          "calls": {
            per-locus: {
              alleles, gl_string, confidence, score,
              alternatives, ambiguity, qc_metrics
            }
          }
        }
        """
        data: dict[str, Any] = {
            "metadata": {
                "hla_unified_version": self.version,
                "imgt_release": result.imgt_release,
                "imgt_commit": result.imgt_commit,
                "data_type": self.data_type,
                "output_resolution": str(self.output_resolution),
                "phases_completed": result.phases_completed,
                "runtime_seconds": round(result.runtime_seconds, 2),
            },
            "calls": {},
        }

        for locus in sorted(result.calls.keys()):
            call = result.calls[locus]
            locus_data: dict[str, Any] = {
                "allele1": call.allele1,
                "allele2": call.allele2,
                "g_group1": call.g_group1,
                "g_group2": call.g_group2,
                "gl_string": call.gl_string,
                "confidence": call.confidence,
                "posterior": round(call.posterior, 6),
                "is_novel": call.is_novel,
                "novel_summary": call.novel_summary,
                "flags": call.flags,
            }

            # Scoring breakdown
            if call.score:
                locus_data["score"] = {
                    "evidence_score": call.score.evidence_score,
                    "ambiguity_gap": call.score.ambiguity_gap,
                    "confidence_posterior": call.score.confidence_posterior,
                    "confidence_tier": call.score.confidence_tier,
                }

            # Read evidence
            locus_data["reads"] = {
                "explained": call.reads_explained,
                "total": call.total_reads,
                "fraction_explained": round(
                    call.reads_explained / max(call.total_reads, 1), 4,
                ),
            }

            # K-mer validation
            locus_data["kmer_validation"] = {
                "coverage": round(call.kmer_covered, 4),
                "concordant": call.kmer_concordant,
            }

            # Ranked alternatives
            if call.alternatives:
                locus_data["alternative_pairs"] = [
                    {
                        "rank": i + 2,
                        "allele1": alt.allele1,
                        "allele2": alt.allele2,
                        "posterior": round(alt.posterior, 6),
                    }
                    for i, alt in enumerate(call.alternatives)
                ]

            # Ambiguity classification
            if ambiguity and locus in ambiguity:
                locus_data["ambiguity"] = ambiguity[locus].to_dict()

            # Detailed QC metrics
            if locus_metrics and locus in locus_metrics:
                metrics = locus_metrics[locus]
                if hasattr(metrics, "to_dict"):
                    locus_data["detailed_qc"] = metrics.to_dict()
                elif isinstance(metrics, dict):
                    locus_data["detailed_qc"] = metrics

            data["calls"][locus] = locus_data

        path = self.out_dir / "hla_types.json"
        path.write_text(json.dumps(data, indent=2))
        logger.info("JSON output written to %s", path)
        return path

    def write_ambiguity_tsv(self, result: PipelineResult) -> Path:
        """Write ranked alternative calls per locus."""
        path = self.out_dir / "ambiguity.tsv"

        fieldnames = [
            "Locus", "Rank", "Allele1", "Allele2",
            "Posterior", "IsBestCall", "AmbiguityReason",
        ]

        with open(path, "w", newline="") as fh:
            fh.write(f"## IPD-IMGT/HLA Release: {result.imgt_release}\n")
            fh.write("## Ranked alternative diploid genotype calls per locus\n")

            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for locus in sorted(result.calls.keys()):
                call = result.calls[locus]
                if not call.allele1:
                    continue

                ambig_reason = ""
                if hasattr(call, "ambiguity") and call.ambiguity:
                    ambig_reason = call.ambiguity.primary_reason.value

                writer.writerow({
                    "Locus": f"HLA-{locus}",
                    "Rank": 1,
                    "Allele1": call.allele1,
                    "Allele2": call.allele2,
                    "Posterior": f"{call.posterior:.4f}",
                    "IsBestCall": True,
                    "AmbiguityReason": ambig_reason,
                })

                for rank, alt in enumerate(call.alternatives, start=2):
                    writer.writerow({
                        "Locus": f"HLA-{locus}",
                        "Rank": rank,
                        "Allele1": alt.allele1,
                        "Allele2": alt.allele2,
                        "Posterior": f"{alt.posterior:.4f}",
                        "IsBestCall": False,
                        "AmbiguityReason": "",
                    })

        logger.info("Ambiguity report written to %s", path)
        return path

    def _write_json_file(self, data: dict, filename: str) -> Path:
        path = self.out_dir / filename
        path.write_text(json.dumps(data, indent=2))
        return path
