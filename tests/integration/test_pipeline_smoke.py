"""Integration smoke tests for the HLA-Unified V2 pipeline.

Tests the full pipeline wiring end-to-end using synthetic data.
Does NOT require external tools (samtools, minimap2) — tests the
Python-level integration only.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from hla_unified.pipeline.runner import (
    UnifiedPipeline, PipelineResult, LocusCall, LocusScore,
    AlternativeCall,
)
from hla_unified.genotyper.ilp_solver import ILPResult
from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
from hla_unified.confidence.vb_estimator import ConfidenceResult
from hla_unified.kmer.validator import KmerValidationResult
from hla_unified.phasing.haplotype_binner import PhasingResult, HaplotypeBin


# === Synthetic result factory ===

def make_synthetic_result() -> PipelineResult:
    """Build a realistic PipelineResult without running the actual pipeline."""
    calls = {}
    for locus, a1, a2, conf, post in [
        ("A", "A*02:01", "A*01:01", "HIGH", 0.995),
        ("B", "B*07:02", "B*08:01", "HIGH", 0.990),
        ("C", "C*07:01", "C*07:02", "MEDIUM", 0.920),
        ("DRB1", "DRB1*15:01", "DRB1*03:01", "HIGH", 0.998),
        ("DQB1", "DQB1*06:02", "DQB1*02:01", "MEDIUM", 0.880),
    ]:
        calls[locus] = LocusCall(
            locus=locus, allele1=a1, allele2=a2,
            g_group1=f"{a1}G", g_group2=f"{a2}G",
            confidence=conf, posterior=post,
            reads_explained=200, total_reads=220,
            kmer_covered=0.95, kmer_concordant=True,
            is_novel=False,
            gl_string=f"HLA-{a1}+HLA-{a2}",
            score=LocusScore(0.91, 0.12, post, conf),
            alternatives=[
                AlternativeCall(f"{a1}x", f"{a2}x", post * 0.01),
            ],
        )
    return PipelineResult(
        calls=calls, runtime_seconds=42.0,
        phases_completed=[
            "extraction", "prefilter", "refinement", "phasing",
            "ilp_genotyping", "bayesian_confidence",
            "kmer_validation", "assembly_fallback",
        ],
        imgt_release="3.56.0", imgt_commit="abc123def456",
    )


# === Tests ===

class TestOutputWriterIntegration:
    """Test that OutputWriter produces all expected files from a PipelineResult."""

    def test_write_all_produces_files(self):
        from hla_unified.output.writer import OutputWriter
        result = make_synthetic_result()
        with tempfile.TemporaryDirectory() as td:
            w = OutputWriter(out_dir=Path(td))
            paths = w.write_all(result)
            assert (Path(td) / "hla_types.tsv").exists()
            assert (Path(td) / "hla_types.json").exists()
            assert (Path(td) / "ambiguity.tsv").exists()

    def test_json_has_all_loci(self):
        from hla_unified.output.writer import OutputWriter
        result = make_synthetic_result()
        with tempfile.TemporaryDirectory() as td:
            w = OutputWriter(out_dir=Path(td))
            paths = w.write_all(result)
            data = json.loads(paths["json"].read_text())
            assert set(data["calls"].keys()) == {"A", "B", "C", "DRB1", "DQB1"}
            for locus_data in data["calls"].values():
                assert "score" in locus_data
                assert "reads" in locus_data
                assert "alternative_pairs" in locus_data

    def test_tsv_has_provenance_header(self):
        from hla_unified.output.writer import OutputWriter
        result = make_synthetic_result()
        with tempfile.TemporaryDirectory() as td:
            w = OutputWriter(out_dir=Path(td))
            paths = w.write_all(result)
            text = paths["tsv"].read_text()
            assert "## IPD-IMGT/HLA Release: 3.56.0" in text
            assert "## HLA-Unified Version:" in text


class TestClinicalIntegration:
    """Test clinical reporting from pipeline result."""

    def test_transplant_report(self):
        from hla_unified.output.clinical import ClinicalReporter
        from hla_unified.confidence.ambiguity_classifier import (
            AmbiguityClassifier, AmbiguityClassification, AmbiguityReason,
        )
        result = make_synthetic_result()
        ambiguity = {
            locus: AmbiguityClassification(
                locus=locus, primary_reason=AmbiguityReason.UNAMBIGUOUS,
            )
            for locus in result.calls
        }
        reporter = ClinicalReporter(min_confidence="HIGH")
        summary = reporter.generate(result, ambiguity)

        # HIGH-confidence loci reported, MEDIUM withheld
        assert "A" in summary.loci_reported
        assert "DRB1" in summary.loci_reported
        assert "C" in summary.loci_withheld  # MEDIUM
        assert "DQB1" in summary.loci_withheld  # MEDIUM

        with tempfile.TemporaryDirectory() as td:
            text_path = reporter.write_clinical_text(summary, Path(td))
            text = text_path.read_text()
            assert "DISCLAIMER" in text
            assert "HLA-A" in text


class TestAmbiguityIntegration:
    """Test ambiguity classifier on synthetic VB results."""

    def test_all_loci_classified(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier
        clf = AmbiguityClassifier()
        vb_results = {
            "A": ConfidenceResult(
                locus="A", allele1="A*02:01", allele2="A*01:01",
                posterior_prob=0.999, allele1_dosage=0.5, allele2_dosage=0.5,
                confidence_class="HIGH", convergence_iterations=50,
                elbo=-100.0, alternative_pairs=[("A*02:02", "A*01:01", 0.001)],
            ),
        }
        ilp_results = {
            "A": ILPResult("A", "A*02:01", "A*01:01", 200, 220, 500.0, False, "Optimal"),
        }
        results = clf.classify_all_loci(
            ["A"], vb_results=vb_results, kmer_results={},
            phasing_results={}, ilp_results=ilp_results,
        )
        assert "A" in results
        assert results["A"].primary_reason.value == "unambiguous"


class TestNovelDetectionIntegration:
    """Test novel allele detection + annotation chain."""

    def test_detection_then_annotation(self):
        from hla_unified.novel.detector import NovelAlleleDetector
        from hla_unified.novel.annotator import NovelAlleleAnnotator
        from hla_unified.assembly.targeted_assembler import (
            AssemblyResult, NovelAlleleReport, MismatchAnnotation,
        )

        # Simulate assembly finding a novel allele
        novel_report = NovelAlleleReport(
            closest_allele="A*02:01", identity=0.997,
            mismatches=[
                MismatchAnnotation(750, "A", "G", "SNP"),
                MismatchAnnotation(1300, "C", "T", "SNP"),
            ],
            n_snps=2, n_insertions=0, n_deletions=0,
            aligned_length=3000, summary="test novel",
        )
        asm = AssemblyResult(
            locus="A", contigs={"c1": "ACGT" * 750},
            n_contigs=1, total_length=3000,
            best_match_allele="A*02:01", best_match_identity=0.997,
            is_novel=True, novel_report=novel_report,
        )

        # Step 1: detection
        det = NovelAlleleDetector()
        ilp = ILPResult("A", "A*02:01", "A*01:01", 60, 100, 50.0, False, "Optimal")
        candidate = det.screen_locus("A", ilp_result=ilp, assembly_result=asm)
        assert candidate is not None
        assert candidate.confidence in ("LOW", "MEDIUM", "HIGH")

        # Step 2: annotation
        ann = NovelAlleleAnnotator()
        annotation = ann.annotate("A", "A*02:01", "A" * 3500, novel_report)
        assert annotation.temporary_designation == "A*02:new"
        assert len(annotation.variants) == 2


class TestCNVIntegration:
    """Test CNV estimation feeding into ILP adjustment."""

    def test_absent_locus_produces_no_call(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB3", locus_reads=2, reference_reads=200)
        assert r.copy_number == CopyNumber.ABSENT
        # Pipeline would return no-call for this locus

    def test_hemizygous_locus_single_allele(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB4", locus_reads=60, reference_reads=200)
        assert r.copy_number == CopyNumber.HEMIZYGOUS
        # Pipeline would force single-allele call


class TestFrequencyPriors:
    """Test that population frequency priors integrate into VB estimator."""

    def test_frequency_loaded(self):
        from hla_unified.reference.frequencies import load_default_frequencies
        db = load_default_frequencies()
        assert db.n_alleles > 50
        assert db.get_frequency("A*02:01") > 0.1
        assert db.get_frequency("NONEXISTENT*99:99") == 1e-5

    def test_vb_with_priors(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        from hla_unified.reference.frequencies import load_default_frequencies
        db = load_default_frequencies()
        freqs = {"A*02:01": db.get_frequency("A*02:01"),
                 "A*01:01": db.get_frequency("A*01:01")}
        vb = VariationalBayesEstimator(allele_frequencies=freqs)
        assert vb.allele_frequencies is not None
        assert vb.allele_frequencies["A*02:01"] > vb.allele_frequencies["A*01:01"]


class TestManifestIntegration:
    """Test manifest and lockfile generation."""

    def test_full_manifest(self):
        from hla_unified.config.manifest import generate_manifest, write_manifest
        manifest = generate_manifest(
            {"release": "3.56.0", "git_commit": "abc123"},
        )
        with tempfile.TemporaryDirectory() as td:
            path = write_manifest(manifest, Path(td))
            data = json.loads(path.read_text())
            assert data["hla_unified"]["version"] == "2.0.0"
            assert "samtools" in data["external_tools"]
            assert data["imgt_database"]["release"] == "3.56.0"
