"""Comprehensive unit tests for all HLA-Unified V2 modules."""

import math
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ============ config/schema.py ============

class TestPipelineConfig:
    def test_from_cli_defaults(self):
        from hla_unified.config.schema import PipelineConfig
        cfg = PipelineConfig.from_cli(imgt_db="/tmp/db", out="/tmp/out")
        assert cfg.data_type == "short"
        assert cfg.threads == 4
        assert len(cfg.loci) == 8

    def test_from_cli_with_profile(self):
        from hla_unified.config.schema import PipelineConfig
        cfg = PipelineConfig.from_cli(
            imgt_db="/tmp/db", out="/tmp/out", profile_name="transplant",
        )
        assert cfg.profile is not None
        assert cfg.profile.name == "transplant"
        assert cfg.clinical_summary is True

    def test_effective_loci_from_profile(self):
        from hla_unified.config.schema import PipelineConfig
        cfg = PipelineConfig.from_cli(
            imgt_db="/tmp/db", out="/tmp/out", profile_name="immuno_onc",
        )
        assert cfg.effective_loci() == ["A", "B", "C"]

    def test_effective_resolution(self):
        from hla_unified.config.schema import PipelineConfig
        cfg = PipelineConfig.from_cli(
            imgt_db="/tmp/db", out="/tmp/out", profile_name="transplant",
        )
        assert cfg.effective_resolution() == 2

    def test_effective_assay(self):
        from hla_unified.config.schema import PipelineConfig
        cfg = PipelineConfig.from_cli(
            imgt_db="/tmp/db", out="/tmp/out", data_type="hifi",
        )
        assay = cfg.effective_assay()
        assert assay.name == "hifi"
        assert assay.minimap2_preset == "map-hifi"


class TestAssayPresets:
    def test_all_presets_exist(self):
        from hla_unified.config.presets import ASSAY_PRESETS
        expected = {"short", "exome", "targeted_capture", "pacbio", "hifi", "ont", "rna"}
        assert set(ASSAY_PRESETS.keys()) == expected

    def test_preset_fields(self):
        from hla_unified.config.presets import ASSAY_PRESETS
        for name, preset in ASSAY_PRESETS.items():
            assert preset.name == name
            assert preset.expected_coverage > 0
            assert preset.kmer_k > 0
            assert 0.0 < preset.kmer_min_coverage <= 1.0

    def test_use_case_profiles(self):
        from hla_unified.config.presets import USE_CASE_PROFILES
        assert "transplant" in USE_CASE_PROFILES
        assert "research" in USE_CASE_PROFILES
        assert "immuno_onc" in USE_CASE_PROFILES
        assert USE_CASE_PROFILES["transplant"].clinical_summary is True


# ============ config/manifest.py ============

class TestManifest:
    def test_generate_manifest(self):
        from hla_unified.config.manifest import generate_manifest
        m = generate_manifest({"release": "3.56.0", "git_commit": "abc123"})
        assert m["manifest_version"] == "2.0.0"
        assert m["imgt_database"]["release"] == "3.56.0"
        assert "python_version" in m["platform"]
        assert "samtools" in m["external_tools"]

    def test_write_and_verify_lockfile(self):
        from hla_unified.config.manifest import write_imgt_lockfile, verify_imgt_lockfile
        with tempfile.TemporaryDirectory() as td:
            prov = {"release": "3.56.0", "git_commit": "abc"}
            path = write_imgt_lockfile(prov, {"A": 100, "B": 200}, Path(td))
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["total_alleles"] == 300
            assert verify_imgt_lockfile(path, prov) is True
            assert verify_imgt_lockfile(path, {"release": "3.55.0", "git_commit": "xyz"}) is False


# ============ genotyper/cnv.py ============

class TestCNVEstimator:
    def test_diploid_non_cnv_locus(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("A", locus_reads=100, reference_reads=100)
        assert r.copy_number == CopyNumber.DIPLOID

    def test_absent(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB3", locus_reads=2, reference_reads=200)
        assert r.copy_number == CopyNumber.ABSENT

    def test_hemizygous(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB4", locus_reads=60, reference_reads=200)
        assert r.copy_number == CopyNumber.HEMIZYGOUS

    def test_diploid_cnv_locus(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB5", locus_reads=180, reference_reads=200)
        assert r.copy_number == CopyNumber.DIPLOID

    def test_low_reference_fallback(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        r = est.estimate("DRB3", locus_reads=5, reference_reads=3)
        assert r.copy_number == CopyNumber.DIPLOID
        assert r.confidence == "LOW"

    def test_kmer_override(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator, CopyNumber
        est = CopyNumberEstimator()
        # Depth suggests hemizygous but k-mer says absent
        r = est.estimate("DRB3", locus_reads=60, reference_reads=200, kmer_coverage=0.05)
        assert r.copy_number == CopyNumber.ABSENT
        assert r.method == "kmer_override"

    def test_estimate_all(self):
        from hla_unified.genotyper.cnv import CopyNumberEstimator
        est = CopyNumberEstimator()
        results = est.estimate_all(
            ["A", "DRB1", "DRB3", "DRB4"],
            {"A": 100, "DRB1": 200, "DRB3": 5, "DRB4": 90},
        )
        assert "DRB3" in results
        assert "DRB4" in results
        assert "A" not in results  # not a CNV locus


# ============ qc/sample_qc.py — Dropout ============

class TestAlleleDropoutDetector:
    def _make_ilp(self, homo=True):
        from hla_unified.genotyper.ilp_solver import ILPResult
        return ILPResult(
            locus="A", allele1="A*02:01", allele2="A*02:01" if homo else "A*01:01",
            reads_explained=50, total_reads=60, objective_value=100.0,
            is_homozygous=homo, solver_status="Optimal",
        )

    def test_heterozygous_no_risk(self):
        from hla_unified.qc.sample_qc import AlleleDropoutDetector
        det = AlleleDropoutDetector()
        r = det.assess("A", ilp_result=self._make_ilp(homo=False))
        assert r.dropout_risk == "NONE"
        assert r.is_homozygous_call is False

    def test_true_homozygous_no_risk(self):
        from hla_unified.qc.sample_qc import AlleleDropoutDetector
        det = AlleleDropoutDetector()
        r = det.assess("A", ilp_result=self._make_ilp(homo=True),
                        locus_reads=100, reference_locus_reads=100)
        assert r.dropout_risk == "NONE"

    def test_dropout_low_depth(self):
        from hla_unified.qc.sample_qc import AlleleDropoutDetector
        det = AlleleDropoutDetector()
        r = det.assess("A", ilp_result=self._make_ilp(homo=True),
                        locus_reads=30, reference_locus_reads=100)
        assert r.dropout_risk in ("LOW", "MEDIUM", "HIGH")

    def test_dropout_with_het_sites(self):
        from hla_unified.qc.sample_qc import AlleleDropoutDetector
        from hla_unified.phasing.haplotype_binner import PhasingResult, HaplotypeBin
        det = AlleleDropoutDetector()
        phasing = PhasingResult(
            locus="A", het_sites=[], n_het_sites=3,
            bins=[HaplotypeBin(0, [], {}), HaplotypeBin(1, [], {})],
            is_phased=False, phase_confidence=0.0,
        )
        r = det.assess("A", ilp_result=self._make_ilp(homo=True),
                        phasing_result=phasing,
                        locus_reads=40, reference_locus_reads=100)
        assert r.dropout_risk in ("MEDIUM", "HIGH")


# ============ qc/sample_qc.py — Contamination ============

class TestContaminationDetector:
    def _make_matrix(self, n_reads, n_alleles, fractions):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        matrix = np.zeros((n_reads, n_alleles))
        start = 0
        for i, frac in enumerate(fractions):
            count = int(n_reads * frac)
            matrix[start:start + count, i] = 1.0
            start += count
        names = [f"A*{i:02d}:01" for i in range(1, n_alleles + 1)]
        return ReadAlleleMatrix(
            matrix=matrix,
            read_names=[f"read_{j}" for j in range(n_reads)],
            allele_names=names,
        )

    def test_clean_sample(self):
        from hla_unified.qc.sample_qc import ContaminationDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = ContaminationDetector()
        # 2 alleles, no third
        mat = self._make_matrix(100, 3, [0.5, 0.5, 0.0])
        ilp = ILPResult("A", "A*01:01", "A*02:01", 100, 100, 50.0, False, "Optimal")
        r = det.screen(["A"], {"A": mat}, {"A": ilp})
        assert r.is_contaminated is False

    def test_contaminated_sample(self):
        from hla_unified.qc.sample_qc import ContaminationDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = ContaminationDetector(min_loci_flagged=1)
        # 3 alleles with substantial support
        mat = self._make_matrix(100, 3, [0.40, 0.35, 0.25])
        ilp = ILPResult("A", "A*01:01", "A*02:01", 75, 100, 50.0, False, "Optimal")
        r = det.screen(["A"], {"A": mat}, {"A": ilp})
        assert r.is_contaminated is True
        assert "A" in r.loci_with_extra_alleles

    def test_low_reads_not_flagged(self):
        from hla_unified.qc.sample_qc import ContaminationDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = ContaminationDetector()
        mat = self._make_matrix(5, 3, [0.4, 0.3, 0.3])
        ilp = ILPResult("A", "A*01:01", "A*02:01", 5, 5, 5.0, False, "Optimal")
        r = det.screen(["A"], {"A": mat}, {"A": ilp})
        assert r.is_contaminated is False


# ============ confidence/ambiguity_classifier.py ============

class TestAmbiguityClassifier:
    def test_unambiguous(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier, AmbiguityReason
        from hla_unified.confidence.vb_estimator import ConfidenceResult
        clf = AmbiguityClassifier()
        vb = ConfidenceResult(
            locus="A", allele1="A*02:01", allele2="A*01:01",
            posterior_prob=0.995, allele1_dosage=0.5, allele2_dosage=0.5,
            confidence_class="HIGH", convergence_iterations=10,
            elbo=-100.0, alternative_pairs=[("A*02:01", "A*03:01", 0.003)],
        )
        r = clf.classify("A", vb_result=vb)
        assert r.primary_reason == AmbiguityReason.UNAMBIGUOUS

    def test_low_depth(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier, AmbiguityReason
        from hla_unified.genotyper.ilp_solver import ILPResult
        clf = AmbiguityClassifier()
        ilp = ILPResult("A", "A*02:01", "A*01:01", 3, 5, 10.0, False, "Optimal")
        r = clf.classify("A", ilp_result=ilp)
        assert AmbiguityReason.LOW_DEPTH in [r.primary_reason] + r.secondary_reasons

    def test_close_alleles(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier, AmbiguityReason
        from hla_unified.confidence.vb_estimator import ConfidenceResult
        clf = AmbiguityClassifier()
        vb = ConfidenceResult(
            locus="A", allele1="A*02:01", allele2="A*01:01",
            posterior_prob=0.80, allele1_dosage=0.5, allele2_dosage=0.5,
            confidence_class="LOW", convergence_iterations=10,
            elbo=-100.0, alternative_pairs=[("A*02:02", "A*01:01", 0.75)],
        )
        seqs = {"A*02:01": "ACGTACGT" * 100, "A*02:02": "ACGTACGT" * 100}
        r = clf.classify("A", vb_result=vb, allele_sequences=seqs)
        assert AmbiguityReason.CLOSE_ALLELES in [r.primary_reason] + r.secondary_reasons

    def test_exon_only_evidence(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier, AmbiguityReason
        from hla_unified.confidence.vb_estimator import ConfidenceResult
        clf = AmbiguityClassifier()
        vb = ConfidenceResult(
            locus="A", allele1="A*02:01:01:01", allele2="A*01:01",
            posterior_prob=0.85, allele1_dosage=0.5, allele2_dosage=0.5,
            confidence_class="MEDIUM", convergence_iterations=10,
            elbo=-100.0,
            alternative_pairs=[("A*02:01:01:02", "A*01:01", 0.80)],
        )
        r = clf.classify("A", vb_result=vb, data_type="exome")
        assert AmbiguityReason.EXON_ONLY_EVIDENCE in [r.primary_reason] + r.secondary_reasons

    def test_classify_all_loci(self):
        from hla_unified.confidence.ambiguity_classifier import AmbiguityClassifier
        clf = AmbiguityClassifier()
        results = clf.classify_all_loci(
            loci=["A", "B"],
            vb_results={}, kmer_results={}, phasing_results={}, ilp_results={},
        )
        assert "A" in results
        assert "B" in results


# ============ output/clinical.py ============

class TestClinicalReporter:
    def _make_result(self):
        from hla_unified.pipeline.runner import PipelineResult, LocusCall, LocusScore
        calls = {
            "A": LocusCall(
                locus="A", allele1="A*02:01", allele2="A*01:01",
                g_group1="A*02:01G", g_group2="A*01:01G",
                confidence="HIGH", posterior=0.99,
                reads_explained=500, total_reads=520,
                kmer_covered=0.98, kmer_concordant=True,
                is_novel=False, gl_string="HLA-A*02:01+HLA-A*01:01",
                score=LocusScore(0.96, 0.15, 0.99, "HIGH"),
            ),
            "B": LocusCall(
                locus="B", allele1="B*07:02", allele2="B*08:01",
                g_group1="B*07:02G", g_group2="B*08:01G",
                confidence="LOW", posterior=0.45,
                reads_explained=20, total_reads=50,
                kmer_covered=0.60, kmer_concordant=False,
                is_novel=False, gl_string="HLA-B*07:02+HLA-B*08:01",
                score=LocusScore(0.40, 0.02, 0.45, "LOW"),
            ),
        }
        return PipelineResult(
            calls=calls, runtime_seconds=30.0,
            phases_completed=["extraction", "prefilter"],
            imgt_release="3.56.0", imgt_commit="abc123",
        )

    def test_generate_withholds_low_confidence(self):
        from hla_unified.output.clinical import ClinicalReporter
        reporter = ClinicalReporter(min_confidence="MEDIUM")
        summary = reporter.generate(self._make_result())
        assert "A" in summary.loci_reported
        assert "B" in summary.loci_withheld

    def test_write_json(self):
        from hla_unified.output.clinical import ClinicalReporter
        reporter = ClinicalReporter()
        summary = reporter.generate(self._make_result())
        with tempfile.TemporaryDirectory() as td:
            path = reporter.write_clinical_json(summary, Path(td))
            assert path.exists()
            data = json.loads(path.read_text())
            assert "disclaimer" in data
            assert "calls" in data

    def test_write_text(self):
        from hla_unified.output.clinical import ClinicalReporter
        reporter = ClinicalReporter()
        summary = reporter.generate(self._make_result())
        with tempfile.TemporaryDirectory() as td:
            path = reporter.write_clinical_text(summary, Path(td))
            assert path.exists()
            text = path.read_text()
            assert "DISCLAIMER" in text
            assert "HLA-A" in text


# ============ output/writer.py ============

class TestOutputWriter:
    def _make_result(self):
        from hla_unified.pipeline.runner import PipelineResult, LocusCall, LocusScore
        calls = {
            "A": LocusCall(
                locus="A", allele1="A*02:01", allele2="A*01:01",
                g_group1="A*02:01G", g_group2="A*01:01G",
                confidence="HIGH", posterior=0.99,
                reads_explained=500, total_reads=520,
                kmer_covered=0.98, kmer_concordant=True,
                is_novel=False, gl_string="HLA-A*02:01+HLA-A*01:01",
                score=LocusScore(0.96, 0.15, 0.99, "HIGH"),
            ),
        }
        return PipelineResult(
            calls=calls, runtime_seconds=10.0,
            phases_completed=["extraction"],
            imgt_release="3.56.0", imgt_commit="abc",
        )

    def test_write_all(self):
        from hla_unified.output.writer import OutputWriter
        with tempfile.TemporaryDirectory() as td:
            w = OutputWriter(out_dir=Path(td))
            paths = w.write_all(self._make_result())
            assert paths["tsv"].exists()
            assert paths["json"].exists()
            assert paths["ambiguity"].exists()

    def test_json_structure(self):
        from hla_unified.output.writer import OutputWriter
        with tempfile.TemporaryDirectory() as td:
            w = OutputWriter(out_dir=Path(td))
            paths = w.write_all(self._make_result())
            data = json.loads(paths["json"].read_text())
            assert "metadata" in data
            assert "calls" in data
            assert "A" in data["calls"]
            assert data["calls"]["A"]["allele1"] == "A*02:01"


# ============ novel/detector.py ============

class TestNovelAlleleDetector:
    def test_no_signal(self):
        from hla_unified.novel.detector import NovelAlleleDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = NovelAlleleDetector()
        ilp = ILPResult("A", "A*02:01", "A*01:01", 95, 100, 50.0, False, "Optimal")
        r = det.screen_locus("A", ilp_result=ilp)
        assert r is None

    def test_high_unexplained_reads(self):
        from hla_unified.novel.detector import NovelAlleleDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = NovelAlleleDetector(unexplained_read_threshold=0.10)
        ilp = ILPResult("A", "A*02:01", "A*01:01", 60, 100, 50.0, False, "Optimal")
        r = det.screen_locus("A", ilp_result=ilp)
        assert r is not None
        assert "unexplained" in r.detection_reason.lower()

    def test_screen_all_loci(self):
        from hla_unified.novel.detector import NovelAlleleDetector
        from hla_unified.genotyper.ilp_solver import ILPResult
        det = NovelAlleleDetector()
        results = det.screen_all_loci(
            ["A", "B"],
            {"A": ILPResult("A", "A*02:01", "A*01:01", 95, 100, 50.0, False, "Optimal")},
            {}, {},
        )
        assert isinstance(results, dict)


# ============ novel/annotator.py ============

class TestNovelAlleleAnnotator:
    def test_annotate(self):
        from hla_unified.novel.annotator import NovelAlleleAnnotator
        from hla_unified.assembly.targeted_assembler import NovelAlleleReport, MismatchAnnotation
        ann = NovelAlleleAnnotator()
        report = NovelAlleleReport(
            closest_allele="A*02:01",
            identity=0.998,
            mismatches=[
                MismatchAnnotation(position=750, ref_base="A", alt_base="G", variant_type="SNP"),
                MismatchAnnotation(position=1300, ref_base="C", alt_base="T", variant_type="SNP"),
            ],
            n_snps=2, n_insertions=0, n_deletions=0,
            aligned_length=3000, summary="test",
        )
        result = ann.annotate("A", "A*02:01", "A" * 3500, report)
        assert result.locus == "A"
        assert len(result.variants) == 2
        assert result.temporary_designation == "A*02:new"

    def test_to_dict(self):
        from hla_unified.novel.annotator import NovelAlleleAnnotator
        from hla_unified.assembly.targeted_assembler import NovelAlleleReport, MismatchAnnotation
        ann = NovelAlleleAnnotator()
        report = NovelAlleleReport(
            closest_allele="B*07:02", identity=0.997,
            mismatches=[MismatchAnnotation(500, "G", "T", "SNP")],
            n_snps=1, n_insertions=0, n_deletions=0,
            aligned_length=3000, summary="test",
        )
        result = ann.annotate("B", "B*07:02", "A" * 3000, report)
        d = result.to_dict()
        assert "variants" in d
        assert d["closest_allele"] == "B*07:02"


# ============ benchmark/metrics.py ============

class TestBenchmarkMetrics:
    def test_compare_diploid_exact_match(self):
        from hla_unified.benchmark.metrics import compare_diploid
        assert compare_diploid(("A*02:01", "A*01:01"), ("A*02:01", "A*01:01")) == 2

    def test_compare_diploid_swapped(self):
        from hla_unified.benchmark.metrics import compare_diploid
        assert compare_diploid(("A*01:01", "A*02:01"), ("A*02:01", "A*01:01")) == 2

    def test_compare_diploid_one_match(self):
        from hla_unified.benchmark.metrics import compare_diploid
        assert compare_diploid(("A*02:01", "A*03:01"), ("A*02:01", "A*01:01")) == 1

    def test_compare_diploid_no_match(self):
        from hla_unified.benchmark.metrics import compare_diploid
        assert compare_diploid(("A*03:01", "A*11:01"), ("A*02:01", "A*01:01")) == 0

    def test_compare_diploid_no_call(self):
        from hla_unified.benchmark.metrics import compare_diploid
        assert compare_diploid(("", ""), ("A*02:01", "A*01:01")) == -1

    def test_compare_at_resolution(self):
        from hla_unified.benchmark.metrics import compare_diploid
        # Different at 4-digit but same at 2-digit
        assert compare_diploid(
            ("A*02:01:01:01", "A*01:01:01:01"),
            ("A*02:01:01:02", "A*01:01:01:03"),
            resolution=2,
        ) == 2

    def test_compute_accuracy(self):
        from hla_unified.benchmark.metrics import compute_accuracy
        calls = {"A": ("A*02:01", "A*01:01"), "B": ("B*07:02", "B*99:99")}
        truth = {"A": ("A*02:01", "A*01:01"), "B": ("B*07:02", "B*08:01")}
        accs = compute_accuracy(calls, truth)
        assert accs["A"].accuracy == 1.0
        assert accs["B"].accuracy == 0.5

    def test_merge_accuracies(self):
        from hla_unified.benchmark.metrics import LocusAccuracy, merge_accuracies
        a1 = {"A": LocusAccuracy(locus="A", n_samples=1, n_correct_both=1)}
        a2 = {"A": LocusAccuracy(locus="A", n_samples=1, n_correct_one=1)}
        merged = merge_accuracies([a1, a2])
        assert merged["A"].n_samples == 2
        assert merged["A"].accuracy == 0.75  # (2 + 1) / 4


# ============ qc/locus_metrics.py ============

class TestLocusMetrics:
    def test_calculate_basic(self):
        from hla_unified.qc.locus_metrics import LocusMetricsCalculator
        calc = LocusMetricsCalculator()
        qc = calc.calculate("A", reads_explained=90, total_reads=100)
        assert qc.locus == "A"
        assert qc.total_reads == 100

    def test_calculate_with_phasing(self):
        from hla_unified.qc.locus_metrics import LocusMetricsCalculator
        from hla_unified.phasing.haplotype_binner import PhasingResult, HaplotypeBin, HetSite
        calc = LocusMetricsCalculator()
        phasing = PhasingResult(
            locus="A",
            het_sites=[
                HetSite("A*02:01", 100, {"A": 20, "G": 18}, "A", "G", 38),
                HetSite("A*02:01", 200, {"C": 15, "T": 12}, "C", "T", 27),
            ],
            n_het_sites=2,
            bins=[
                HaplotypeBin(0, [f"r{i}" for i in range(25)], {}),
                HaplotypeBin(1, [f"r{i}" for i in range(20)], {}),
            ],
            is_phased=True, phase_confidence=0.9,
        )
        qc = calc.calculate("A", phasing_result=phasing)
        assert qc.is_phased is True
        assert qc.haplotype_balance == pytest.approx(20 / 25, abs=0.01)
        assert qc.n_informative_positions == 2

    def test_to_dict(self):
        from hla_unified.qc.locus_metrics import LocusMetricsCalculator
        calc = LocusMetricsCalculator()
        qc = calc.calculate("A")
        d = qc.to_dict()
        assert "haplotype_balance" in d
        assert "informative_positions" in d
        assert "verdict" in d
