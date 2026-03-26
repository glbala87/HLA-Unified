"""Unit tests for hla_unified package."""

import math
import numpy as np
import pytest


# ============ reference/loci.py ============

class TestAlleleParsing:
    def test_parse_standard(self):
        from hla_unified.reference.loci import parse_allele_name
        info = parse_allele_name("A*02:01:01:01")
        assert info.locus == "A"
        assert info.field1 == "02"
        assert info.field2 == "01"
        assert info.field3 == "01"
        assert info.field4 == "01"
        assert info.two_digit == "A*02"
        assert info.four_digit == "A*02:01"
        assert info.full_name == "A*02:01:01:01"

    def test_parse_with_hla_prefix(self):
        from hla_unified.reference.loci import parse_allele_name
        info = parse_allele_name("HLA-B*07:02")
        assert info.locus == "B"
        assert info.four_digit == "B*07:02"

    def test_parse_with_suffix(self):
        from hla_unified.reference.loci import parse_allele_name
        info = parse_allele_name("A*02:01:01N")
        assert info.suffix == "N"
        assert info.locus == "A"

    def test_group_by_2digit(self):
        from hla_unified.reference.loci import group_alleles_by_resolution
        alleles = ["A*02:01:01", "A*02:01:02", "A*02:02:01", "A*03:01:01"]
        groups = group_alleles_by_resolution(alleles, level=1)
        assert len(groups) == 2
        assert len(groups["A*02"]) == 3
        assert len(groups["A*03"]) == 1

    def test_group_by_4digit(self):
        from hla_unified.reference.loci import group_alleles_by_resolution
        alleles = ["A*02:01:01", "A*02:01:02", "A*02:02:01"]
        groups = group_alleles_by_resolution(alleles, level=2)
        assert len(groups) == 2
        assert len(groups["A*02:01"]) == 2


# ============ utils/seq.py ============

class TestSequenceUtils:
    def test_reverse_complement(self):
        from hla_unified.utils.seq import reverse_complement
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("ATCG") == "CGAT"

    def test_canonical_kmer(self):
        from hla_unified.utils.seq import canonical_kmer
        km = "ACGT"
        rc = "ACGT"  # palindrome
        assert canonical_kmer(km) == canonical_kmer(rc)

    def test_extract_kmers(self):
        from hla_unified.utils.seq import extract_kmers
        seq = "ACGTACGT"
        kmers = extract_kmers(seq, k=4)
        assert len(kmers) == 5
        assert kmers[0] == "ACGT"

    def test_extract_kmers_skips_N(self):
        from hla_unified.utils.seq import extract_kmers
        seq = "ACNGTACGT"
        kmers = extract_kmers(seq, k=4)
        # k-mers containing N should be skipped
        assert all("N" not in km for km in kmers)

    def test_phred_quality(self):
        from hla_unified.utils.seq import phred_char_to_p_correct
        # '!' = Q0 -> p_correct = 0.0
        assert phred_char_to_p_correct("!") == pytest.approx(0.0, abs=0.01)
        # 'I' = Q40 -> p_correct ≈ 0.9999
        assert phred_char_to_p_correct("I") == pytest.approx(0.9999, abs=0.001)


# ============ genotyper/ilp_solver.py ============

class TestILPSolver:
    def _make_matrix(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        return ReadAlleleMatrix(
            matrix=np.array([
                [10, 0, 0, 0],
                [9, 0, 0, 0],
                [0, 0, 12, 0],
                [0, 0, 11, 0],
                [10, 0, 11, 0],
            ], dtype=np.float64),
            read_names=["r1", "r2", "r3", "r4", "r5"],
            allele_names=["A*02:01", "A*02:02", "A*03:01", "A*03:02"],
        )

    def test_selects_two_alleles(self):
        from hla_unified.genotyper.ilp_solver import solve_ilp
        matrix = self._make_matrix()
        result = solve_ilp(matrix, locus="A")
        assert result.solver_status == "Optimal"
        assert result.allele1 != ""
        assert result.allele2 != ""

    def test_selects_correct_pair(self):
        from hla_unified.genotyper.ilp_solver import solve_ilp
        matrix = self._make_matrix()
        result = solve_ilp(matrix, locus="A")
        pair = {result.allele1, result.allele2}
        assert pair == {"A*02:01", "A*03:01"}

    def test_explains_reads(self):
        from hla_unified.genotyper.ilp_solver import solve_ilp
        matrix = self._make_matrix()
        result = solve_ilp(matrix, locus="A")
        assert result.reads_explained == 5
        assert result.total_reads == 5

    def test_single_allele(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        from hla_unified.genotyper.ilp_solver import solve_ilp
        matrix = ReadAlleleMatrix(
            matrix=np.array([[5], [3]], dtype=np.float64),
            read_names=["r1", "r2"],
            allele_names=["A*01:01"],
        )
        result = solve_ilp(matrix, locus="A")
        assert result.is_homozygous
        assert result.allele1 == "A*01:01"

    def test_empty_matrix(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        from hla_unified.genotyper.ilp_solver import solve_ilp
        matrix = ReadAlleleMatrix(
            matrix=np.zeros((0, 0)),
            read_names=[],
            allele_names=[],
        )
        result = solve_ilp(matrix, locus="A")
        assert result.solver_status == "no_candidates"


# ============ confidence/vb_estimator.py ============

class TestVBEstimator:
    def _make_matrix(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        return ReadAlleleMatrix(
            matrix=np.array([
                [10, 8, 0, 0],
                [9, 7, 0, 0],
                [0, 0, 12, 5],
                [0, 0, 11, 6],
                [10, 0, 11, 0],
            ], dtype=np.float64),
            read_names=["r1", "r2", "r3", "r4", "r5"],
            allele_names=["A*02:01", "A*02:02", "A*03:01", "A*03:02"],
        )

    def test_returns_result(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        vb = VariationalBayesEstimator()
        matrix = self._make_matrix()
        result = vb.estimate(matrix, locus="A")
        assert result.allele1 != ""
        assert result.allele2 != ""
        assert 0 <= result.posterior_prob <= 1.0

    def test_confidence_classes(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        vb = VariationalBayesEstimator()
        matrix = self._make_matrix()
        result = vb.estimate(matrix, locus="A")
        assert result.confidence_class in ("HIGH", "MEDIUM", "LOW")

    def test_dosages_sum_to_one(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        vb = VariationalBayesEstimator()
        matrix = self._make_matrix()
        result = vb.estimate(matrix, locus="A")
        assert result.allele1_dosage + result.allele2_dosage == pytest.approx(1.0, abs=0.01)

    def test_elbo_is_finite(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        vb = VariationalBayesEstimator()
        matrix = self._make_matrix()
        result = vb.estimate(matrix, locus="A")
        assert math.isfinite(result.elbo)

    def test_trivial_result(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        vb = VariationalBayesEstimator()
        matrix = ReadAlleleMatrix(
            matrix=np.array([[5]], dtype=np.float64),
            read_names=["r1"],
            allele_names=["A*01:01"],
        )
        result = vb.estimate(matrix, locus="A")
        assert result.allele1 == "A*01:01"


# ============ kmer/validator.py ============

class TestKmerValidator:
    def test_perfect_match(self):
        from hla_unified.kmer.validator import KmerValidator
        kv = KmerValidator(k=5)
        seq = "ACGTACGTACGTACGTACGTACGTACGT"
        result = kv.validate(seq, seq, [seq, seq], locus="A")
        assert result.proportion_kmers_covered == 1.0
        assert result.is_concordant

    def test_no_reads(self):
        from hla_unified.kmer.validator import KmerValidator
        kv = KmerValidator(k=5)
        seq = "ACGTACGTACGTACGTACGTACGTACGT"
        result = kv.validate(seq, seq, [], locus="A")
        assert result.proportion_kmers_covered == 0.0
        assert not result.is_concordant


# ============ genotyper/read_matrix.py ============

class TestReadMatrix:
    def test_subset_locus(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        matrix = ReadAlleleMatrix(
            matrix=np.array([
                [10, 0, 5],
                [0, 8, 3],
            ], dtype=np.float64),
            read_names=["r1", "r2"],
            allele_names=["A*02:01", "B*07:02", "A*03:01"],
        )
        sub = matrix.subset_locus("A")
        assert sub.n_alleles == 2
        assert "A*02:01" in sub.allele_names
        assert "A*03:01" in sub.allele_names
        assert "B*07:02" not in sub.allele_names

    def test_binary_property(self):
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        matrix = ReadAlleleMatrix(
            matrix=np.array([[10, 0], [0, 5]], dtype=np.float64),
            read_names=["r1", "r2"],
            allele_names=["A*01:01", "A*02:01"],
        )
        binary = matrix.binary
        np.testing.assert_array_equal(binary, [[1, 0], [0, 1]])


# ============ reference/g_groups.py ============

class TestGGroupTranslator:
    def test_no_data_passthrough(self):
        from hla_unified.reference.g_groups import GGroupTranslator
        gt = GGroupTranslator()
        allele, perfect = gt.translate("A*02:01:01")
        assert allele == "A*02:01:01"
        assert perfect is False


# ============ Feature 1: Haplotype phasing ============

class TestHaplotypeBinner:
    def test_unphased_result(self):
        from hla_unified.phasing.haplotype_binner import HaplotypeBinner
        binner = HaplotypeBinner()
        result = binner._unphased_result("A")
        assert not result.is_phased
        assert result.n_het_sites == 0
        assert len(result.bins) == 2

    def test_cluster_reads_basic(self):
        from hla_unified.phasing.haplotype_binner import HaplotypeBinner, HetSite
        binner = HaplotypeBinner()
        # 6 reads, 3 het sites, two clear haplotypes
        matrix = np.array([
            [0, 0, 0],   # hap 0: major-major-major
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],   # hap 1: minor-minor-minor
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=np.int8)
        het_sites = [
            HetSite("ref", i, {"A": 3, "C": 3}, "A", "C", 6)
            for i in range(3)
        ]
        assignments = binner._cluster_reads(matrix, het_sites)
        # Should separate into two groups
        assert len(set(assignments[assignments >= 0])) == 2
        # Reads 0-2 should be in same bin, 3-5 in other
        assert assignments[0] == assignments[1] == assignments[2]
        assert assignments[3] == assignments[4] == assignments[5]
        assert assignments[0] != assignments[3]


# ============ Feature 2: Explicit ambiguity model ============

class TestAmbiguityModel:
    def test_alternative_calls_populated(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        vb = VariationalBayesEstimator()
        matrix = ReadAlleleMatrix(
            matrix=np.array([
                [10, 8, 2, 1],
                [9, 7, 1, 2],
                [1, 2, 12, 5],
                [2, 1, 11, 6],
            ], dtype=np.float64),
            read_names=["r1", "r2", "r3", "r4"],
            allele_names=["A*02:01", "A*02:02", "A*03:01", "A*03:02"],
        )
        result = vb.estimate(matrix, locus="A")
        # Should have alternative pairs
        assert len(result.alternative_pairs) > 0
        # Each alternative should have (allele1, allele2, posterior)
        for a1, a2, post in result.alternative_pairs:
            assert isinstance(a1, str)
            assert isinstance(a2, str)
            assert 0 <= post <= 1.0

    def test_posteriors_sum_to_one(self):
        from hla_unified.confidence.vb_estimator import VariationalBayesEstimator
        from hla_unified.genotyper.read_matrix import ReadAlleleMatrix
        vb = VariationalBayesEstimator()
        matrix = ReadAlleleMatrix(
            matrix=np.array([
                [10, 0, 0], [9, 0, 0],
                [0, 12, 0], [0, 11, 0],
            ], dtype=np.float64),
            read_names=["r1", "r2", "r3", "r4"],
            allele_names=["A*02:01", "A*03:01", "A*24:02"],
        )
        result = vb.estimate(matrix, locus="A")
        total = result.posterior_prob + sum(
            p for _, _, p in result.alternative_pairs
        )
        # All posteriors should approximately sum to 1
        assert total == pytest.approx(1.0, abs=0.05)


# ============ Feature 3: Novel allele mismatch annotation ============

class TestNovelAlleleFlagging:
    def test_mismatch_annotation_dataclass(self):
        from hla_unified.assembly.targeted_assembler import MismatchAnnotation
        m = MismatchAnnotation(position=42, ref_base="A", alt_base="G", variant_type="SNP")
        assert m.position == 42
        assert m.variant_type == "SNP"

    def test_novel_report_dataclass(self):
        from hla_unified.assembly.targeted_assembler import (
            NovelAlleleReport, MismatchAnnotation,
        )
        mm = MismatchAnnotation(100, "A", "G", "SNP")
        report = NovelAlleleReport(
            closest_allele="A*02:01", identity=0.995,
            mismatches=[mm], n_snps=1, n_insertions=0, n_deletions=0,
            aligned_length=3000,
            summary="Closest: A*02:01 (99.5%). 1 SNP.",
        )
        assert report.n_snps == 1
        assert "A*02:01" in report.summary


# ============ Feature 4: Assay presets ============

class TestAssayPresets:
    def test_all_presets_exist(self):
        from hla_unified.prefilter.fast_mapper import MINIMAP2_PRESETS, ASSAY_CONFIGS
        expected = ["short", "exome", "targeted_capture", "pacbio", "hifi", "ont", "rna"]
        for preset in expected:
            assert preset in MINIMAP2_PRESETS, f"Missing minimap2 preset: {preset}"
            assert preset in ASSAY_CONFIGS, f"Missing assay config: {preset}"

    def test_assay_configs_have_required_keys(self):
        from hla_unified.prefilter.fast_mapper import ASSAY_CONFIGS
        for name, config in ASSAY_CONFIGS.items():
            assert "description" in config, f"{name}: missing description"
            assert "min_coverage" in config, f"{name}: missing min_coverage"
            assert "expected_coverage" in config, f"{name}: missing expected_coverage"

    def test_refinement_aligner_configs(self):
        from hla_unified.refinement.iterative_refiner import ALIGNER_CONFIG
        for preset in ["short", "exome", "targeted_capture", "pacbio", "hifi", "ont", "rna"]:
            assert preset in ALIGNER_CONFIG, f"Missing refiner config: {preset}"


# ============ Feature 5: QC report ============

class TestQCReport:
    def test_qc_report_generation(self):
        from hla_unified.pipeline.runner import PipelineResult, LocusCall
        from hla_unified.qc.report import generate_qc_report

        calls = {
            "A": LocusCall(
                locus="A", allele1="A*02:01", allele2="A*03:01",
                g_group1="A*02:01G", g_group2="A*03:01G",
                confidence="HIGH", posterior=0.999,
                reads_explained=100, total_reads=105,
                kmer_covered=0.98, kmer_concordant=True,
                is_novel=False,
            ),
        }
        result = PipelineResult(
            calls=calls, runtime_seconds=42.5,
            phases_completed=["prefilter", "refinement", "ilp"],
            imgt_release="3.51.0", imgt_commit="abc123",
        )
        provenance = {"release": "3.51.0", "git_commit": "abc123"}

        qc = generate_qc_report(result, provenance, "short", "test.bam")
        assert qc.total_loci == 1
        assert qc.loci_high_confidence == 1
        assert qc.locus_qc["A"].pass_qc is True
        assert qc.imgt_release == "3.51.0"

    def test_qc_json_roundtrip(self, tmp_path):
        import json
        from hla_unified.pipeline.runner import PipelineResult, LocusCall
        from hla_unified.qc.report import generate_qc_report, write_qc_json

        calls = {
            "A": LocusCall(
                locus="A", allele1="A*02:01", allele2="A*03:01",
                g_group1="", g_group2="", confidence="HIGH", posterior=0.99,
                reads_explained=50, total_reads=50,
                kmer_covered=1.0, kmer_concordant=True, is_novel=False,
            ),
        }
        result = PipelineResult(
            calls=calls, runtime_seconds=1.0, phases_completed=[],
            imgt_release="3.51.0", imgt_commit="abc",
        )
        qc = generate_qc_report(result, {"release": "3.51.0", "git_commit": "abc"})

        json_path = tmp_path / "qc.json"
        write_qc_json(qc, json_path)

        data = json.loads(json_path.read_text())
        assert data["metadata"]["imgt_release"] == "3.51.0"
        assert "A" in data["loci"]

    def test_qc_html_generation(self, tmp_path):
        from hla_unified.pipeline.runner import PipelineResult, LocusCall
        from hla_unified.qc.report import generate_qc_report, write_qc_html

        calls = {
            "A": LocusCall(
                locus="A", allele1="A*02:01", allele2="A*03:01",
                g_group1="", g_group2="", confidence="HIGH", posterior=0.99,
                reads_explained=50, total_reads=50,
                kmer_covered=1.0, kmer_concordant=True, is_novel=False,
            ),
        }
        result = PipelineResult(
            calls=calls, runtime_seconds=1.0, phases_completed=["prefilter"],
            imgt_release="3.51.0", imgt_commit="abc123def456",
        )
        qc = generate_qc_report(result, {"release": "3.51.0", "git_commit": "abc123def456"})

        html_path = tmp_path / "qc.html"
        write_qc_html(qc, html_path)

        html = html_path.read_text()
        assert "HLA-Unified QC Dashboard" in html
        assert "3.51.0" in html
        assert "HLA-A" in html


# ============ Feature 6: Database provenance ============

class TestDatabaseProvenance:
    def test_provenance_file_constant(self):
        from hla_unified.reference.imgt_db import PROVENANCE_FILE
        assert PROVENANCE_FILE == ".hla_unified_provenance.json"

    def test_provenance_detection_missing_db(self, tmp_path):
        from hla_unified.reference.imgt_db import IMGTDatabase
        db = IMGTDatabase(tmp_path / "nonexistent")
        prov = db.provenance
        assert "release" in prov
        assert prov["release"] == "unknown" or isinstance(prov["release"], str)

    def test_pipeline_result_has_provenance(self):
        from hla_unified.pipeline.runner import PipelineResult
        result = PipelineResult(
            calls={}, runtime_seconds=0,
            phases_completed=[],
            imgt_release="3.51.0",
            imgt_commit="abc123",
        )
        assert result.imgt_release == "3.51.0"
        assert result.imgt_commit == "abc123"
