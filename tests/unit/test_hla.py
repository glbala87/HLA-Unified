"""Unit tests for hlala.hla modules."""

import math
from pathlib import Path

import pytest

from hlala.hla.exon_position import OneExonPosition
from hlala.hla.loci_config import LOCI_FOR_TYPING, LOCI_TO_EXONS
from hlala.hla.g_groups import GGroupTranslator
from hlala.hla.kmer_analysis import (
    compute_kmer_coverage, proportion_kmers_covered,
    kmer_chi_square, compute_coverage_stats,
)
from hlala.hla.typer import log_avg, log_sum_log_ps


class TestExonPosition:
    def test_construction(self):
        pos = OneExonPosition(
            position_in_exon=42,
            graph_level=1000,
            genotype="A",
            this_read_id="read1",
        )
        assert pos.position_in_exon == 42
        assert pos.graph_level == 1000
        assert pos.genotype == "A"


class TestLociConfig:
    def test_loci_present(self):
        assert "A" in LOCI_FOR_TYPING
        assert "B" in LOCI_FOR_TYPING
        assert "C" in LOCI_FOR_TYPING
        assert "DRB1" in LOCI_FOR_TYPING

    def test_exons(self):
        assert "exon2" in LOCI_TO_EXONS["A"]
        assert "exon3" in LOCI_TO_EXONS["A"]
        assert "exon2" in LOCI_TO_EXONS["DRB1"]


class TestGGroupTranslator:
    @pytest.fixture
    def translator(self):
        g_file = Path(__file__).parent.parent.parent / "hla_nom_g.txt"
        if not g_file.exists():
            pytest.skip("hla_nom_g.txt not available")
        gt = GGroupTranslator()
        gt.load(g_file)
        return gt

    def test_load(self, translator):
        assert len(translator.allele_to_g) > 0
        assert "A" in translator.g_loci

    def test_translate_known(self, translator):
        g, perfect = translator.translate("A*01:01:01:01")
        assert "G" in g
        assert perfect

    def test_translate_unknown(self, translator):
        g, perfect = translator.translate("Z*99:99:99:99")
        assert g == "Z*99:99:99:99"
        assert not perfect

    def test_can_translate(self, translator):
        assert translator.can_translate("A")
        assert translator.can_translate("B")
        assert not translator.can_translate("NONEXISTENT")


class TestKmerAnalysis:
    def test_compute_coverage_stats(self):
        stats = compute_coverage_stats([10, 20, 30, 40, 50])
        assert stats["mean"] == 30.0
        assert stats["minimum"] == 10.0
        assert stats["first_decile"] == 10.0

    def test_compute_coverage_stats_empty(self):
        stats = compute_coverage_stats([])
        assert stats["mean"] == 0.0

    def test_proportion_kmers_covered(self):
        coverage = {"AA": 5, "CC": 0, "GG": 3}
        assert abs(proportion_kmers_covered(coverage) - 2 / 3) < 1e-10

    def test_proportion_kmers_covered_all(self):
        coverage = {"AA": 1, "CC": 2}
        assert proportion_kmers_covered(coverage) == 1.0

    def test_proportion_kmers_covered_empty(self):
        assert proportion_kmers_covered({}) == 0.0

    def test_chi_square(self):
        observed = [10.0, 20.0, 30.0]
        expected = [20.0, 20.0, 20.0]
        chi = kmer_chi_square(observed, expected)
        assert chi == pytest.approx(10.0)

    def test_chi_square_perfect(self):
        observed = [10.0, 10.0, 10.0]
        expected = [10.0, 10.0, 10.0]
        assert kmer_chi_square(observed, expected) == 0.0

    def test_compute_kmer_coverage(self):
        allele = "ACGTACGT"
        reads = ["ACGTAC", "CGTACG"]
        coverage = compute_kmer_coverage(allele, reads, k=3)
        assert len(coverage) > 0
        assert all(v >= 0 for v in coverage.values())


class TestLogFunctions:
    def test_log_avg_equal(self):
        a = math.log(0.5)
        result = log_avg(a, a)
        assert abs(math.exp(result) - 0.5) < 1e-10

    def test_log_avg_different(self):
        a = math.log(0.3)
        b = math.log(0.7)
        result = log_avg(a, b)
        expected = math.log(0.5 * (0.3 + 0.7))
        assert abs(result - expected) < 1e-10

    def test_log_sum_log_ps(self):
        values = [math.log(0.2), math.log(0.3), math.log(0.5)]
        result = log_sum_log_ps(values)
        assert abs(math.exp(result) - 1.0) < 1e-10

    def test_log_sum_log_ps_empty(self):
        assert log_sum_log_ps([]) == float('-inf')

    def test_log_sum_log_ps_single(self):
        val = math.log(0.42)
        assert abs(log_sum_log_ps([val]) - val) < 1e-10
