"""Unit tests for hlala.utils modules."""

import math
import tempfile
from pathlib import Path

import pytest

from hlala.utils.seq import (
    reverse_complement, phred_to_p_correct, p_correct_to_phred,
    partition_into_kmers, kmer_canonical, remove_gaps,
    proportion_n, sequence_all_ns,
)
from hlala.utils.io import read_fasta, write_fasta, get_all_lines, ensure_directory
from hlala.utils.intervals import build_interval_tree, query_overlapping, intervals_overlap


class TestSeq:
    def test_reverse_complement(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AACG") == "CGTT"
        assert reverse_complement("") == ""
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("N") == "N"

    def test_phred_to_p_correct(self):
        # Q=0 -> P=0.25 (special case)
        assert phred_to_p_correct("!") == 0.25
        # Q=10 -> P=0.9
        assert abs(phred_to_p_correct("+") - 0.9) < 1e-10
        # Q=20 -> P=0.99
        assert abs(phred_to_p_correct("5") - 0.99) < 1e-10
        # Q=40 -> P=0.9999
        assert phred_to_p_correct("I") >= 0.9999

    def test_p_correct_to_phred(self):
        assert p_correct_to_phred(0.9) == 10
        assert p_correct_to_phred(0.99) == 20
        assert p_correct_to_phred(1.0) == 60
        assert p_correct_to_phred(0.0) == 0

    def test_partition_into_kmers(self):
        assert partition_into_kmers("ACGTAC", 3) == ["ACG", "CGT", "GTA", "TAC"]
        assert partition_into_kmers("AC", 3) == []
        assert partition_into_kmers("ACG", 3) == ["ACG"]

    def test_kmer_canonical(self):
        # ACG vs CGT (rc) -> ACG is smaller
        assert kmer_canonical("ACG") == "ACG"
        # TTT vs AAA (rc) -> AAA is smaller
        assert kmer_canonical("TTT") == "AAA"

    def test_remove_gaps(self):
        assert remove_gaps("AC_GT-A") == "ACGTA"
        assert remove_gaps("ACGT") == "ACGT"
        assert remove_gaps("___") == ""

    def test_proportion_n(self):
        assert proportion_n("ACGT") == 0.0
        assert proportion_n("NNNN") == 1.0
        assert abs(proportion_n("ACNN") - 0.5) < 1e-10
        assert proportion_n("") == 0.0

    def test_sequence_all_ns(self):
        assert sequence_all_ns("NNN")
        assert sequence_all_ns("nnn")
        assert not sequence_all_ns("ANN")


class TestIO:
    def test_fasta_roundtrip(self, tmp_path):
        seqs = {"seq1": "ACGTACGT", "seq2": "TTTTAAAA"}
        fasta_path = tmp_path / "test.fa"
        write_fasta(fasta_path, seqs)
        loaded = read_fasta(fasta_path)
        assert loaded == seqs

    def test_fasta_full_identifier(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        fasta_path.write_text(">seq1 extra info\nACGT\n>seq2 more\nTTTT\n")
        short = read_fasta(fasta_path, full_identifier=False)
        assert "seq1" in short
        full = read_fasta(fasta_path, full_identifier=True)
        assert "seq1 extra info" in full

    def test_get_all_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        lines = get_all_lines(f)
        assert lines == ["line1", "line2", "line3"]

    def test_ensure_directory(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_directory(new_dir)
        assert result.exists()


class TestIntervals:
    def test_intervals_overlap(self):
        assert intervals_overlap(1, 5, 3, 7)
        assert intervals_overlap(1, 5, 5, 10)
        assert not intervals_overlap(1, 5, 6, 10)
        assert intervals_overlap(3, 3, 1, 5)

    def test_build_and_query(self):
        tree = build_interval_tree([(1, 10, "A"), (5, 15, "B"), (20, 30, "C")])
        assert query_overlapping(tree, 3, 3) == {"A"}
        assert query_overlapping(tree, 7, 7) == {"A", "B"}
        assert query_overlapping(tree, 25, 25) == {"C"}
        assert query_overlapping(tree, 16, 19) == set()
