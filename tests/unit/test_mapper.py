"""Unit tests for hlala.mapper modules."""

import pytest

from hlala.mapper.reads import OneRead, OneReadPair, VerboseSeedChain, VerboseSeedChainPair
from hlala.mapper.seed_chain import ProtoSeed, SeedChain
from hlala.aligner.aligner_base import AlignerBase
from hlala.aligner.statistics import AlignmentStats


class TestOneRead:
    def test_construction(self):
        r = OneRead(name="read1", sequence="ACGT", quality="IIII")
        assert r.name == "read1"
        assert len(r.sequence) == 4

    def test_invert_palindrome(self):
        r = OneRead(name="r", sequence="ACGT", quality="ABCD")
        r.invert()
        assert r.sequence == "ACGT"  # ACGT is its own reverse complement
        assert r.quality == "DCBA"

    def test_invert_non_palindrome(self):
        r = OneRead(name="r", sequence="AACC", quality="ABCD")
        r.invert()
        assert r.sequence == "GGTT"
        assert r.quality == "DCBA"

    def test_mismatched_length_raises(self):
        with pytest.raises(AssertionError):
            OneRead(name="r", sequence="ACGT", quality="II")


class TestOneReadPair:
    def test_invert(self):
        r1 = OneRead(name="r", sequence="AAAA", quality="IIII")
        r2 = OneRead(name="r", sequence="CCCC", quality="JJJJ")
        pair = OneReadPair(read1=r1, read2=r2)
        pair.invert()
        assert pair.read1.sequence == "GGGG"  # rc of CCCC
        assert pair.read2.sequence == "TTTT"  # rc of AAAA


class TestVerboseSeedChain:
    def test_quality(self):
        chain = VerboseSeedChain(
            graph_aligned="ACGT",
            sequence_aligned="ACGT",
        )
        length, matches = chain.quality()
        assert length == 4 and matches == 4

    def test_quality_with_mismatches(self):
        chain = VerboseSeedChain(
            graph_aligned="ACGT",
            sequence_aligned="AXGT",
        )
        length, matches = chain.quality()
        assert length == 4 and matches == 3

    def test_first_last_level(self):
        chain = VerboseSeedChain(
            graph_aligned_levels=[-1, 5, 6, 7, -1],
            graph_aligned="_ACGT_",
            sequence_aligned="_ACGT_",
        )
        assert chain.alignment_first_level() == 5
        assert chain.alignment_last_level() == 7

    def test_first_last_levels_multiple(self):
        chain = VerboseSeedChain(
            graph_aligned_levels=[10, 11, 12, 13, 14],
            graph_aligned="ACGTX",
            sequence_aligned="ACGTX",
        )
        assert chain.alignment_first_levels(3) == [10, 11, 12]
        assert chain.alignment_last_levels(2) == [14, 13]

    def test_empty_chain(self):
        chain = VerboseSeedChain()
        assert chain.alignment_first_level() == -1
        assert chain.alignment_last_level() == -1

    def test_extend_with_other_right(self):
        c1 = VerboseSeedChain(
            sequence_begin=0, sequence_end=3,
            graph_aligned_edges=[1, 2, 3, 4],
            graph_aligned_levels=[0, 1, 2, 3],
            graph_aligned="ACGT",
            sequence_aligned="ACGT",
        )
        c2 = VerboseSeedChain(
            sequence_begin=4, sequence_end=5,
            graph_aligned_edges=[5, 6],
            graph_aligned_levels=[4, 5],
            graph_aligned="AA",
            sequence_aligned="AA",
        )
        c1.extend_with_other(c2, left=False)
        assert len(c1.graph_aligned_edges) == 6
        assert c1.sequence_end == 5
        assert c1.graph_aligned == "ACGTAA"

    def test_get_segments(self):
        chain = VerboseSeedChain(
            graph_aligned_levels=[0, 1, 2],
            graph_aligned="ACG",
            sequence_aligned="ACG",
        )
        level_names = ["HLA-A_exon2", "HLA-A_exon2", "HLA-A_exon3"]
        segments = chain.get_segments(level_names)
        assert segments == {"HLA-A_exon2", "HLA-A_exon3"}


class TestProtoSeed:
    def test_take_alignment(self):
        ps = ProtoSeed()
        ps.take_alignment("chr6", 100, None, 1)
        ps.take_alignment("chr6", 200, None, 2)
        assert ps.is_complete()

    def test_incomplete(self):
        ps = ProtoSeed()
        ps.take_alignment("chr6", 100, None, 1)
        assert not ps.is_complete()


class TestAlignmentStats:
    def test_update(self):
        stats = AlignmentStats()
        stats.update(100, 95)
        stats.update(100, 90)
        assert stats.total_reads == 2
        assert stats.mapped_reads == 2
        assert stats.mean_alignment_length == 100.0
        assert abs(stats.mean_match_fraction - 0.925) < 1e-10

    def test_summary(self):
        stats = AlignmentStats()
        stats.update(100, 95)
        s = stats.summary()
        assert "Total reads: 1" in s
