"""Unit tests for hlala.graph modules."""

import tempfile
from pathlib import Path

import pytest

from hlala.graph.node import Node, NodeHaplotype
from hlala.graph.edge import Edge, EMISSION_ENCODE, EMISSION_DECODE
from hlala.graph.graph import PRGGraph
from hlala.graph.locus_code_allocation import LocusCodeAllocation
from hlala.graph.graph_and_edge_index import GraphAndEdgeIndex, KMerInGraphSpec
from hlala.graph.serialization import save_graph, load_graph


class TestEdge:
    def test_emission_encoding(self):
        assert EMISSION_ENCODE["A"] == 0
        assert EMISSION_ENCODE["C"] == 1
        assert EMISSION_ENCODE["G"] == 2
        assert EMISSION_ENCODE["T"] == 3
        assert EMISSION_ENCODE["_"] == 4

    def test_emission_decoding(self):
        for char in "ACGT_N":
            code = EMISSION_ENCODE[char]
            assert EMISSION_DECODE[code] == char

    def test_edge_get_emission(self):
        e = Edge(id=0, from_node=0, to_node=1, emission=EMISSION_ENCODE["A"])
        assert e.get_emission() == "A"
        e2 = Edge(id=1, from_node=0, to_node=1, emission=4)
        assert e2.get_emission() == "_"


class TestPRGGraph:
    def _make_simple_graph(self):
        """Build: level 0 --(A/G)--> level 1 --(C)--> level 2"""
        g = PRGGraph()
        n0 = g.register_node(level=0)
        n1 = g.register_node(level=1)
        n2 = g.register_node(level=1)
        n3 = g.register_node(level=2, terminal=True)
        g.register_edge(n0.id, n1.id, emission="A", locus_id="HLA-A")
        g.register_edge(n0.id, n2.id, emission="G", locus_id="HLA-A")
        g.register_edge(n1.id, n3.id, emission="C", locus_id="HLA-A")
        g.register_edge(n2.id, n3.id, emission="C", locus_id="HLA-A")
        return g

    def test_basic_construction(self):
        g = self._make_simple_graph()
        assert g.num_levels == 3
        assert len(g.nodes) == 4
        assert len(g.edges) == 4

    def test_nodes_per_level(self):
        g = self._make_simple_graph()
        assert len(g.nodes_per_level[0]) == 1
        assert len(g.nodes_per_level[1]) == 2
        assert len(g.nodes_per_level[2]) == 1

    def test_check_sequence_presence(self):
        g = self._make_simple_graph()
        assert g.check_sequence_presence("AC")
        assert g.check_sequence_presence("GC")
        assert not g.check_sequence_presence("TC")
        assert not g.check_sequence_presence("AG")

    def test_consistency_check(self):
        g = self._make_simple_graph()
        g.check_consistency()  # Should not raise

    def test_get_assigned_loci(self):
        g = self._make_simple_graph()
        loci = g.get_assigned_loci()
        assert "HLA-A" in loci

    def test_get_level_info(self):
        g = self._make_simple_graph()
        info = g.get_level_info()
        assert len(info) == 3
        assert info[0]["nodes"] == 1
        assert info[0]["edges"] == 2  # A and G edges
        assert info[0]["symbols"] == 2  # A and G

    def test_unregister_edge(self):
        g = self._make_simple_graph()
        edge_ids = list(g.edges.keys())
        g.unregister_edge(edge_ids[0])
        assert len(g.edges) == 3

    def test_trim_graph(self):
        g = PRGGraph()
        n0 = g.register_node(level=0)
        n1 = g.register_node(level=1)
        g.register_edge(n0.id, n1.id, emission="A", count=5.0)
        g.register_edge(n0.id, n1.id, emission="G", count=0.0)
        removed = g.trim_graph()
        assert removed == 1
        assert len(g.edges) == 1

    def test_serialization_roundtrip(self, tmp_path):
        g = self._make_simple_graph()
        filepath = tmp_path / "test.msgpack"
        save_graph(g, filepath)
        g2 = load_graph(filepath)
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.edges) == len(g.edges)
        assert g2.num_levels == g.num_levels
        assert g2.check_sequence_presence("AC")
        assert g2.check_sequence_presence("GC")


class TestLocusCodeAllocation:
    def test_encode_decode(self):
        alloc = LocusCodeAllocation()
        c1 = alloc.encode("HLA-A", "A*01:01")
        c2 = alloc.encode("HLA-A", "A*02:01")
        assert c1 != c2
        assert alloc.decode("HLA-A", c1) == "A*01:01"
        assert alloc.decode("HLA-A", c2) == "A*02:01"

    def test_knows(self):
        alloc = LocusCodeAllocation()
        alloc.encode("HLA-A", "A*01:01")
        assert alloc.knows_code("HLA-A", 0)
        assert alloc.knows_allele("HLA-A", "A*01:01")
        assert not alloc.knows_allele("HLA-A", "A*99:99")
        assert not alloc.knows_code("HLA-B", 0)

    def test_get_loci(self):
        alloc = LocusCodeAllocation()
        alloc.encode("HLA-A", "A*01:01")
        alloc.encode("HLA-B", "B*07:02")
        assert set(alloc.get_loci()) == {"HLA-A", "HLA-B"}

    def test_serialization(self):
        alloc = LocusCodeAllocation()
        alloc.encode("HLA-A", "A*01:01")
        alloc.encode("HLA-A", "A*02:01")
        lines = alloc.serialize()
        alloc2 = LocusCodeAllocation.deserialize(lines)
        assert alloc2.decode("HLA-A", 0) == "A*01:01"

    def test_remove_locus(self):
        alloc = LocusCodeAllocation()
        alloc.encode("HLA-A", "A*01:01")
        alloc.remove_locus("HLA-A")
        assert "HLA-A" not in alloc.get_loci()


class TestGraphAndEdgeIndex:
    def test_index_and_query(self):
        g = PRGGraph()
        nodes = [g.register_node(level=i) for i in range(5)]
        for i in range(4):
            g.register_edge(nodes[i].id, nodes[i + 1].id, emission="ACGT"[i % 4])

        idx = GraphAndEdgeIndex(g, k=2)
        idx.index()
        assert len(idx.kmers) > 0

        # "AC" should be found
        hits = idx.query_index("AC")
        assert len(hits) > 0

    def test_find_chains(self):
        g = PRGGraph()
        nodes = [g.register_node(level=i) for i in range(5)]
        for i in range(4):
            g.register_edge(nodes[i].id, nodes[i + 1].id, emission="ACGT"[i % 4])

        idx = GraphAndEdgeIndex(g, k=2)
        idx.index()

        chains = idx.find_chains("ACGT")
        assert len(chains) > 0
