"""Extension aligner: extends seed chains through the graph using NW DP.

Translates mapper::aligner::extensionAligner from C++.
This is the core alignment algorithm that extends initial seed matches
in both directions through the population reference graph.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .aligner_base import AlignerBase
from .nw_unique import VirtualNWTable, NWEdge, LocalExtensionPath
from ..mapper.reads import VerboseSeedChain, OneRead
from ..graph.graph import PRGGraph

logger = logging.getLogger(__name__)


class ExtensionAligner(AlignerBase):
    """Seed-and-extend aligner on the population reference graph.

    Given initial seed chains (from BWA remapping), extends the alignment
    in both directions using banded Needleman-Wunsch on the graph DAG.
    """

    def __init__(self, graph: PRGGraph) -> None:
        super().__init__(graph)
        self.paranoid = False

    def extend_seed_chain(self, sequence: str,
                          seed_chain: VerboseSeedChain) -> VerboseSeedChain:
        """Extend a seed chain to cover more of the read sequence.

        Extends leftward from seed start and rightward from seed end
        using diagonal-banded NW alignment on the graph.
        """
        result = VerboseSeedChain(
            read_id=seed_chain.read_id,
            reverse=seed_chain.reverse,
            from_first_read=seed_chain.from_first_read,
        )

        first_level = seed_chain.alignment_first_level()
        last_level = seed_chain.alignment_last_level()

        if first_level == -1 or last_level == -1:
            return seed_chain

        # Extend leftward
        if seed_chain.sequence_begin > 0 and first_level > 0:
            left_ext = self._extend_direction(
                sequence=sequence,
                start_seq=seed_chain.sequence_begin - 1,
                start_level=first_level - 1,
                direction_positive=False,
            )
            if left_ext:
                result.extend_with_other(left_ext, left=True)

        # Copy the seed chain itself
        result.graph_aligned_edges.extend(seed_chain.graph_aligned_edges)
        result.graph_aligned_levels.extend(seed_chain.graph_aligned_levels)
        result.graph_aligned += seed_chain.graph_aligned
        result.sequence_aligned += seed_chain.sequence_aligned
        result.sequence_begin = seed_chain.sequence_begin
        result.sequence_end = seed_chain.sequence_end

        # Extend rightward
        if seed_chain.sequence_end < len(sequence) - 1 and last_level < self.graph.num_levels - 2:
            right_ext = self._extend_direction(
                sequence=sequence,
                start_seq=seed_chain.sequence_end + 1,
                start_level=last_level + 1,
                direction_positive=True,
            )
            if right_ext:
                result.extend_with_other(right_ext, left=False)

        return result

    def _extend_direction(self, sequence: str, start_seq: int, start_level: int,
                          direction_positive: bool,
                          diagonal_stop_threshold: int = 20) -> Optional[VerboseSeedChain]:
        """Extend alignment in one direction using banded NW.

        Args:
            sequence: full read sequence
            start_seq: starting position in sequence
            start_level: starting graph level
            direction_positive: True for rightward, False for leftward
            diagonal_stop_threshold: stop when score drops this much from best
        """
        if direction_positive:
            max_seq = len(sequence) - 1
            max_level = self.graph.num_levels - 1
        else:
            max_seq = 0
            max_level = 0

        num_seq_positions = abs(max_seq - start_seq) + 1
        num_levels = abs(max_level - start_level) + 1

        if num_seq_positions == 0 or num_levels == 0:
            return None

        # Determine which start nodes to use
        if direction_positive:
            start_nodes = self.nodes_per_level_ordered[start_level]
        else:
            start_nodes = self.nodes_per_level_ordered[start_level]

        if not start_nodes:
            return None

        num_z = len(start_nodes)

        # Allocate DP tables (3 matrices for affine gaps)
        # Dimensions: num_levels x num_seq_positions x num_z
        band_width = min(num_seq_positions, num_levels)
        INF = float('-inf')

        dp_match = np.full((num_levels, num_seq_positions, num_z), INF)
        dp_graph_gap = np.full((num_levels, num_seq_positions, num_z), INF)
        dp_seq_gap = np.full((num_levels, num_seq_positions, num_z), INF)

        # Backtrace storage
        bt = {}  # (level_i, seq_i, z_i) -> (prev_level_i, prev_seq_i, prev_z_i, edge_id, matrix)

        # Initialize
        for z_i in range(num_z):
            dp_match[0, 0, z_i] = 0.0

        best_score = 0.0
        best_pos = (0, 0, 0)

        step = 1 if direction_positive else -1

        for li in range(num_levels):
            actual_level = start_level + li * step

            if actual_level < 0 or actual_level >= self.graph.num_levels:
                break

            current_nodes = self.nodes_per_level_ordered[actual_level]
            num_z_current = len(current_nodes)

            for si in range(min(num_seq_positions, li + diagonal_stop_threshold + 1)):
                actual_seq = start_seq + si * step
                if actual_seq < 0 or actual_seq >= len(sequence):
                    break

                seq_char = sequence[actual_seq]

                for z_i in range(num_z_current):
                    node_id = current_nodes[z_i]

                    # Match/mismatch: come from (li-1, si-1, z_prev) via an edge
                    if li > 0 and si > 0:
                        prev_level = actual_level - step
                        if 0 <= prev_level < self.graph.num_levels:
                            for edge_id in self.graph.nodes[node_id].incoming_edges:
                                edge = self.graph.edges[edge_id]
                                from_node = edge.from_node
                                from_level = self.graph.nodes[from_node].level
                                if from_level == prev_level:
                                    prev_nodes = self.nodes_per_level_ordered[from_level]
                                    z_prev = self.nodes_per_level_ordered_rev[from_level].get(from_node, -1)
                                    if z_prev >= 0 and z_prev < dp_match.shape[2]:
                                        emission = edge.get_emission()
                                        score_delta = self.S_match if emission == seq_char else self.S_mismatch

                                        for prev_mat in [dp_match, dp_graph_gap, dp_seq_gap]:
                                            if li - 1 < prev_mat.shape[0] and si - 1 < prev_mat.shape[1]:
                                                prev_score = prev_mat[li - 1, si - 1, z_prev]
                                                if prev_score > INF:
                                                    new_score = prev_score + score_delta
                                                    if new_score > dp_match[li, si, z_i]:
                                                        dp_match[li, si, z_i] = new_score
                                                        bt[(li, si, z_i)] = (li - 1, si - 1, z_prev, edge_id, 0)

                    # Gap in sequence: consume graph level, no sequence
                    if li > 0:
                        prev_level = actual_level - step
                        if 0 <= prev_level < self.graph.num_levels:
                            for edge_id in self.graph.nodes[node_id].incoming_edges:
                                edge = self.graph.edges[edge_id]
                                from_node = edge.from_node
                                from_level = self.graph.nodes[from_node].level
                                if from_level == prev_level:
                                    z_prev = self.nodes_per_level_ordered_rev[from_level].get(from_node, -1)
                                    if z_prev >= 0 and z_prev < dp_match.shape[2]:
                                        # Open gap
                                        if li - 1 < dp_match.shape[0] and si < dp_match.shape[1]:
                                            prev_score = dp_match[li - 1, si, z_prev]
                                            if prev_score > INF:
                                                new_score = prev_score + self.S_open_gap
                                                if new_score > dp_seq_gap[li, si, z_i]:
                                                    dp_seq_gap[li, si, z_i] = new_score
                                                    bt[(li, si, z_i)] = (li - 1, si, z_prev, edge_id, 2)
                                        # Extend gap
                                        if li - 1 < dp_seq_gap.shape[0] and si < dp_seq_gap.shape[1]:
                                            prev_score = dp_seq_gap[li - 1, si, z_prev]
                                            if prev_score > INF:
                                                new_score = prev_score + self.S_extend_gap
                                                if new_score > dp_seq_gap[li, si, z_i]:
                                                    dp_seq_gap[li, si, z_i] = new_score
                                                    bt[(li, si, z_i)] = (li - 1, si, z_prev, edge_id, 2)

                    # Gap in graph: consume sequence position, stay at same level
                    if si > 0:
                        # Open gap
                        if si - 1 < dp_match.shape[1]:
                            prev_score = dp_match[li, si - 1, z_i]
                            if prev_score > INF:
                                new_score = prev_score + self.S_open_gap
                                if new_score > dp_graph_gap[li, si, z_i]:
                                    dp_graph_gap[li, si, z_i] = new_score
                                    bt[(li, si, z_i)] = (li, si - 1, z_i, -1, 1)
                        # Extend gap
                        if si - 1 < dp_graph_gap.shape[1]:
                            prev_score = dp_graph_gap[li, si - 1, z_i]
                            if prev_score > INF:
                                new_score = prev_score + self.S_extend_gap
                                if new_score > dp_graph_gap[li, si, z_i]:
                                    dp_graph_gap[li, si, z_i] = new_score
                                    bt[(li, si, z_i)] = (li, si - 1, z_i, -1, 1)

                    # Track best score
                    for mat in [dp_match, dp_graph_gap, dp_seq_gap]:
                        if mat[li, si, z_i] > best_score:
                            best_score = mat[li, si, z_i]
                            best_pos = (li, si, z_i)

        # Backtrace
        if best_score <= 0:
            return None

        path_edges: list[int] = []
        path_levels: list[int] = []
        graph_chars: list[str] = []
        seq_chars: list[str] = []

        pos = best_pos
        while pos in bt:
            li, si, z_i = pos
            prev_li, prev_si, prev_z_i, edge_id, matrix = bt[pos]

            actual_level = start_level + li * step
            actual_seq = start_seq + si * step

            if matrix == 0:  # match/mismatch
                edge = self.graph.edges[edge_id]
                path_edges.append(edge_id)
                path_levels.append(actual_level)
                graph_chars.append(edge.get_emission())
                seq_chars.append(sequence[actual_seq])
            elif matrix == 1:  # graph gap (insertion in read)
                path_edges.append(-1)
                path_levels.append(-1)
                graph_chars.append("_")
                seq_chars.append(sequence[actual_seq])
            elif matrix == 2:  # sequence gap (deletion in read)
                edge = self.graph.edges[edge_id]
                path_edges.append(edge_id)
                path_levels.append(actual_level)
                graph_chars.append(edge.get_emission())
                seq_chars.append("_")

            pos = (prev_li, prev_si, prev_z_i)

        # Reverse if we built the path backward
        if not direction_positive:
            pass  # Already in correct order for leftward extension
        else:
            path_edges.reverse()
            path_levels.reverse()
            graph_chars.reverse()
            seq_chars.reverse()

        chain = VerboseSeedChain()
        chain.graph_aligned_edges = path_edges
        chain.graph_aligned_levels = path_levels
        chain.graph_aligned = "".join(graph_chars)
        chain.sequence_aligned = "".join(seq_chars)
        chain.sequence_begin = min(start_seq, start_seq + (best_pos[1]) * step)
        chain.sequence_end = max(start_seq, start_seq + (best_pos[1]) * step)

        return chain

    def score_alignment(self, alignment: VerboseSeedChain, read: OneRead,
                        long_read_mode: str = "") -> float:
        """Score a completed alignment using base qualities.

        Computes log-likelihood of the alignment given the read qualities.
        """
        from ..utils.seq import phred_to_p_correct
        import math

        score = 0.0
        seq_idx = alignment.sequence_begin

        for i in range(len(alignment.graph_aligned)):
            g_char = alignment.graph_aligned[i]
            s_char = alignment.sequence_aligned[i]

            if s_char == "_":
                # Deletion in read
                score += self.S_open_gap
            elif g_char == "_":
                # Insertion in read
                score += self.S_open_gap
            else:
                # Match or mismatch
                if seq_idx < len(read.quality):
                    p_correct = phred_to_p_correct(read.quality[seq_idx])
                else:
                    p_correct = 0.999

                if g_char == s_char:
                    score += math.log(p_correct) if p_correct > 0 else -100
                else:
                    p_error = 1.0 - p_correct
                    score += math.log(p_error / 3.0) if p_error > 0 else -100

                seq_idx += 1

        return score
