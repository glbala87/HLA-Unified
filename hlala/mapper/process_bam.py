"""BAM processing: extract seeds and prepare reads for graph alignment.

Translates processBAM from C++. Parses remapped BAM files, converts
CIGAR operations to graph-level coordinates, and produces seed chains.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .reads import OneRead, OneReadPair, VerboseSeedChain, VerboseSeedChainPair
from .seed_chain import ProtoSeed

logger = logging.getLogger(__name__)


class BAMProcessor:
    """Process remapped BAM files to extract alignment seeds.

    Translates mapper::processBAM from C++. Main responsibilities:
    1. Parse BAM remapped to extended reference genome
    2. Convert CIGAR coordinates to graph levels
    3. Group alignments by read name
    4. Produce seed chains for extension alignment
    """

    def __init__(self, graph_dir: str | Path) -> None:
        self.graph_dir = Path(graph_dir)
        self.level_translation: dict[int, list[int]] = {}
        self.extended_ref_sequences: dict[str, str] = {}
        self.prg_ref_sequences: dict[str, str] = {}
        self._load_level_translation()

    def _load_level_translation(self) -> None:
        """Load the coordinate mapping from reference genome positions to graph levels."""
        translation_file = self.graph_dir / "translation.txt"
        if translation_file.exists():
            with open(translation_file) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ref_pos = int(parts[0])
                        graph_levels = [int(x) for x in parts[1].split(",") if x]
                        self.level_translation[ref_pos] = graph_levels

    def estimate_insert_size(self, bam_path: str | Path,
                             num_reads: int = 10000) -> tuple[float, float]:
        """Estimate insert size mean and standard deviation from a BAM file.

        Samples properly paired reads and computes insert size statistics.
        Returns (mean, std_dev).
        """
        import pysam

        insert_sizes: list[int] = []
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch():
                if (read.is_proper_pair and not read.is_secondary
                        and not read.is_supplementary and read.template_length > 0):
                    insert_sizes.append(read.template_length)
                    if len(insert_sizes) >= num_reads:
                        break

        if len(insert_sizes) < 100:
            logger.warning(f"Only {len(insert_sizes)} insert sizes sampled, using defaults")
            return 300.0, 50.0

        arr = np.array(insert_sizes, dtype=float)
        # Remove outliers (beyond 3 sigma)
        mu, sigma = np.mean(arr), np.std(arr)
        mask = np.abs(arr - mu) < 3 * sigma
        arr = arr[mask]
        return float(np.mean(arr)), float(np.std(arr))

    def extract_seeds_from_bam(
        self, bam_path: str | Path, is_paired: bool = True
    ) -> tuple[list[OneReadPair], list[VerboseSeedChainPair],
               list[OneRead], list[VerboseSeedChain]]:
        """Extract seed chains from a remapped BAM file.

        Returns:
            - paired_reads: list of OneReadPair
            - paired_chains: list of VerboseSeedChainPair
            - unpaired_reads: list of OneRead
            - unpaired_chains: list of VerboseSeedChain
        """
        import pysam

        proto_seeds: dict[str, ProtoSeed] = defaultdict(ProtoSeed)
        reads_by_name: dict[str, dict[int, OneRead]] = defaultdict(dict)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch():
                if read.is_secondary or read.is_supplementary:
                    continue
                if read.is_unmapped:
                    continue

                name = read.query_name
                seq = read.query_sequence or ""
                qual = read.qual or ""

                read_num = 1 if read.is_read1 or not read.is_paired else 2
                one_read = OneRead(name=name, sequence=seq, quality=qual)
                reads_by_name[name][read_num] = one_read

                # Convert BAM alignment to graph-level seed
                contig = read.reference_name
                pos = read.reference_start
                proto_seeds[name].take_alignment(contig, pos, read, read_num)

        paired_reads: list[OneReadPair] = []
        paired_chains: list[VerboseSeedChainPair] = []
        unpaired_reads: list[OneRead] = []
        unpaired_chains: list[VerboseSeedChain] = []

        for name, proto in proto_seeds.items():
            if is_paired and proto.is_complete():
                r1 = reads_by_name[name].get(1, OneRead())
                r2 = reads_by_name[name].get(2, OneRead())
                paired_reads.append(OneReadPair(read1=r1, read2=r2))

                chain1 = self._alignment_to_seed_chain(proto.read1_alignments, r1)
                chain2 = self._alignment_to_seed_chain(proto.read2_alignments, r2)
                chain1.from_first_read = True
                chain2.from_first_read = False
                paired_chains.append(VerboseSeedChainPair(
                    read_id=name, chain1=chain1, chain2=chain2))
            else:
                for read_num in sorted(reads_by_name[name].keys()):
                    rd = reads_by_name[name][read_num]
                    alns = proto.read1_alignments if read_num == 1 else proto.read2_alignments
                    if alns:
                        unpaired_reads.append(rd)
                        chain = self._alignment_to_seed_chain(alns, rd)
                        unpaired_chains.append(chain)

        logger.info(f"Extracted {len(paired_reads)} paired, {len(unpaired_reads)} unpaired reads")
        return paired_reads, paired_chains, unpaired_reads, unpaired_chains

    def _alignment_to_seed_chain(self, alignments: list, read: OneRead) -> VerboseSeedChain:
        """Convert BAM alignments to a VerboseSeedChain.

        Uses CIGAR operations and the level_translation mapping to determine
        which graph levels the read aligns to.
        """
        chain = VerboseSeedChain()
        chain.read_id = read.name

        if not alignments:
            return chain

        # Use the primary alignment
        _, pos, bam_aln, _ = alignments[0]

        chain.reverse = bam_aln.is_reverse if hasattr(bam_aln, 'is_reverse') else False

        # Convert CIGAR to graph levels
        ref_pos = pos
        seq_pos = 0
        graph_aligned_chars: list[str] = []
        seq_aligned_chars: list[str] = []
        levels: list[int] = []
        edge_ids: list[int] = []

        if hasattr(bam_aln, 'cigartuples') and bam_aln.cigartuples:
            for op, length in bam_aln.cigartuples:
                for _ in range(length):
                    if op == 0:  # M (match/mismatch)
                        level = self.level_translation.get(ref_pos, [ref_pos])
                        levels.append(level[0] if level else -1)
                        seq_char = read.sequence[seq_pos] if seq_pos < len(read.sequence) else "N"
                        graph_aligned_chars.append(seq_char)  # placeholder
                        seq_aligned_chars.append(seq_char)
                        edge_ids.append(-1)
                        ref_pos += 1
                        seq_pos += 1
                    elif op == 1:  # I (insertion to reference)
                        levels.append(-1)
                        graph_aligned_chars.append("_")
                        seq_char = read.sequence[seq_pos] if seq_pos < len(read.sequence) else "N"
                        seq_aligned_chars.append(seq_char)
                        edge_ids.append(-1)
                        seq_pos += 1
                    elif op == 2:  # D (deletion from reference)
                        level = self.level_translation.get(ref_pos, [ref_pos])
                        levels.append(level[0] if level else -1)
                        graph_aligned_chars.append("_")
                        seq_aligned_chars.append("_")
                        edge_ids.append(-1)
                        ref_pos += 1
                    elif op == 4:  # S (soft clip)
                        seq_pos += 1
                    elif op == 5:  # H (hard clip)
                        pass

        chain.graph_aligned = "".join(graph_aligned_chars)
        chain.sequence_aligned = "".join(seq_aligned_chars)
        chain.graph_aligned_levels = levels
        chain.graph_aligned_edges = edge_ids
        chain.sequence_begin = 0
        chain.sequence_end = max(0, seq_pos - 1)

        return chain


def strands_valid(chain1: VerboseSeedChain, chain2: VerboseSeedChain) -> bool:
    """Check if paired-end read strands are valid (FR orientation)."""
    return chain1.reverse != chain2.reverse


def pair_distance_in_levels(chain1: VerboseSeedChain, chain2: VerboseSeedChain) -> int:
    """Compute the distance between paired chains in graph levels."""
    l1 = chain1.alignment_last_level()
    l2 = chain2.alignment_first_level()
    if l1 == -1 or l2 == -1:
        return -1
    return abs(l2 - l1)
