"""Iterative refinement: HLA-HD-style progressive narrowing.

Strategy:
1. Start with pre-filter candidates (~50-100 per locus)
2. Align reads precisely to candidate sequences
3. Score at 2-digit level -> lock in top 2-digit groups
4. Re-score within those groups at 4-digit level -> lock in
5. If genomic sequences available, refine to 6/8-digit

Supports short reads (bowtie2), long reads (minimap2), and RNA-seq (hisat2/minimap2 splice).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pysam

from ..reference.loci import (
    parse_allele_name,
    group_alleles_by_resolution,
)
from ..prefilter.fast_mapper import PrefilterResults
from ..utils.io import write_fasta
from ..utils.external import run_cmd, run_pipeline, check_tool, ToolError

logger = logging.getLogger(__name__)

# Shared bowtie2 config generator
def _bt2_config():
    return {
        "tool": "bowtie2",
        "index_cmd": lambda ref, idx: ["bowtie2-build", "--quiet", str(ref), str(idx)],
        "align_cmd": lambda idx, r1, r2, threads: [
            "bowtie2", "-x", str(idx), "--very-sensitive",
            "-k", "20", "--no-unal", "-p", str(threads),
        ] + (["-1", str(r1), "-2", str(r2)] if r2 else ["-U", str(r1)]),
    }

# Aligner presets per data type
ALIGNER_CONFIG = {
    "short": _bt2_config(),
    "exome": _bt2_config(),             # Same aligner as WGS
    "targeted_capture": _bt2_config(),   # Same aligner, deep coverage
    "pacbio": {
        "tool": "minimap2",
        "index_cmd": lambda ref, idx: ["minimap2", "-d", str(idx), str(ref)],
        "align_cmd": lambda idx, r1, r2, threads: [
            "minimap2", "-a", "-x", "map-pb",
            "-t", str(threads), "--secondary=yes", "-N", "20",
            str(idx), str(r1),
        ],
    },
    "hifi": {
        "tool": "minimap2",
        "index_cmd": lambda ref, idx: ["minimap2", "-d", str(idx), str(ref)],
        "align_cmd": lambda idx, r1, r2, threads: [
            "minimap2", "-a", "-x", "map-hifi",
            "-t", str(threads), "--secondary=yes", "-N", "20",
            str(idx), str(r1),
        ],
    },
    "ont": {
        "tool": "minimap2",
        "index_cmd": lambda ref, idx: ["minimap2", "-d", str(idx), str(ref)],
        "align_cmd": lambda idx, r1, r2, threads: [
            "minimap2", "-a", "-x", "map-ont",
            "-t", str(threads), "--secondary=yes", "-N", "20",
            str(idx), str(r1),
        ],
    },
    "rna": {
        "tool": "minimap2",
        "index_cmd": lambda ref, idx: ["minimap2", "-d", str(idx), str(ref)],
        "align_cmd": lambda idx, r1, r2, threads: [
            "minimap2", "-a", "-x", "splice",
            "-t", str(threads), "--secondary=yes", "-N", "20",
            str(idx),
        ] + ([str(r1), str(r2)] if r2 else [str(r1)]),
    },
}


@dataclass
class RefinementResult:
    """Result of iterative refinement for one locus."""
    locus: str
    level: int
    top_alleles: list[str]
    scores: dict[str, float]
    two_digit_groups: list[str] = field(default_factory=list)
    four_digit_groups: list[str] = field(default_factory=list)


class IterativeRefiner:
    """HLA-HD-style progressive resolution refinement."""

    def __init__(
        self,
        threads: int = 4,
        top_n_2digit: int = 6,
        top_n_4digit: int = 20,
        data_type: str = "short",
    ) -> None:
        if data_type not in ALIGNER_CONFIG:
            raise ValueError(f"Unknown data_type '{data_type}'. Supported: {list(ALIGNER_CONFIG)}")

        self.threads = threads
        self.top_n_2digit = top_n_2digit
        self.top_n_4digit = top_n_4digit
        self.data_type = data_type
        self.config = ALIGNER_CONFIG[data_type]

        check_tool(self.config["tool"])
        check_tool("samtools")

    def refine(
        self,
        r1_fastq: Path,
        r2_fastq: Path | None,
        candidates: dict[str, list[str]],
        allele_sequences: dict[str, str],
        work_dir: Path,
    ) -> dict[str, RefinementResult]:
        """Run iterative refinement for all loci."""
        work_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, RefinementResult] = {}

        for locus, allele_list in candidates.items():
            if not allele_list:
                results[locus] = RefinementResult(
                    locus=locus, level=0, top_alleles=[], scores={},
                )
                continue

            logger.info("Refining %s (%d candidates)", locus, len(allele_list))
            try:
                result = self._refine_locus(
                    r1_fastq, r2_fastq, locus, allele_list,
                    allele_sequences, work_dir / locus,
                )
            except ToolError as e:
                logger.error("Refinement failed for %s: %s", locus, e)
                result = RefinementResult(
                    locus=locus, level=0, top_alleles=allele_list,
                    scores={a: 0 for a in allele_list},
                )
            results[locus] = result

        return results

    def _refine_locus(
        self, r1: Path, r2: Path | None, locus: str,
        alleles: list[str], sequences: dict[str, str], work_dir: Path,
    ) -> RefinementResult:
        """Iteratively refine one locus through resolution levels."""
        work_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Score all candidates
        scores = self._align_and_score(
            r1, r2, alleles, sequences, work_dir / "phase1",
        )

        if not scores:
            return RefinementResult(
                locus=locus, level=0, top_alleles=alleles, scores={},
            )

        # Phase 2: Collapse to 2-digit, pick top groups
        groups_2d = group_alleles_by_resolution(list(scores.keys()), level=1)
        group_scores_2d = {
            g: sum(scores.get(m, 0) for m in members)
            for g, members in groups_2d.items()
        }

        top_2d = sorted(group_scores_2d.items(), key=lambda x: -x[1])
        top_2d = top_2d[:self.top_n_2digit]
        top_2d_names = [g for g, _ in top_2d]

        logger.info("  %s 2-digit groups: %s", locus, top_2d_names)

        # Phase 3: Expand top 2-digit groups -> re-score
        surviving_alleles = []
        for group_name in top_2d_names:
            surviving_alleles.extend(groups_2d.get(group_name, []))

        if len(surviving_alleles) > 1:
            scores_refined = self._align_and_score(
                r1, r2, surviving_alleles, sequences, work_dir / "phase2",
            )
        else:
            scores_refined = {a: scores.get(a, 0) for a in surviving_alleles}

        # Phase 4: Collapse to 4-digit, pick top
        groups_4d = group_alleles_by_resolution(
            list(scores_refined.keys()), level=2,
        )
        group_scores_4d = {
            g: sum(scores_refined.get(m, 0) for m in members)
            for g, members in groups_4d.items()
        }

        top_4d = sorted(group_scores_4d.items(), key=lambda x: -x[1])
        top_4d = top_4d[:self.top_n_4digit]
        top_4d_names = [g for g, _ in top_4d]

        logger.info("  %s 4-digit groups: %s", locus, top_4d_names[:5])

        final_alleles = []
        for group_name in top_4d_names:
            final_alleles.extend(groups_4d.get(group_name, []))

        return RefinementResult(
            locus=locus, level=4, top_alleles=final_alleles,
            scores=scores_refined,
            two_digit_groups=top_2d_names,
            four_digit_groups=top_4d_names,
        )

    def _align_and_score(
        self, r1: Path, r2: Path | None, alleles: list[str],
        sequences: dict[str, str], work_dir: Path,
    ) -> dict[str, float]:
        """Align reads to a set of allele sequences and compute scores."""
        work_dir.mkdir(parents=True, exist_ok=True)

        ref_seqs = {a: sequences[a] for a in alleles if a in sequences}
        if not ref_seqs:
            return {}

        ref_fasta = work_dir / "ref.fa"
        write_fasta(ref_fasta, ref_seqs)

        # Build index
        idx_prefix = work_dir / "idx"
        run_cmd(
            self.config["index_cmd"](ref_fasta, idx_prefix),
            description=f"build {self.config['tool']} index",
        )

        # Align reads -> sorted BAM
        output_bam = work_dir / "aligned.bam"
        r2_for_align = r2 if (r2 and r2.exists()) else None
        align_cmd = self.config["align_cmd"](
            idx_prefix, r1, r2_for_align, self.threads,
        )
        sort_cmd = ["samtools", "sort", "-@", "2", "-"]

        run_pipeline(
            [align_cmd, sort_cmd],
            output_path=output_bam,
            description="refinement alignment",
        )

        run_cmd(
            ["samtools", "index", str(output_bam)],
            description="index refinement BAM",
        )

        # Score: sum of alignment scores per allele
        allele_scores: Counter[str] = Counter()
        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue
                ref = read.reference_name
                if ref:
                    try:
                        score = read.get_tag("AS")
                    except KeyError:
                        score = read.mapping_quality
                    allele_scores[ref] += max(0, score)

        return dict(allele_scores)
