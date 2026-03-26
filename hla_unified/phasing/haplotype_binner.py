"""Haplotype binning and read-backed phasing for HLA loci.

Core step in the hybrid engine between refinement and ILP genotyping.
Separates reads into two haplotype bins (one per chromosome) using
heterozygous SNP positions as phase anchors.

Strategy:
1. Identify heterozygous SNP positions from read pileup across candidates
2. Build a read-SNP matrix linking each read to its observed alleles at het sites
3. Cluster reads into two haplotype bins (spectral clustering or greedy)
4. Within each bin, run local consensus to reconstruct the haplotype sequence
5. Match each haplotype to the best candidate allele

This dramatically improves accuracy for heterozygous loci where reads
from two alleles are mixed, which confuses alignment-only approaches.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam

from ..reference.loci import parse_allele_name

logger = logging.getLogger(__name__)


@dataclass
class HetSite:
    """A heterozygous SNP position detected in the pileup."""
    ref_name: str  # reference allele name
    position: int  # 0-based position in reference
    alleles: dict[str, int]  # base -> count
    major: str
    minor: str
    depth: int


@dataclass
class HaplotypeBin:
    """A set of reads assigned to one haplotype."""
    bin_id: int  # 0 or 1
    read_names: list[str]
    snp_profile: dict[int, str]  # position -> consensus base at het sites
    consensus_seq: str = ""
    best_allele: str = ""
    allele_score: float = 0.0


@dataclass
class PhasingResult:
    """Result of haplotype phasing for one locus."""
    locus: str
    het_sites: list[HetSite]
    n_het_sites: int
    bins: list[HaplotypeBin]  # exactly 2 for diploid
    is_phased: bool  # True if enough het sites to phase
    phase_confidence: float  # fraction of reads confidently assigned


class HaplotypeBinner:
    """Read-backed haplotype phasing for HLA loci.

    Uses heterozygous SNPs detected from the read pileup to separate
    reads into two haplotype bins, then matches each bin to a candidate allele.
    """

    def __init__(
        self,
        min_het_depth: int = 5,
        min_minor_af: float = 0.15,
        max_minor_af: float = 0.85,
        min_het_sites: int = 2,
        min_base_quality: int = 20,
    ) -> None:
        self.min_het_depth = min_het_depth
        self.min_minor_af = min_minor_af
        self.max_minor_af = max_minor_af
        self.min_het_sites = min_het_sites
        self.min_base_quality = min_base_quality

    def phase_locus(
        self,
        bam_path: Path,
        locus: str,
        candidate_alleles: list[str],
        allele_sequences: dict[str, str],
    ) -> PhasingResult:
        """Phase reads at a single locus into two haplotype bins.

        Args:
            bam_path: BAM file aligned to candidate alleles
            locus: HLA locus name
            candidate_alleles: Allele names for this locus
            allele_sequences: allele_name -> sequence
        """
        # Filter candidates to this locus
        locus_alleles = [
            a for a in candidate_alleles
            if parse_allele_name(a).locus == locus
        ]

        if not locus_alleles:
            return self._unphased_result(locus)

        # Use the top-scoring allele as phasing reference
        ref_allele = locus_alleles[0]

        # Step 1: Detect heterozygous sites from pileup
        het_sites = self._detect_het_sites(bam_path, ref_allele)

        if len(het_sites) < self.min_het_sites:
            logger.info(
                "%s: only %d het sites found (need %d), skipping phasing",
                locus, len(het_sites), self.min_het_sites,
            )
            return self._unphased_result(locus, het_sites)

        # Step 2: Build read-SNP matrix
        read_snp_matrix, read_names = self._build_read_snp_matrix(
            bam_path, ref_allele, het_sites,
        )

        if len(read_names) < 4:
            return self._unphased_result(locus, het_sites)

        # Step 3: Cluster reads into 2 haplotype bins
        # Select algorithm based on sequencing modality
        data_type = getattr(self, "_data_type", "short")
        if data_type in ("pacbio", "hifi", "ont"):
            logger.debug("%s: using long-read clustering", locus)
            assignments = self._cluster_reads_longread(read_snp_matrix, het_sites)
        else:
            assignments = self._cluster_reads(read_snp_matrix, het_sites)

        # Fallback to spectral clustering if greedy/longread produced
        # highly imbalanced bins (>90% in one bin)
        n_bin0 = int(np.sum(assignments == 0))
        n_bin1 = int(np.sum(assignments == 1))
        n_assigned = n_bin0 + n_bin1
        if n_assigned > 0:
            balance = min(n_bin0, n_bin1) / max(n_assigned, 1)
            if balance < 0.10 and len(het_sites) >= 3:
                logger.debug(
                    "%s: imbalanced bins (%.0f%%), retrying with spectral clustering",
                    locus, balance * 100,
                )
                assignments = self._cluster_reads_spectral(read_snp_matrix, het_sites)

        # Step 4: Build haplotype bins
        bins = self._build_bins(
            assignments, read_names, read_snp_matrix, het_sites,
        )

        # Step 5: Match bins to candidate alleles
        for hap_bin in bins:
            best_allele, score = self._match_bin_to_allele(
                hap_bin, locus_alleles, allele_sequences, het_sites,
            )
            hap_bin.best_allele = best_allele
            hap_bin.allele_score = score

        # Compute phasing confidence
        n_assigned = sum(1 for a in assignments if a >= 0)
        phase_confidence = n_assigned / max(len(assignments), 1)

        logger.info(
            "%s: phased with %d het sites, %.0f%% reads assigned, "
            "bins: %s (%d reads) / %s (%d reads)",
            locus, len(het_sites), phase_confidence * 100,
            bins[0].best_allele, len(bins[0].read_names),
            bins[1].best_allele, len(bins[1].read_names),
        )

        return PhasingResult(
            locus=locus,
            het_sites=het_sites,
            n_het_sites=len(het_sites),
            bins=bins,
            is_phased=True,
            phase_confidence=phase_confidence,
        )

    def _detect_het_sites(
        self, bam_path: Path, ref_allele: str,
    ) -> list[HetSite]:
        """Detect heterozygous SNP positions from pileup."""
        het_sites = []

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            # Check if reference exists in BAM
            refs = bam.references
            if ref_allele not in refs:
                return het_sites

            for col in bam.pileup(
                ref_allele, min_base_quality=self.min_base_quality,
            ):
                bases = Counter()
                for read in col.pileups:
                    if read.is_del or read.is_refskip:
                        continue
                    base = read.alignment.query_sequence[read.query_position]
                    if base in "ACGT":
                        bases[base] += 1

                depth = sum(bases.values())
                if depth < self.min_het_depth:
                    continue

                # Check for heterozygosity
                if len(bases) < 2:
                    continue

                top2 = bases.most_common(2)
                major_base, major_count = top2[0]
                minor_base, minor_count = top2[1]
                minor_af = minor_count / depth

                if self.min_minor_af <= minor_af <= self.max_minor_af:
                    het_sites.append(HetSite(
                        ref_name=ref_allele,
                        position=col.reference_pos,
                        alleles=dict(bases),
                        major=major_base,
                        minor=minor_base,
                        depth=depth,
                    ))

        logger.debug("Detected %d het sites for %s", len(het_sites), ref_allele)
        return het_sites

    def _build_read_snp_matrix(
        self,
        bam_path: Path,
        ref_allele: str,
        het_sites: list[HetSite],
    ) -> tuple[np.ndarray, list[str]]:
        """Build matrix: rows=reads, cols=het sites, values=0(major)/1(minor)/-1(missing)."""
        site_positions = {s.position: i for i, s in enumerate(het_sites)}
        read_data: dict[str, dict[int, int]] = defaultdict(dict)

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(ref_allele):
                if read.is_unmapped or not read.query_sequence:
                    continue

                pairs = read.get_aligned_pairs(matches_only=True)
                for query_pos, ref_pos in pairs:
                    if ref_pos in site_positions:
                        base = read.query_sequence[query_pos]
                        site = het_sites[site_positions[ref_pos]]
                        if base == site.major:
                            read_data[read.query_name][site_positions[ref_pos]] = 0
                        elif base == site.minor:
                            read_data[read.query_name][site_positions[ref_pos]] = 1
                        # else: non-major/minor base, treat as missing

        read_names = sorted(read_data.keys())
        n_reads = len(read_names)
        n_sites = len(het_sites)

        matrix = np.full((n_reads, n_sites), -1, dtype=np.int8)
        for i, rname in enumerate(read_names):
            for col, val in read_data[rname].items():
                matrix[i, col] = val

        return matrix, read_names

    def _cluster_reads(
        self,
        matrix: np.ndarray,
        het_sites: list[HetSite],
    ) -> np.ndarray:
        """Cluster reads into 2 haplotype bins using greedy linkage.

        For each read, compute its haplotype signature (pattern of
        major/minor alleles at het sites). Group reads with consistent
        signatures into the same bin.
        """
        n_reads, n_sites = matrix.shape
        assignments = np.full(n_reads, -1, dtype=np.int32)

        # Find seed reads: those covering the most het sites
        coverage = np.sum(matrix >= 0, axis=1)
        sorted_reads = np.argsort(-coverage)

        # Seed bin 0 with the highest-coverage read
        seed0 = sorted_reads[0]
        assignments[seed0] = 0

        # Find the most different read as seed for bin 1
        seed0_sig = matrix[seed0]
        best_diff = -1
        seed1 = -1
        for idx in sorted_reads[1:]:
            overlap = (seed0_sig >= 0) & (matrix[idx] >= 0)
            if overlap.sum() == 0:
                continue
            diff = np.sum(seed0_sig[overlap] != matrix[idx][overlap])
            if diff > best_diff:
                best_diff = diff
                seed1 = idx

        if seed1 >= 0:
            assignments[seed1] = 1
        else:
            # No opposing read found; can't phase
            return assignments

        # Assign remaining reads to the most similar seed
        bin_profiles = [matrix[seed0].copy(), matrix[seed1].copy()]

        for idx in sorted_reads:
            if assignments[idx] >= 0:
                continue

            scores = [0, 0]
            for b in range(2):
                overlap = (bin_profiles[b] >= 0) & (matrix[idx] >= 0)
                if overlap.sum() == 0:
                    continue
                matches = np.sum(bin_profiles[b][overlap] == matrix[idx][overlap])
                mismatches = np.sum(bin_profiles[b][overlap] != matrix[idx][overlap])
                scores[b] = matches - mismatches

            if scores[0] == scores[1] == 0:
                continue  # unassignable
            assignments[idx] = 0 if scores[0] >= scores[1] else 1

        return assignments

    def _build_bins(
        self,
        assignments: np.ndarray,
        read_names: list[str],
        matrix: np.ndarray,
        het_sites: list[HetSite],
    ) -> list[HaplotypeBin]:
        """Build haplotype bin objects from read assignments."""
        bins = []
        for bin_id in range(2):
            mask = assignments == bin_id
            bin_reads = [read_names[i] for i in range(len(read_names)) if mask[i]]

            # Consensus SNP profile for this bin
            snp_profile = {}
            if mask.sum() > 0:
                bin_matrix = matrix[mask]
                for col in range(len(het_sites)):
                    values = bin_matrix[:, col]
                    valid = values[values >= 0]
                    if len(valid) > 0:
                        consensus = int(np.round(np.mean(valid)))
                        site = het_sites[col]
                        snp_profile[site.position] = (
                            site.major if consensus == 0 else site.minor
                        )

            bins.append(HaplotypeBin(
                bin_id=bin_id,
                read_names=bin_reads,
                snp_profile=snp_profile,
            ))

        return bins

    def _match_bin_to_allele(
        self,
        hap_bin: HaplotypeBin,
        candidates: list[str],
        sequences: dict[str, str],
        het_sites: list[HetSite],
    ) -> tuple[str, float]:
        """Match a haplotype bin to the best candidate allele by SNP concordance."""
        best_allele = ""
        best_score = -1.0

        for allele in candidates:
            seq = sequences.get(allele, "")
            if not seq:
                continue

            matches = 0
            total = 0
            for pos, base in hap_bin.snp_profile.items():
                if pos < len(seq):
                    total += 1
                    if seq[pos] == base:
                        matches += 1

            score = matches / max(total, 1)
            if score > best_score:
                best_score = score
                best_allele = allele

        return best_allele, best_score

    def _unphased_result(
        self, locus: str, het_sites: list[HetSite] | None = None,
    ) -> PhasingResult:
        """Return an unphased result when phasing is not possible."""
        return PhasingResult(
            locus=locus,
            het_sites=het_sites or [],
            n_het_sites=len(het_sites) if het_sites else 0,
            bins=[
                HaplotypeBin(bin_id=0, read_names=[], snp_profile={}),
                HaplotypeBin(bin_id=1, read_names=[], snp_profile={}),
            ],
            is_phased=False,
            phase_confidence=0.0,
        )

    def _cluster_reads_longread(
        self,
        matrix: np.ndarray,
        het_sites: list[HetSite],
    ) -> np.ndarray:
        """Cluster long reads into 2 haplotype bins.

        Long reads typically span many het sites simultaneously, making
        clustering more reliable. Uses consistency voting: each read's
        full het-site pattern defines a haplotype signature.
        """
        n_reads, n_sites = matrix.shape
        assignments = np.full(n_reads, -1, dtype=np.int32)

        # Long reads cover many sites — find the two most common signatures
        signatures: dict[tuple, list[int]] = {}
        for i in range(n_reads):
            # Create signature from observed bases (ignore missing)
            sig = tuple(matrix[i])
            signatures.setdefault(sig, []).append(i)

        if len(signatures) < 2:
            # All reads have same signature — likely homozygous
            for reads in signatures.values():
                for r in reads:
                    assignments[r] = 0
            return assignments

        # Group by similarity: compute pairwise hamming distance of signatures
        sig_list = list(signatures.keys())
        sig_reads = [signatures[s] for s in sig_list]

        # Split into two groups by most divergent pair
        best_div = -1
        best_i, best_j = 0, 1
        for i in range(len(sig_list)):
            for j in range(i + 1, len(sig_list)):
                overlap = [
                    (sig_list[i][k], sig_list[j][k])
                    for k in range(n_sites)
                    if sig_list[i][k] >= 0 and sig_list[j][k] >= 0
                ]
                if overlap:
                    diff = sum(1 for a, b in overlap if a != b)
                    if diff > best_div:
                        best_div = diff
                        best_i, best_j = i, j

        # Assign reads to bins based on similarity to the two seed signatures
        ref0 = np.array(sig_list[best_i])
        ref1 = np.array(sig_list[best_j])

        for sig, reads in zip(sig_list, sig_reads):
            sig_arr = np.array(sig)
            overlap0 = (ref0 >= 0) & (sig_arr >= 0)
            overlap1 = (ref1 >= 0) & (sig_arr >= 0)

            score0 = np.sum(ref0[overlap0] == sig_arr[overlap0]) if overlap0.any() else 0
            score1 = np.sum(ref1[overlap1] == sig_arr[overlap1]) if overlap1.any() else 0

            bin_id = 0 if score0 >= score1 else 1
            for r in reads:
                assignments[r] = bin_id

        return assignments

    def _cluster_reads_spectral(
        self,
        matrix: np.ndarray,
        het_sites: list[HetSite],
    ) -> np.ndarray:
        """Spectral clustering fallback for difficult phasing cases.

        Builds a similarity matrix from read-SNP patterns and uses
        spectral clustering (Laplacian eigenvector) to partition reads
        into 2 haplotype bins.
        """
        n_reads, n_sites = matrix.shape
        assignments = np.full(n_reads, -1, dtype=np.int32)

        # Build similarity matrix
        sim = np.zeros((n_reads, n_reads))
        for i in range(n_reads):
            for j in range(i + 1, n_reads):
                overlap = (matrix[i] >= 0) & (matrix[j] >= 0)
                if overlap.sum() == 0:
                    continue
                matches = np.sum(matrix[i][overlap] == matrix[j][overlap])
                mismatches = np.sum(matrix[i][overlap] != matrix[j][overlap])
                # Similarity: match fraction
                sim[i, j] = sim[j, i] = matches / (matches + mismatches) if (matches + mismatches) > 0 else 0.5

        # Laplacian eigenvector clustering
        degree = sim.sum(axis=1)
        degree[degree == 0] = 1e-10
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        laplacian = np.eye(n_reads) - d_inv_sqrt @ sim @ d_inv_sqrt

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            # Use Fiedler vector (2nd smallest eigenvalue)
            fiedler = eigenvectors[:, 1]
            median = np.median(fiedler)
            for i in range(n_reads):
                assignments[i] = 0 if fiedler[i] <= median else 1
        except np.linalg.LinAlgError:
            # Fallback to greedy
            return self._cluster_reads(matrix, het_sites)

        return assignments

    def phase_all_loci(
        self,
        bam_path: Path,
        loci: list[str],
        candidates: dict[str, list[str]],
        sequences: dict[str, str],
        data_type: str = "short",
    ) -> dict[str, PhasingResult]:
        """Phase all loci.

        Args:
            data_type: Sequencing modality. Long-read modes use a
                       specialized clustering algorithm.
        """
        self._data_type = data_type
        results = {}
        for locus in loci:
            alleles = candidates.get(locus, [])
            results[locus] = self.phase_locus(
                bam_path, locus, alleles, sequences,
            )
        return results
