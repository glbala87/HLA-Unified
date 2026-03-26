"""Targeted de novo assembly fallback (HLAminer-style).

When alignment-based methods report low confidence, this module
performs targeted assembly of reads from the problematic locus
to reconstruct the allele sequence directly.

Supports:
- Short reads: megahit or SPAdes
- Long reads: flye or hifiasm
- Matching via k-mer Jaccard + optional minimap2 alignment
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..utils.io import read_fasta, write_fasta
from ..utils.seq import reverse_complement, extract_canonical_kmers
from ..utils.external import run_cmd, ToolError

logger = logging.getLogger(__name__)

# Assembler configurations per data type
ASSEMBLER_CONFIG = {
    "megahit": {"type": "short", "timeout": 300},
    "spades": {"type": "short", "timeout": 600},
    "flye": {"type": "long", "timeout": 600},
    "hifiasm": {"type": "long", "timeout": 600},
}


@dataclass
class MismatchAnnotation:
    """A single mismatch between assembled contig and closest known allele."""
    position: int        # 0-based position in reference allele
    ref_base: str        # base in known allele (or '-' for insertion)
    alt_base: str        # base in assembled contig (or '-' for deletion)
    variant_type: str    # 'SNP', 'INS', 'DEL'


@dataclass
class NovelAlleleReport:
    """Detailed annotation of a potential novel allele."""
    closest_allele: str
    identity: float
    mismatches: list[MismatchAnnotation]
    n_snps: int
    n_insertions: int
    n_deletions: int
    aligned_length: int
    summary: str  # human-readable summary


@dataclass
class AssemblyResult:
    """Result of targeted assembly for one locus."""
    locus: str
    contigs: dict[str, str]
    n_contigs: int
    total_length: int
    best_match_allele: str
    best_match_identity: float
    is_novel: bool
    novel_report: NovelAlleleReport | None = None


class TargetedAssembler:
    """Targeted de novo assembly for low-confidence HLA calls."""

    def __init__(
        self,
        assembler: str = "megahit",
        threads: int = 4,
        identity_threshold: float = 0.995,
        data_type: str = "short",
    ) -> None:
        self.threads = threads
        self.identity_threshold = identity_threshold
        self.data_type = data_type

        # Auto-select assembler based on data type if default doesn't match
        if data_type in ("pacbio", "hifi", "ont") and assembler in ("megahit", "spades"):
            assembler = "flye"
            logger.info("Auto-selected flye assembler for long-read data")
        elif data_type in ("short", "rna") and assembler in ("flye", "hifiasm"):
            assembler = "megahit"
            logger.info("Auto-selected megahit assembler for short-read data")

        self.assembler = assembler

        # Check if assembler is available (warn, don't fail — assembly is optional)
        if not shutil.which(assembler):
            logger.warning(
                "Assembler '%s' not found on PATH. "
                "Assembly fallback will be skipped.", assembler
            )

    def assemble_locus(
        self,
        r1_fastq: Path,
        r2_fastq: Path | None,
        locus: str,
        allele_db: dict[str, str],
        work_dir: Path,
    ) -> AssemblyResult:
        """Run targeted assembly and match against known alleles."""
        work_dir.mkdir(parents=True, exist_ok=True)

        if not shutil.which(self.assembler):
            return AssemblyResult(
                locus=locus, contigs={}, n_contigs=0, total_length=0,
                best_match_allele="", best_match_identity=0.0, is_novel=False,
            )

        contigs = self._run_assembly(r1_fastq, r2_fastq, work_dir)

        if not contigs:
            return AssemblyResult(
                locus=locus, contigs={}, n_contigs=0, total_length=0,
                best_match_allele="", best_match_identity=0.0, is_novel=True,
            )

        best_allele, best_identity = self._match_contigs(
            contigs, allele_db, work_dir,
        )

        is_novel = best_identity < self.identity_threshold

        # Generate detailed mismatch annotation for novel alleles
        novel_report = None
        if is_novel and best_allele and best_allele in allele_db:
            novel_report = self._annotate_mismatches(
                contigs, best_allele, allele_db[best_allele], work_dir,
            )
            logger.info(
                "%s: potential novel allele (best match %s at %.2f%%, "
                "%d SNPs, %d ins, %d del)",
                locus, best_allele, best_identity * 100,
                novel_report.n_snps, novel_report.n_insertions,
                novel_report.n_deletions,
            )

        return AssemblyResult(
            locus=locus,
            contigs=contigs,
            n_contigs=len(contigs),
            total_length=sum(len(s) for s in contigs.values()),
            best_match_allele=best_allele,
            novel_report=novel_report,
            best_match_identity=best_identity,
            is_novel=is_novel,
        )

    def _run_assembly(
        self, r1: Path, r2: Path | None, work_dir: Path,
    ) -> dict[str, str]:
        """Run the assembler on locus-specific reads."""
        output_dir = work_dir / "assembly"
        config = ASSEMBLER_CONFIG.get(self.assembler, {"timeout": 300})
        timeout = config["timeout"]

        try:
            if self.assembler == "megahit":
                cmd = [
                    "megahit", "-t", str(self.threads),
                    "--min-contig-len", "200", "-o", str(output_dir),
                ]
                if r2 and r2.exists():
                    cmd.extend(["-1", str(r1), "-2", str(r2)])
                else:
                    cmd.extend(["-r", str(r1)])
                contig_file = output_dir / "final.contigs.fa"

            elif self.assembler == "spades":
                cmd = [
                    "spades.py", "-t", str(self.threads),
                    "--careful", "-o", str(output_dir),
                ]
                if r2 and r2.exists():
                    cmd.extend(["-1", str(r1), "-2", str(r2)])
                else:
                    cmd.extend(["-s", str(r1)])
                contig_file = output_dir / "contigs.fasta"

            elif self.assembler == "flye":
                # Flye for long reads
                read_type_flag = {
                    "pacbio": "--pacbio-raw",
                    "hifi": "--pacbio-hifi",
                    "ont": "--nano-raw",
                }.get(self.data_type, "--pacbio-raw")
                cmd = [
                    "flye", read_type_flag, str(r1),
                    "-t", str(self.threads),
                    "--genome-size", "15k",  # HLA gene size estimate
                    "-o", str(output_dir),
                ]
                contig_file = output_dir / "assembly.fasta"

            elif self.assembler == "hifiasm":
                cmd = [
                    "hifiasm", "-t", str(self.threads),
                    "-o", str(output_dir / "asm"),
                    str(r1),
                ]
                contig_file = output_dir / "asm.bp.p_ctg.fa"

            else:
                logger.error("Unknown assembler: %s", self.assembler)
                return {}

            subprocess.run(
                cmd, check=True, capture_output=True, timeout=timeout,
            )

        except FileNotFoundError:
            logger.warning("Assembler '%s' not found", self.assembler)
            return {}
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode()[:500] if e.stderr else ""
            logger.warning("Assembly failed for %s: %s", self.assembler, stderr)
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("Assembly timed out after %ds", timeout)
            return {}

        if not contig_file.exists():
            logger.warning("Assembly produced no contigs: %s", contig_file)
            return {}

        return dict(read_fasta(contig_file))

    def _match_contigs(
        self,
        contigs: dict[str, str],
        allele_db: dict[str, str],
        work_dir: Path,
    ) -> tuple[str, float]:
        """Match assembled contigs against known alleles.

        Uses k-mer Jaccard similarity, then refines top hits with
        minimap2 alignment if available.
        """
        all_contig_seq = "".join(contigs.values())
        if not all_contig_seq:
            return "", 0.0

        contig_kmers = extract_canonical_kmers(all_contig_seq, k=21)

        best_allele = ""
        best_identity = 0.0

        # Phase 1: k-mer Jaccard for fast screening
        top_hits: list[tuple[str, float]] = []
        for allele_name, allele_seq in allele_db.items():
            allele_kmers = extract_canonical_kmers(allele_seq, k=21)
            if not allele_kmers:
                continue

            intersection = len(contig_kmers & allele_kmers)
            union = len(contig_kmers | allele_kmers)
            if union == 0:
                continue

            jaccard = intersection / union
            top_hits.append((allele_name, jaccard))

        top_hits.sort(key=lambda x: -x[1])

        if not top_hits:
            return "", 0.0

        # Phase 2: Refine top 5 hits with minimap2 if available
        if shutil.which("minimap2") and len(top_hits) >= 1:
            refined = self._refine_with_alignment(
                contigs, {n: allele_db[n] for n, _ in top_hits[:5]}, work_dir,
            )
            if refined[1] > 0:
                return refined

        # Fall back to k-mer Jaccard
        return top_hits[0][0], top_hits[0][1]

    def _refine_with_alignment(
        self,
        contigs: dict[str, str],
        candidates: dict[str, str],
        work_dir: Path,
    ) -> tuple[str, float]:
        """Refine match using minimap2 alignment for precise identity."""
        contig_fa = work_dir / "contigs.fa"
        ref_fa = work_dir / "match_ref.fa"
        write_fasta(contig_fa, contigs)
        write_fasta(ref_fa, candidates)

        try:
            result = run_cmd(
                ["minimap2", "-a", "-x", "asm5", str(ref_fa), str(contig_fa)],
                description="match contigs to alleles",
            )
        except ToolError:
            return "", 0.0

        # Parse SAM output for best alignment identity
        best_allele = ""
        best_identity = 0.0

        for line in result.stdout.split("\n"):
            if line.startswith("@"):
                continue
            fields = line.split("\t")
            if len(fields) < 11:
                continue
            ref_name = fields[2]
            if ref_name == "*":
                continue

            # Compute identity from CIGAR + NM tag
            cigar = fields[5]
            nm = 0
            for tag in fields[11:]:
                if tag.startswith("NM:i:"):
                    nm = int(tag[5:])
                    break

            aligned_len = sum(
                int(n) for n, op in self._parse_cigar(cigar)
                if op in "MID"
            )
            if aligned_len > 0:
                identity = (aligned_len - nm) / aligned_len
                if identity > best_identity:
                    best_identity = identity
                    best_allele = ref_name

        return best_allele, best_identity

    @staticmethod
    def _parse_cigar(cigar: str) -> list[tuple[int, str]]:
        """Parse a CIGAR string into (length, operation) tuples."""
        ops = []
        num = ""
        for ch in cigar:
            if ch.isdigit():
                num += ch
            else:
                if num:
                    ops.append((int(num), ch))
                num = ""
        return ops

    def _annotate_mismatches(
        self,
        contigs: dict[str, str],
        ref_allele: str,
        ref_seq: str,
        work_dir: Path,
    ) -> NovelAlleleReport:
        """Generate detailed mismatch annotation between contig and closest allele.

        Uses minimap2 for precise alignment, then walks the CIGAR+MD to
        extract each SNP, insertion, and deletion with positions.
        """
        contig_fa = work_dir / "novel_contig.fa"
        ref_fa = work_dir / "novel_ref.fa"
        write_fasta(contig_fa, contigs)
        write_fasta(ref_fa, {ref_allele: ref_seq})

        mismatches: list[MismatchAnnotation] = []

        try:
            result = run_cmd(
                ["minimap2", "-a", "-x", "asm5", "--MD",
                 str(ref_fa), str(contig_fa)],
                description="annotate novel allele mismatches",
            )
        except ToolError:
            return NovelAlleleReport(
                closest_allele=ref_allele, identity=0.0,
                mismatches=[], n_snps=0, n_insertions=0, n_deletions=0,
                aligned_length=0, summary="alignment failed",
            )

        aligned_length = 0

        for line in result.stdout.split("\n"):
            if line.startswith("@") or not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) < 11 or fields[2] == "*":
                continue

            cigar = fields[5]
            query_seq = fields[9]
            ref_pos = int(fields[3]) - 1  # Convert to 0-based

            # Extract MD tag for mismatch details
            md_tag = ""
            for tag in fields[11:]:
                if tag.startswith("MD:Z:"):
                    md_tag = tag[5:]
                    break

            cigar_ops = self._parse_cigar(cigar)
            aligned_length = sum(n for n, op in cigar_ops if op in "MID")

            # Walk CIGAR + query to find mismatches
            q_pos = 0
            r_pos = ref_pos

            for length, op in cigar_ops:
                if op == "M" or op == "=" or op == "X":
                    # Match/mismatch block — check each position
                    for i in range(length):
                        if q_pos < len(query_seq) and r_pos < len(ref_seq):
                            q_base = query_seq[q_pos]
                            r_base = ref_seq[r_pos]
                            if q_base != r_base and q_base in "ACGT" and r_base in "ACGT":
                                mismatches.append(MismatchAnnotation(
                                    position=r_pos,
                                    ref_base=r_base,
                                    alt_base=q_base,
                                    variant_type="SNP",
                                ))
                        q_pos += 1
                        r_pos += 1
                elif op == "I":
                    # Insertion in query relative to reference
                    ins_seq = query_seq[q_pos:q_pos + length] if q_pos + length <= len(query_seq) else "?"
                    mismatches.append(MismatchAnnotation(
                        position=r_pos,
                        ref_base="-",
                        alt_base=ins_seq,
                        variant_type="INS",
                    ))
                    q_pos += length
                elif op == "D":
                    # Deletion in query relative to reference
                    del_seq = ref_seq[r_pos:r_pos + length] if r_pos + length <= len(ref_seq) else "?"
                    mismatches.append(MismatchAnnotation(
                        position=r_pos,
                        ref_base=del_seq,
                        alt_base="-",
                        variant_type="DEL",
                    ))
                    r_pos += length
                elif op == "S" or op == "H":
                    if op == "S":
                        q_pos += length
                elif op == "N":
                    r_pos += length

            break  # Use first (primary) alignment only

        n_snps = sum(1 for m in mismatches if m.variant_type == "SNP")
        n_ins = sum(1 for m in mismatches if m.variant_type == "INS")
        n_del = sum(1 for m in mismatches if m.variant_type == "DEL")

        identity = (aligned_length - len(mismatches)) / max(aligned_length, 1)

        summary = (
            f"Closest: {ref_allele} ({identity:.2%} identity over {aligned_length}bp). "
            f"Variants: {n_snps} SNPs, {n_ins} insertions, {n_del} deletions."
        )
        if mismatches:
            top = mismatches[:5]
            details = "; ".join(
                f"pos{m.position}:{m.ref_base}>{m.alt_base}({m.variant_type})"
                for m in top
            )
            summary += f" Top variants: {details}"
            if len(mismatches) > 5:
                summary += f" ...and {len(mismatches) - 5} more"

        return NovelAlleleReport(
            closest_allele=ref_allele,
            identity=identity,
            mismatches=mismatches,
            n_snps=n_snps,
            n_insertions=n_ins,
            n_deletions=n_del,
            aligned_length=aligned_length,
            summary=summary,
        )
