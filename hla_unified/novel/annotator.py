"""Variant annotation for novel HLA alleles.

Maps mismatch positions from assembly to gene model coordinates,
classifies variants (synonymous/non-synonymous/splice), and generates
structured annotation in a format suitable for submission or review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..assembly.targeted_assembler import MismatchAnnotation, NovelAlleleReport
from ..reference.loci import parse_allele_name, TYPING_EXONS

logger = logging.getLogger(__name__)

# Codon table (standard genetic code)
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass
class AnnotatedVariant:
    """A variant annotated with gene model context."""
    position: int              # 0-based in reference allele
    ref_base: str
    alt_base: str
    variant_type: str          # SNP, INS, DEL
    region: str                # exon_2, intron_1, utr_5, etc.
    is_in_typing_exon: bool
    codon_change: str          # e.g., "AGT>ACT" (for SNPs in CDS)
    amino_acid_change: str     # e.g., "S>T" or "synonymous"
    functional_class: str      # synonymous, non_synonymous, splice, frameshift, non_coding
    hgvs_like: str             # HGVS-like notation

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "ref_base": self.ref_base,
            "alt_base": self.alt_base,
            "variant_type": self.variant_type,
            "region": self.region,
            "is_in_typing_exon": self.is_in_typing_exon,
            "codon_change": self.codon_change,
            "amino_acid_change": self.amino_acid_change,
            "functional_class": self.functional_class,
            "hgvs_like": self.hgvs_like,
        }


@dataclass
class NovelAlleleAnnotation:
    """Complete annotation for a novel allele candidate."""
    locus: str
    closest_allele: str
    identity: float
    variants: list[AnnotatedVariant]
    n_coding_variants: int
    n_non_synonymous: int
    n_synonymous: int
    n_non_coding: int
    affects_typing_exons: bool
    temporary_designation: str  # e.g., "A*02:new"
    summary: str

    def to_dict(self) -> dict:
        return {
            "locus": self.locus,
            "closest_allele": self.closest_allele,
            "identity": round(self.identity, 6),
            "temporary_designation": self.temporary_designation,
            "variants": [v.to_dict() for v in self.variants],
            "variant_summary": {
                "coding": self.n_coding_variants,
                "non_synonymous": self.n_non_synonymous,
                "synonymous": self.n_synonymous,
                "non_coding": self.n_non_coding,
            },
            "affects_typing_exons": self.affects_typing_exons,
            "summary": self.summary,
        }


class NovelAlleleAnnotator:
    """Annotates novel allele variants with gene model context.

    Given mismatches from assembly alignment, determines which region
    of the gene each variant falls in, classifies coding impact, and
    generates structured annotation.
    """

    # Simplified HLA gene model: approximate exon boundaries
    # Real implementation would load from IMGT gene feature tables
    GENE_MODEL: dict[str, list[tuple[str, int, int]]] = {
        "A": [
            ("utr_5", 0, 300), ("exon_1", 300, 573), ("intron_1", 573, 700),
            ("exon_2", 700, 970), ("intron_2", 970, 1200),
            ("exon_3", 1200, 1470), ("intron_3", 1470, 1900),
            ("exon_4", 1900, 2176), ("intron_4", 2176, 2400),
            ("exon_5", 2400, 2500), ("utr_3", 2500, 3500),
        ],
    }

    def annotate(
        self,
        locus: str,
        closest_allele: str,
        ref_seq: str,
        novel_report: NovelAlleleReport,
    ) -> NovelAlleleAnnotation:
        """Annotate all mismatches from a novel allele report."""
        gene_model = self._get_gene_model(locus, len(ref_seq))
        typing_exons = TYPING_EXONS.get(locus, [2, 3])

        annotated: list[AnnotatedVariant] = []

        for mismatch in novel_report.mismatches:
            region = self._find_region(mismatch.position, gene_model)
            is_typing = any(
                f"exon_{e}" == region for e in typing_exons
            )

            # Classify coding impact
            codon_change = ""
            aa_change = ""
            func_class = "non_coding"

            if region.startswith("exon_"):
                if mismatch.variant_type == "SNP":
                    codon_change, aa_change, func_class = self._classify_snp(
                        mismatch.position, mismatch.ref_base, mismatch.alt_base,
                        ref_seq, gene_model, region,
                    )
                elif mismatch.variant_type == "INS":
                    func_class = "frameshift" if len(mismatch.alt_base) % 3 != 0 else "in_frame_ins"
                elif mismatch.variant_type == "DEL":
                    func_class = "frameshift" if len(mismatch.ref_base) % 3 != 0 else "in_frame_del"
            elif region.startswith("intron_"):
                func_class = "intronic"

            # HGVS-like notation
            hgvs = self._make_hgvs(mismatch, closest_allele)

            annotated.append(AnnotatedVariant(
                position=mismatch.position,
                ref_base=mismatch.ref_base,
                alt_base=mismatch.alt_base,
                variant_type=mismatch.variant_type,
                region=region,
                is_in_typing_exon=is_typing,
                codon_change=codon_change,
                amino_acid_change=aa_change,
                functional_class=func_class,
                hgvs_like=hgvs,
            ))

        # Counts
        n_coding = sum(1 for v in annotated if v.region.startswith("exon_"))
        n_nonsyn = sum(1 for v in annotated if v.functional_class == "non_synonymous")
        n_syn = sum(1 for v in annotated if v.functional_class == "synonymous")
        n_noncoding = sum(1 for v in annotated if v.functional_class in ("intronic", "non_coding"))
        affects_typing = any(v.is_in_typing_exon for v in annotated)

        # Temporary designation
        info = parse_allele_name(closest_allele)
        temp_name = f"{info.locus}*{info.field1}:new"

        summary = (
            f"Novel allele closest to {closest_allele} "
            f"({novel_report.identity:.2%} identity). "
            f"{len(annotated)} variants: "
            f"{n_nonsyn} non-synonymous, {n_syn} synonymous, "
            f"{n_noncoding} non-coding."
        )
        if affects_typing:
            summary += " AFFECTS TYPING EXONS."

        return NovelAlleleAnnotation(
            locus=locus,
            closest_allele=closest_allele,
            identity=novel_report.identity,
            variants=annotated,
            n_coding_variants=n_coding,
            n_non_synonymous=n_nonsyn,
            n_synonymous=n_syn,
            n_non_coding=n_noncoding,
            affects_typing_exons=affects_typing,
            temporary_designation=temp_name,
            summary=summary,
        )

    def _get_gene_model(
        self, locus: str, seq_length: int,
    ) -> list[tuple[str, int, int]]:
        """Get gene model for a locus, or generate a generic one."""
        if locus in self.GENE_MODEL:
            return self.GENE_MODEL[locus]

        # Generic model based on typical HLA gene structure
        info = parse_allele_name(locus) if "*" in locus else None
        gene_class = "I" if locus in ("A", "B", "C", "E", "F", "G") else "II"

        if gene_class == "I":
            return [
                ("utr_5", 0, int(seq_length * 0.08)),
                ("exon_1", int(seq_length * 0.08), int(seq_length * 0.15)),
                ("intron_1", int(seq_length * 0.15), int(seq_length * 0.20)),
                ("exon_2", int(seq_length * 0.20), int(seq_length * 0.28)),
                ("intron_2", int(seq_length * 0.28), int(seq_length * 0.35)),
                ("exon_3", int(seq_length * 0.35), int(seq_length * 0.43)),
                ("intron_3", int(seq_length * 0.43), int(seq_length * 0.55)),
                ("exon_4", int(seq_length * 0.55), int(seq_length * 0.63)),
                ("intron_4", int(seq_length * 0.63), int(seq_length * 0.70)),
                ("exon_5", int(seq_length * 0.70), int(seq_length * 0.73)),
                ("utr_3", int(seq_length * 0.73), seq_length),
            ]
        else:
            return [
                ("utr_5", 0, int(seq_length * 0.05)),
                ("exon_1", int(seq_length * 0.05), int(seq_length * 0.10)),
                ("intron_1", int(seq_length * 0.10), int(seq_length * 0.20)),
                ("exon_2", int(seq_length * 0.20), int(seq_length * 0.30)),
                ("intron_2", int(seq_length * 0.30), int(seq_length * 0.60)),
                ("exon_3", int(seq_length * 0.60), int(seq_length * 0.67)),
                ("intron_3", int(seq_length * 0.67), int(seq_length * 0.80)),
                ("exon_4", int(seq_length * 0.80), int(seq_length * 0.85)),
                ("utr_3", int(seq_length * 0.85), seq_length),
            ]

    def _find_region(
        self, position: int, gene_model: list[tuple[str, int, int]],
    ) -> str:
        """Find which gene region a position falls in."""
        for region_name, start, end in gene_model:
            if start <= position < end:
                return region_name
        return "unknown"

    def _classify_snp(
        self,
        position: int,
        ref_base: str,
        alt_base: str,
        ref_seq: str,
        gene_model: list[tuple[str, int, int]],
        region: str,
    ) -> tuple[str, str, str]:
        """Classify a SNP as synonymous or non-synonymous."""
        # Find the exon and compute codon position
        exon_start = 0
        for region_name, start, end in gene_model:
            if region_name == region:
                exon_start = start
                break

        offset_in_exon = position - exon_start
        codon_pos = offset_in_exon % 3
        codon_start = position - codon_pos

        # Extract reference codon
        if codon_start + 3 <= len(ref_seq):
            ref_codon = ref_seq[codon_start:codon_start + 3].upper()
            alt_codon = list(ref_codon)
            alt_codon[codon_pos] = alt_base
            alt_codon = "".join(alt_codon)

            ref_aa = CODON_TABLE.get(ref_codon, "?")
            alt_aa = CODON_TABLE.get(alt_codon, "?")

            codon_change = f"{ref_codon}>{alt_codon}"
            if ref_aa == alt_aa:
                return codon_change, "synonymous", "synonymous"
            else:
                return codon_change, f"{ref_aa}>{alt_aa}", "non_synonymous"

        return "", "", "coding_unknown"

    def _make_hgvs(
        self, mismatch: MismatchAnnotation, allele: str,
    ) -> str:
        """Generate HGVS-like notation for a variant."""
        pos = mismatch.position + 1  # 1-based

        if mismatch.variant_type == "SNP":
            return f"{allele}:g.{pos}{mismatch.ref_base}>{mismatch.alt_base}"
        elif mismatch.variant_type == "INS":
            return f"{allele}:g.{pos}_{pos+1}ins{mismatch.alt_base}"
        elif mismatch.variant_type == "DEL":
            end = pos + len(mismatch.ref_base) - 1
            return f"{allele}:g.{pos}_{end}del"
        return f"{allele}:g.{pos}{mismatch.variant_type}"
