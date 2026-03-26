"""Sequence utilities: reverse complement, k-mers, quality scores."""

from __future__ import annotations

import math

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def phred_char_to_prob(ch: str) -> float:
    """Convert a single Phred+33 ASCII character to error probability."""
    q = ord(ch) - 33
    return 10.0 ** (-q / 10.0)


def phred_char_to_p_correct(ch: str) -> float:
    return 1.0 - phred_char_to_prob(ch)


def canonical_kmer(kmer: str) -> str:
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def extract_kmers(seq: str, k: int = 31) -> list[str]:
    """Extract all k-mers from a sequence, skipping those with N."""
    kmers = []
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if "N" not in km and "n" not in km:
            kmers.append(km)
    return kmers


def extract_canonical_kmers(seq: str, k: int = 31) -> set[str]:
    return {canonical_kmer(km) for km in extract_kmers(seq, k)}
