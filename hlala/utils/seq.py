"""Sequence utilities: reverse complement, Phred conversion, k-mer partitioning."""

from __future__ import annotations

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def phred_to_p_correct(quality_char: str) -> float:
    """Convert a single Phred quality character to P(correct).

    Phred quality Q is encoded as chr(Q + 33). P(correct) = 1 - 10^(-Q/10).
    """
    q = ord(quality_char) - 33
    if q <= 0:
        return 0.25
    return 1.0 - 10.0 ** (-q / 10.0)


def p_correct_to_phred(p_correct: float) -> int:
    """Convert P(correct) to integer Phred score."""
    import math
    if p_correct >= 1.0:
        return 60
    if p_correct <= 0.0:
        return 0
    return min(60, max(0, round(-10.0 * math.log10(1.0 - p_correct))))


def partition_into_kmers(sequence: str, k: int) -> list[str]:
    """Split a sequence into overlapping k-mers."""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def kmer_canonical(kmer: str) -> str:
    """Return the canonical (lexicographically smaller) k-mer representation."""
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def remove_gaps(seq: str) -> str:
    """Remove gap characters ('_' and '-') from a sequence."""
    return seq.replace("_", "").replace("-", "")


def proportion_n(seq: str) -> float:
    """Return the proportion of N characters in a sequence."""
    if not seq:
        return 0.0
    return seq.upper().count("N") / len(seq)


def sequence_all_ns(seq: str) -> bool:
    """Check if a sequence is entirely N characters."""
    return all(c in "Nn" for c in seq)
