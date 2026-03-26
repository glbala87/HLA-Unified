"""Interval tree wrapper for genomic range queries."""

from __future__ import annotations

from intervaltree import IntervalTree, Interval


def build_interval_tree(intervals: list[tuple[int, int, str]]) -> IntervalTree:
    """Build an IntervalTree from a list of (start, end, data) tuples.

    Note: IntervalTree uses half-open intervals [start, end).
    """
    tree = IntervalTree()
    for start, end, data in intervals:
        tree.addi(start, end + 1, data)  # +1 because original uses closed intervals
    return tree


def query_overlapping(tree: IntervalTree, start: int, end: int) -> set[str]:
    """Return data values for all intervals overlapping [start, end]."""
    results = tree.overlap(start, end + 1)
    return {iv.data for iv in results}


def intervals_overlap(x1: int, x2: int, y1: int, y2: int) -> bool:
    """Check whether two closed intervals [x1,x2] and [y1,y2] overlap."""
    return x1 <= y2 and y1 <= x2
