"""File I/O helpers."""

from __future__ import annotations

import os
from pathlib import Path


def read_fasta(filepath: str | Path, full_identifier: bool = False) -> dict[str, str]:
    """Read a FASTA file and return a dict of {header: sequence}.

    If full_identifier is False, only the first whitespace-delimited token
    of the header line is used as the key.
    """
    sequences: dict[str, str] = {}
    current_id = ""
    parts: list[str] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(parts)
                header = line[1:]
                current_id = header if full_identifier else header.split()[0]
                parts = []
            else:
                parts.append(line)
    if current_id:
        sequences[current_id] = "".join(parts)
    return sequences


def write_fasta(filepath: str | Path, sequences: dict[str, str], line_width: int = 80) -> None:
    """Write sequences to a FASTA file."""
    with open(filepath, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i:i + line_width] + "\n")


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_all_lines(filepath: str | Path) -> list[str]:
    """Read all lines from a file, stripping newlines."""
    with open(filepath) as fh:
        return [line.rstrip("\n\r") for line in fh]


def get_first_line(filepath: str | Path) -> str:
    """Read and return the first line of a file."""
    with open(filepath) as fh:
        return fh.readline().rstrip("\n\r")
