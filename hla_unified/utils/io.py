"""File I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) tuples from a FASTA file."""
    name = ""
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper())
    if name:
        yield name, "".join(parts)


def write_fasta(path: str | Path, sequences: dict[str, str],
                line_width: int = 80) -> None:
    with open(path, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i:i + line_width] + "\n")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
