"""CLI: Convert extracted BAM to FASTQ."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("bam-to-fastq")
@click.option("--bam", required=True, type=click.Path(exists=True),
              help="Input BAM file (from extract-reads)")
@click.option("--output-dir", "-o", required=True, help="Output directory for FASTQs")
@click.option("--long-reads", type=click.Choice(["", "ont2d", "pacbio"]),
              default="", help="Long read technology")
@click.option("--max-long-read-length", default=50000, type=int,
              help="Maximum long read length before splitting")
def bam_to_fastq(bam: str, output_dir: str, long_reads: str,
                 max_long_read_length: int) -> None:
    """Convert extracted BAM to FASTQ files.

    For paired-end data: produces R_1.fastq, R_2.fastq, R_U.fastq
    For long reads: produces R_U.fastq with optional splitting of very long reads.
    """
    import pysam

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    r1_path = out / "R_1.fastq"
    r2_path = out / "R_2.fastq"
    ru_path = out / "R_U.fastq"

    paired: dict[str, dict[int, tuple[str, str, str]]] = {}
    unpaired: list[tuple[str, str, str]] = []

    with pysam.AlignmentFile(str(bam), "rb", check_sq=False) as bam_file:
        for read in bam_file.fetch(until_eof=True):
            name = read.query_name
            seq = read.query_sequence or ""
            qual = read.qual or ""
            if not seq:
                continue

            if read.is_paired:
                if name not in paired:
                    paired[name] = {}
                read_num = 1 if read.is_read1 else 2
                paired[name][read_num] = (name, seq, qual)
            else:
                unpaired.append((name, seq, qual))

    with open(r1_path, "w") as f1, open(r2_path, "w") as f2:
        for name, reads in paired.items():
            if 1 in reads and 2 in reads:
                n1, s1, q1 = reads[1]
                f1.write(f"@{n1}/1\n{s1}\n+\n{q1}\n")
                n2, s2, q2 = reads[2]
                f2.write(f"@{n2}/2\n{s2}\n+\n{q2}\n")
            else:
                for _, (n, s, q) in reads.items():
                    unpaired.append((n, s, q))

    with open(ru_path, "w") as fu:
        for name, seq, qual in unpaired:
            if long_reads and len(seq) > max_long_read_length:
                # Split long reads into chunks
                chunk_idx = 0
                while seq:
                    chunk_len = min(max_long_read_length, len(seq))
                    fu.write(f"@rP{chunk_idx}_{name}\n")
                    fu.write(f"{seq[:chunk_len]}\n+\n{qual[:chunk_len]}\n")
                    seq = seq[chunk_len:]
                    qual = qual[chunk_len:]
                    chunk_idx += 1
            else:
                fu.write(f"@{name}\n{seq}\n+\n{qual}\n")

    click.echo(f"FASTQ files written to {out}")
