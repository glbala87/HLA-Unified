"""CLI: Main alignment and HLA typing orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("align-and-type")
@click.option("--bam", required=True, type=click.Path(exists=True),
              help="Remapped BAM (against PRG reference)")
@click.option("--graph-dir", required=True, type=click.Path(exists=True),
              help="Graph directory")
@click.option("--sample-id", required=True, help="Sample identifier")
@click.option("--output-dir", "-o", required=True, help="Output directory")
@click.option("--threads", default=1, type=int, help="Number of threads")
@click.option("--long-reads", type=click.Choice(["", "ont2d", "pacbio"]),
              default="", help="Long read technology")
def align_and_type(bam: str, graph_dir: str, sample_id: str, output_dir: str,
                   threads: int, long_reads: str) -> None:
    """Perform graph alignment and HLA type inference.

    This is the main computational step:
    1. Load the population reference graph
    2. Extract seed chains from remapped BAM
    3. Extend seeds through the graph using NW alignment
    4. Infer HLA types from extended alignments
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    graph_dir_path = Path(graph_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load graph
    click.echo(f"Loading graph from {graph_dir}...")
    from ..graph.serialization import load_graph

    msgpack_path = graph_dir_path / "serializedGRAPH.msgpack"
    if msgpack_path.exists():
        graph = load_graph(msgpack_path)
    else:
        click.echo("No msgpack graph found. Run 'hlala prepare-graph' first.", err=True)
        raise SystemExit(1)

    # Process BAM - extract seeds
    click.echo("Processing BAM file...")
    from ..mapper.process_bam import BAMProcessor
    processor = BAMProcessor(graph_dir)

    is_paired = not bool(long_reads)
    paired_reads, paired_chains, unpaired_reads, unpaired_chains = \
        processor.extract_seeds_from_bam(bam, is_paired=is_paired)

    # Estimate insert size for paired reads
    insert_mean, insert_sd = 300.0, 50.0
    if is_paired and paired_reads:
        insert_mean, insert_sd = processor.estimate_insert_size(bam)
        click.echo(f"Insert size: mean={insert_mean:.1f}, sd={insert_sd:.1f}")

    # Extend seed chains
    click.echo("Extending seed chains...")
    from ..aligner.extension_aligner import ExtensionAligner
    aligner = ExtensionAligner(graph)

    for i, (chain_pair, read_pair) in enumerate(zip(paired_chains, paired_reads)):
        chain_pair.chain1 = aligner.extend_seed_chain(
            read_pair.read1.sequence, chain_pair.chain1)
        chain_pair.chain2 = aligner.extend_seed_chain(
            read_pair.read2.sequence, chain_pair.chain2)
        if (i + 1) % 1000 == 0:
            logger.info(f"Extended {i + 1}/{len(paired_chains)} paired chains")

    for i, (chain, read) in enumerate(zip(unpaired_chains, unpaired_reads)):
        unpaired_chains[i] = aligner.extend_seed_chain(read.sequence, chain)
        if (i + 1) % 1000 == 0:
            logger.info(f"Extended {i + 1}/{len(unpaired_chains)} unpaired chains")

    # HLA typing
    click.echo("Inferring HLA types...")
    from ..hla.typer import HLATyper
    typer = HLATyper(graph, graph_dir)

    results = typer.type_hla(
        paired_reads=paired_reads,
        paired_alignments=paired_chains,
        unpaired_reads=unpaired_reads,
        unpaired_alignments=unpaired_chains,
        insert_size_mean=insert_mean,
        insert_size_sd=insert_sd,
        output_dir=output_path,
        long_reads_mode=long_reads,
    )

    click.echo(f"Typing complete. Results in {output_path / 'hla_types.txt'}")
