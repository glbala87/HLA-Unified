"""CLI dispatcher for HLA-LA.

Provides the main 'hlala' command with subcommands for each pipeline step.
"""

from __future__ import annotations

import click

from .cli.detect_reference import detect_reference
from .cli.extract_reads import extract_reads
from .cli.bam_to_fastq import bam_to_fastq
from .cli.align_and_type import align_and_type
from .cli.translate_g_groups import translate_g_groups


@click.group()
@click.version_option()
def cli() -> None:
    """HLA-LA: HLA typing from NGS data using a population reference graph."""
    pass


cli.add_command(detect_reference)
cli.add_command(extract_reads)
cli.add_command(bam_to_fastq)
cli.add_command(align_and_type)
cli.add_command(translate_g_groups)


@cli.command("prepare-graph")
@click.option("--graph-dir", required=True, type=click.Path(exists=True),
              help="Graph directory to prepare")
@click.option("--kmer-size", default=25, type=int, help="K-mer size for index")
def prepare_graph(graph_dir: str, kmer_size: int) -> None:
    """Prepare and index a population reference graph.

    Reads graph.txt and segment files from the graph directory,
    builds the PRG graph, creates the k-mer index, and serializes
    to msgpack format for fast loading.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from .graph.graph import PRGGraph
    from .graph.graph_and_edge_index import GraphAndEdgeIndex
    from .graph.serialization import save_graph

    graph_dir_path = Path(graph_dir)
    click.echo(f"Preparing graph in {graph_dir_path}...")

    # Build graph from text files
    graph = PRGGraph()
    graph_file = graph_dir_path / "graph.txt"
    if graph_file.exists():
        click.echo("Reading graph.txt...")
        _read_graph_from_text(graph, graph_file)
    else:
        click.echo("ERROR: graph.txt not found in graph directory", err=True)
        raise SystemExit(1)

    click.echo(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {graph.num_levels} levels")

    # Build k-mer index
    click.echo(f"Building k-mer index (k={kmer_size})...")
    index = GraphAndEdgeIndex(graph, k=kmer_size)
    index.index()
    click.echo(f"Indexed {len(index.kmers)} unique k-mers")

    # Save
    output = graph_dir_path / "serializedGRAPH.msgpack"
    save_graph(graph, output)
    click.echo(f"Graph saved to {output}")


def _read_graph_from_text(graph, filepath) -> None:
    """Read a graph from the text format (graph.txt).

    Format: tab-separated lines, each defining an edge:
    level_from    level_to    emission    locus_id    count
    """
    from pathlib import Path

    node_cache: dict[tuple[int, int], int] = {}  # (level, idx) -> node_id

    def get_or_create_node(level: int, idx: int = 0) -> int:
        key = (level, idx)
        if key not in node_cache:
            node = graph.register_node(level=level)
            node_cache[key] = node.id
        return node_cache[key]

    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue

            level_from = int(fields[0])
            level_to = int(fields[1])
            emission = fields[2] if len(fields) > 2 else "_"
            locus_id = fields[3] if len(fields) > 3 else ""
            count = float(fields[4]) if len(fields) > 4 else 1.0

            from_node = get_or_create_node(level_from)
            to_node = get_or_create_node(level_to)

            graph.register_edge(
                from_node=from_node,
                to_node=to_node,
                emission=emission,
                locus_id=locus_id,
                count=count,
            )


if __name__ == "__main__":
    cli()
