"""CLI: Translate HLA types to G-group notation."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("translate-g-groups")
@click.option("--input", "input_file", required=True, type=click.Path(exists=True),
              help="HLA types file (from align-and-type)")
@click.option("--graph-dir", required=True, type=click.Path(exists=True),
              help="Graph directory")
@click.option("--output", "-o", required=True, help="Output file path")
def translate_g_groups(input_file: str, graph_dir: str, output: str) -> None:
    """Translate HLA type calls to G-group designations.

    Reads the hla_types.txt from align-and-type and applies G-group
    translation using IMGT nomenclature.
    """
    from ..hla.g_groups import GGroupTranslator

    graph_dir_path = Path(graph_dir)
    g_file = graph_dir_path / "hla_nom_g.txt"
    if not g_file.exists():
        g_file = graph_dir_path.parent / "hla_nom_g.txt"
    if not g_file.exists():
        click.echo("Warning: hla_nom_g.txt not found, copying input as-is", err=True)
        import shutil
        shutil.copy(input_file, output)
        return

    translator = GGroupTranslator()
    translator.load(g_file)

    with open(input_file) as fin, open(output, "w") as fout:
        header = fin.readline()
        fout.write(header)

        for line in fin:
            fields = line.strip().split("\t")
            if len(fields) >= 3:
                allele = fields[2]
                g_allele, perfect = translator.translate(allele)
                fields[2] = g_allele
                if len(fields) > 11:
                    fields[11] = "1" if perfect else "0"
                fout.write("\t".join(fields) + "\n")
            else:
                fout.write(line)

    click.echo(f"G-group translated results written to {output}")
