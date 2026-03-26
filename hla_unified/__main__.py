"""HLA-Unified V2 CLI: Multi-strategy HLA typing.

Usage:
    hla-unified type --bam input.bam --imgt-db /path/to/IMGTHLA --out ./results
    hla-unified type --bam input.bam --imgt-db /path --out ./results --profile transplant
    hla-unified type --r1 R1.fq.gz --r2 R2.fq.gz --imgt-db /path --out ./results
    hla-unified setup-db --out /path/to/IMGTHLA
    hla-unified benchmark --dataset truth.tsv --bam-dir ./bams --imgt-db /path --out ./bench
"""

from __future__ import annotations

import click

from .utils.log import setup_logging


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """HLA-Unified V2: Multi-strategy HLA typing with ambiguity classification."""
    pass


@cli.command()
@click.option("--bam", type=click.Path(exists=True), help="Input BAM/CRAM file")
@click.option("--r1", type=click.Path(exists=True), help="Input R1 FASTQ")
@click.option("--r2", type=click.Path(exists=True), help="Input R2 FASTQ")
@click.option("--imgt-db", required=True, type=click.Path(),
              help="Path to IMGT/HLA database directory")
@click.option("--out", "-o", required=True, type=click.Path(),
              help="Output directory")
@click.option("--threads", "-t", default=4, help="Number of threads")
@click.option("--reference", type=click.Choice(["GRCh37", "GRCh38", "hg19", "hg38"]),
              default=None, help="Reference build (auto-detected if omitted)")
@click.option("--loci", default=None,
              help="Comma-separated loci to type (default: all classical)")
@click.option("--skip-refinement", is_flag=True, help="Skip Phase 2 refinement")
@click.option("--skip-confidence", is_flag=True, help="Skip Phase 4 Bayesian confidence")
@click.option("--skip-kmer", is_flag=True, help="Skip Phase 5a k-mer validation")
@click.option("--skip-assembly", is_flag=True, help="Skip Phase 5b assembly fallback")
@click.option("--max-candidates", default=80,
              help="Max candidates per locus after pre-filter")
@click.option("--data-type",
              type=click.Choice(["short", "exome", "targeted_capture",
                                 "pacbio", "hifi", "ont", "rna"]),
              default="short",
              help="Assay preset: short (WGS), exome (WES), targeted_capture, "
                   "pacbio, hifi, ont, rna")
@click.option("--output-resolution", type=click.Choice(["1", "2", "3", "4", "G", "max"]),
              default="max",
              help="Output resolution: 1-4 field, G (G-group), or max (assay limit)")
@click.option("--imgt-release", default=None,
              help="Required IPD-IMGT/HLA release version (enforced if set)")
@click.option("--profile",
              type=click.Choice(["transplant", "research", "immuno_onc"]),
              default=None,
              help="Use-case profile: transplant (conservative), "
                   "research (max resolution), immuno_onc (fast Class I)")
@click.option("--clinical", is_flag=True,
              help="Generate clinical summary report")
@click.option("--strict-reproducibility", is_flag=True,
              help="Write IMGT lockfile + manifest for full reproducibility")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v, -vv)")
def type(bam, r1, r2, imgt_db, out, threads, reference, loci,
         skip_refinement, skip_confidence, skip_kmer, skip_assembly,
         max_candidates, data_type, output_resolution, imgt_release,
         profile, clinical, strict_reproducibility, verbose):
    """Run HLA typing on BAM or FASTQ input.

    Phases:
      0. Read extraction (MHC region + unmapped)
      1. Fast pre-filter (xHLA-style minimap2)
      2. Iterative refinement (HLA-HD-style bowtie2)
      2.5. Haplotype phasing (read-backed)
      3. ILP genotyping (OptiType-style, class I+II)
      4. Bayesian confidence (VBSeq-style posteriors)
      5. K-mer validation + assembly fallback
      6. Ambiguity classification + novel allele detection
    """
    setup_logging(verbose + 1)

    if not bam and not r1:
        raise click.UsageError("Provide --bam or --r1 (and optionally --r2)")

    from .config.schema import PipelineConfig

    loci_list = loci.split(",") if loci else None
    res = output_resolution if output_resolution in ("max", "G") else int(output_resolution)

    config = PipelineConfig.from_cli(
        imgt_db=imgt_db,
        out=out,
        threads=threads,
        loci=loci_list,
        data_type=data_type,
        output_resolution=res,
        imgt_release=imgt_release,
        strict_reproducibility=strict_reproducibility,
        skip_refinement=skip_refinement,
        skip_confidence=skip_confidence,
        skip_kmer=skip_kmer,
        skip_assembly=skip_assembly,
        profile_name=profile,
        clinical=clinical,
        max_candidates=max_candidates,
    )

    from .pipeline.runner import UnifiedPipeline

    pipeline = UnifiedPipeline(
        imgt_db_path=config.imgt_db_path,
        work_dir=config.work_dir,
        threads=config.threads,
        loci=config.effective_loci(),
        data_type=config.data_type,
        output_resolution=config.effective_resolution(),
        required_imgt_release=config.imgt_release,
        skip_refinement=config.skip_refinement,
        skip_confidence=config.skip_confidence,
        skip_kmer=config.skip_kmer,
        skip_assembly=config.skip_assembly,
    )

    if bam:
        result = pipeline.run(bam, input_type="bam", reference_build=reference)
    else:
        result = pipeline.run(r1, input_type="fastq", r2_path=r2)

    # === V2 Post-processing: Ambiguity + Novel + Clinical + Manifest ===
    from pathlib import Path

    # Ambiguity classification
    from .confidence.ambiguity_classifier import AmbiguityClassifier
    classifier = AmbiguityClassifier()
    ambiguity_results = classifier.classify_all_loci(
        loci=config.effective_loci(),
        vb_results=getattr(pipeline, '_last_vb_results', {}),
        kmer_results=getattr(pipeline, '_last_kmer_results', {}),
        phasing_results=getattr(pipeline, '_last_phasing_results', {}),
        ilp_results=getattr(pipeline, '_last_ilp_results', {}),
        data_type=config.data_type,
    )

    # Novel allele detection
    from .novel.detector import NovelAlleleDetector
    novel_detector = NovelAlleleDetector(
        threads=config.threads, data_type=config.data_type,
    )
    novel_candidates = novel_detector.screen_all_loci(
        loci=config.effective_loci(),
        ilp_results=getattr(pipeline, '_last_ilp_results', {}),
        kmer_results=getattr(pipeline, '_last_kmer_results', {}),
        assembly_results=getattr(pipeline, '_last_assembly_results', {}),
    )

    # Novel allele annotation (variant-level detail for candidates)
    novel_annotations = {}
    if novel_candidates:
        from .novel.annotator import NovelAlleleAnnotator
        from .reference.imgt_db import IMGTDatabase as _DB
        annotator = NovelAlleleAnnotator()
        _db = _DB(config.imgt_db_path)
        for locus, candidate in novel_candidates.items():
            if candidate.novel_report and candidate.closest_known_allele:
                ref_seq = ""
                seqs = _db.load_genomic(locus)
                ref_seq = seqs.get(candidate.closest_known_allele, "")
                if ref_seq:
                    novel_annotations[locus] = annotator.annotate(
                        locus=locus,
                        closest_allele=candidate.closest_known_allele,
                        ref_seq=ref_seq,
                        novel_report=candidate.novel_report,
                    )

    # Detailed locus metrics from pipeline
    locus_metrics = getattr(pipeline, '_last_locus_metrics', {})

    # Write V2 outputs
    from .output.writer import OutputWriter
    writer = OutputWriter(
        out_dir=Path(out),
        data_type=config.data_type,
        output_resolution=config.effective_resolution(),
    )

    # Manifest
    manifest = None
    if config.strict_reproducibility:
        from .config.manifest import generate_manifest
        from .reference.imgt_db import IMGTDatabase
        db = IMGTDatabase(config.imgt_db_path)
        manifest = generate_manifest(db.provenance)
        db.write_lockfile(Path(out))

    writer.write_all(
        result,
        ambiguity=ambiguity_results,
        locus_metrics=locus_metrics,
        manifest=manifest,
    )

    # Novel allele annotation output
    if novel_annotations:
        import json as _json
        novel_out = Path(out) / "novel_alleles.json"
        novel_data = {
            locus: ann.to_dict() for locus, ann in novel_annotations.items()
        }
        novel_out.write_text(_json.dumps(novel_data, indent=2))

    # Clinical summary
    if config.clinical_summary:
        from .output.clinical import ClinicalReporter
        reporter = ClinicalReporter()
        summary = reporter.generate(result, ambiguity_results, config.data_type)
        reporter.write_clinical_json(summary, Path(out))
        reporter.write_clinical_text(summary, Path(out))

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo(f"HLA-Unified V2 Results ({result.runtime_seconds:.1f}s)")
    click.echo(f"{'='*60}")
    click.echo(f"{'Locus':<12} {'Allele 1':<20} {'Allele 2':<20} {'Conf':<8} {'Ambiguity'}")
    click.echo(f"{'-'*12} {'-'*20} {'-'*20} {'-'*8} {'-'*15}")

    for locus in sorted(result.calls.keys()):
        call = result.calls[locus]
        ambig = ambiguity_results.get(locus)
        ambig_str = ambig.primary_reason.value if ambig else ""
        click.echo(
            f"HLA-{locus:<8} {call.allele1:<20} {call.allele2:<20} "
            f"{call.confidence:<8} {ambig_str}"
        )

    if novel_candidates:
        click.echo(f"\nNovel allele candidates:")
        for locus, nc in novel_candidates.items():
            click.echo(f"  HLA-{locus}: {nc.detection_reason} (confidence={nc.confidence})")

    flags_any = any(c.flags for c in result.calls.values())
    if flags_any:
        click.echo(f"\nFlags:")
        for locus, call in result.calls.items():
            if call.flags:
                click.echo(f"  HLA-{locus}: {', '.join(call.flags)}")

    click.echo(f"\nOutputs: {out}/")
    click.echo(f"  hla_types.tsv     - Main results")
    click.echo(f"  hla_types.json    - Full JSON with evidence trail")
    click.echo(f"  ambiguity.tsv     - Ranked alternatives")
    if config.clinical_summary:
        click.echo(f"  clinical_summary.txt/json - Clinical report")
    if config.strict_reproducibility:
        click.echo(f"  manifest.json     - Reproducibility manifest")
        click.echo(f"  imgt_lock.json    - IMGT version lock")


@cli.command("setup-db")
@click.option("--out", "-o", required=True, type=click.Path(),
              help="Output directory for IMGT/HLA database")
@click.option("--release", default="Latest", help="IMGT/HLA release tag")
def setup_db(out, release):
    """Download and set up the IMGT/HLA reference database."""
    setup_logging(2)

    from .reference.imgt_db import IMGTDatabase

    db = IMGTDatabase(out)
    db.setup(release=release)
    click.echo(f"IMGT/HLA database ready at: {out}")
    click.echo(f"Release: {db.release_version}")


@cli.command("build-panel")
@click.option("--imgt-db", required=True, type=click.Path(exists=True),
              help="Path to IMGT/HLA database")
@click.option("--out", "-o", required=True, type=click.Path(),
              help="Output directory for reduced panels")
@click.option("--loci", default=None,
              help="Comma-separated loci (default: all classical)")
def build_panel(imgt_db, out, loci):
    """Build reduced (non-redundant) reference panels for fast pre-filtering."""
    setup_logging(2)
    from pathlib import Path
    from .reference.imgt_db import IMGTDatabase
    from .reference.loci import ALL_TYPING_LOCI

    db = IMGTDatabase(imgt_db)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loci_list = loci.split(",") if loci else list(ALL_TYPING_LOCI)
    for locus in loci_list:
        n = db.build_reduced_panel(locus, out_dir / f"{locus}_reduced.fa")
        click.echo(f"  {locus}: {n} representative alleles")

    click.echo(f"Panels written to: {out}")


@cli.command("validate")
@click.argument("results_tsv", type=click.Path(exists=True))
@click.argument("truth_tsv", type=click.Path(exists=True))
@click.option("--resolution", "-r", default=2, help="Comparison resolution (fields)")
def validate(results_tsv, truth_tsv, resolution):
    """Compare typing results against a truth set."""
    setup_logging(1)
    import csv

    # Load results
    results = {}
    with open(results_tsv) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            break
    with open(results_tsv) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    import io
    for row in csv.DictReader(io.StringIO("".join(lines)), delimiter="\t"):
        locus = row["Locus"].replace("HLA-", "")
        chrom = int(row["Chromosome"])
        results.setdefault(locus, {})[chrom] = row["Allele"]

    # Load truth
    truth = {}
    with open(truth_tsv) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            locus = row["Locus"].replace("HLA-", "")
            chrom = int(row["Chromosome"])
            truth.setdefault(locus, {})[chrom] = row["Allele"]

    # Compare using V2 metrics
    from .benchmark.metrics import compare_diploid
    total = 0
    correct = 0
    for locus in sorted(set(results) | set(truth)):
        r = results.get(locus, {})
        t = truth.get(locus, {})
        call = (r.get(1, ""), r.get(2, ""))
        true = (t.get(1, ""), t.get(2, ""))

        match = compare_diploid(call, true, resolution)
        total += 2
        if match == 2:
            correct += 2
            status = "OK"
        elif match == 1:
            correct += 1
            status = "PARTIAL"
        elif match == -1:
            status = "NO_CALL"
        else:
            status = "MISMATCH"

        click.echo(
            f"  HLA-{locus}: {status}  "
            f"call=({call[0]}, {call[1]})  truth=({true[0]}, {true[1]})"
        )

    acc = correct / total * 100 if total else 0
    click.echo(f"\nAccuracy: {correct}/{total} alleles ({acc:.1f}%)")


@cli.command("benchmark")
@click.option("--dataset", required=True, type=click.Path(exists=True),
              help="Truth TSV file (SampleID, Locus, Allele1, Allele2)")
@click.option("--bam-dir", required=True, type=click.Path(exists=True),
              help="Directory containing BAM files (named {SampleID}.bam)")
@click.option("--imgt-db", required=True, type=click.Path(exists=True),
              help="Path to IMGT/HLA database")
@click.option("--out", "-o", required=True, type=click.Path(),
              help="Output directory for benchmark results")
@click.option("--threads", "-t", default=4, help="Number of threads")
@click.option("--data-type", default="short", help="Assay type")
@click.option("--resolution", "-r", default=2, help="Comparison resolution")
@click.option("--skip-typing", is_flag=True,
              help="Skip typing, evaluate existing results only")
@click.option("--results-dir", type=click.Path(),
              help="Directory with pre-computed results (for --skip-typing)")
def benchmark(dataset, bam_dir, imgt_db, out, threads, data_type,
              resolution, skip_typing, results_dir):
    """Run benchmark evaluation on a dataset with known truth."""
    setup_logging(2)
    from pathlib import Path
    from .benchmark.datasets import BenchmarkDataset
    from .benchmark.runner import BenchmarkRunner

    ds = BenchmarkDataset.from_tsv(
        name=Path(dataset).stem,
        truth_tsv=dataset,
        bam_dir=bam_dir,
        assay=data_type,
        resolution=resolution,
    )

    click.echo(f"Dataset: {ds.name} ({ds.n_samples} samples, {len(ds.loci)} loci)")

    runner = BenchmarkRunner(
        imgt_db_path=imgt_db,
        work_dir=out,
        threads=threads,
        data_type=data_type,
    )

    report = runner.run_dataset(
        ds, resolution=resolution,
        skip_typing=skip_typing,
        results_dir=Path(results_dir) if results_dir else None,
    )

    runner.write_report(report, Path(out))

    click.echo(f"\nOverall accuracy: {report.overall_accuracy:.1%}")
    click.echo(f"Call rate: {report.overall_call_rate:.1%}")
    click.echo(f"\nPer-locus:")
    for locus, acc in sorted(report.per_locus.items()):
        click.echo(f"  HLA-{locus}: {acc.accuracy:.1%} ({acc.n_samples} samples)")


if __name__ == "__main__":
    cli()
