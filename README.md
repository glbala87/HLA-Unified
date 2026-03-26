# HLA-Unified V2

Multi-strategy HLA typing with ambiguity classification, novel allele detection, and clinical reporting.

## Quick Start

```bash
# Install
pip install -e .

# Set up IMGT/HLA database
hla-unified setup-db --out ./IMGTHLA

# Type a sample
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results
```

## Pipeline Phases

| Phase | Method | Output |
|-------|--------|--------|
| 0 | Read extraction (MHC region + unmapped) | FASTQ |
| 1 | Fast pre-filter (xHLA-style minimap2) | ~80 candidates/locus |
| 2 | Iterative refinement (HLA-HD-style bowtie2) | ~20 candidates/locus |
| 2.5 | Haplotype phasing (read-backed, long-read, spectral) | Phased bins |
| 3 | ILP genotyping (OptiType-style) + DRB3/4/5 CNV | Diploid pair/locus |
| 3.5 | Contamination screening + allele dropout detection | QC flags |
| 4 | Bayesian confidence (VBSeq-style + population priors) | Posteriors |
| 5 | K-mer validation + assembly fallback | Concordance flags |
| 6 | Ambiguity classification + novel allele detection | Evidence report |

## Assay Presets

| Flag | Description | Resolution |
|------|-------------|------------|
| `--data-type short` | Paired-end WGS (default) | 3-field |
| `--data-type exome` | Whole exome sequencing | 2-field |
| `--data-type targeted_capture` | HLA capture panel | 4-field |
| `--data-type pacbio` | PacBio CLR | 4-field |
| `--data-type hifi` | PacBio HiFi | 4-field |
| `--data-type ont` | Oxford Nanopore | 3-field |
| `--data-type rna` | RNA-seq | 2-field |

## Use-Case Profiles

```bash
# Transplant: conservative, clinical report, strict reproducibility
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results --profile transplant

# Research: all loci, max resolution, novel allele discovery
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results --profile research

# Immuno-oncology: fast Class I for neoantigen prediction
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results --profile immuno_onc
```

## Output Files

| File | Description |
|------|-------------|
| `hla_types.tsv` | Main results with confidence, ambiguity reason, and flags |
| `hla_types.json` | Full JSON with evidence trail and ranked alternatives |
| `ambiguity.tsv` | Ranked alternative diploid pairs per locus |
| `qc_report.json` | Structured QC: haplotype balance, k-mer, phasing, assembly |
| `qc_dashboard.html` | Visual HTML dashboard (standalone, no CDN) |
| `clinical_summary.txt` | Human-readable clinical report (`--clinical`) |
| `clinical_summary.json` | Machine-readable clinical report (`--clinical`) |
| `novel_alleles.json` | Annotated novel allele variants with HGVS notation |
| `manifest.json` | Full environment snapshot (`--strict-reproducibility`) |
| `imgt_lock.json` | IMGT DB version lock with SHA256 (`--strict-reproducibility`) |

## Ambiguity Classification

Every call includes an ambiguity reason explaining *why* it may be uncertain:

| Reason | Meaning | Suggestion |
|--------|---------|------------|
| `unambiguous` | Clear winner | — |
| `coverage_gap` | Missing reads in key region | Increase depth |
| `phase_break` | Het sites detected but reads don't span them | Use long reads |
| `close_alleles` | Top pairs differ by few SNPs | Higher coverage |
| `exon_only` | Alleles differ only in introns | Full-gene sequencing |
| `low_depth` | Insufficient reads at locus | Increase depth |
| `homo_ambig` | Can't distinguish homozygous from close het | Family data |

## DRB3/4/5 Copy Number

These loci have variable copy number (0-2 copies). The tool automatically:
- Estimates copy number from depth ratio vs DRB1
- Adjusts genotyping: 0 copies = no call, 1 copy = hemizygous, 2 = diploid
- Flags CNV status in output

## Benchmarking

```bash
# Run benchmark with truth set
hla-unified benchmark \
  --dataset truth.tsv --bam-dir ./bams/ \
  --imgt-db ./IMGTHLA --out ./bench

# Validate a single result
hla-unified validate results/hla_types.tsv truth.tsv --resolution 2
```

## Nextflow (Batch Mode)

```bash
# Single sample
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --bam sample.bam --imgt_db ./IMGTHLA --outdir ./results

# Batch: provide a sample sheet CSV (sample_id,bam_path,data_type)
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --sample_sheet samples.csv --imgt_db ./IMGTHLA --outdir ./results

# With Docker
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --bam sample.bam --imgt_db ./IMGTHLA -with-docker hla-unified:2.0.0
```

## Docker

```bash
docker build -f Dockerfile.unified -t hla-unified:2.0.0 .
docker run -v /data:/data hla-unified:2.0.0 type \
  --bam /data/sample.bam --imgt-db /data/IMGTHLA --out /data/results
```

## Dependencies

**Python**: >=3.10 with numpy, scipy, pysam, click, PuLP, biopython, msgpack

**External tools**: samtools, minimap2, bowtie2 (short reads), megahit (assembly)

Install via: `conda install -c bioconda samtools minimap2 bowtie2 megahit`

## Citation

If you use HLA-Unified, please cite:
- HLA*LA: Dilthey et al., Bioinformatics 2019
- OptiType: Szolek et al., Bioinformatics 2014
- HLA-VBSeq: Nariai et al., BMC Genomics 2015
