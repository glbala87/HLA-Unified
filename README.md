# HLA-Unified V2

Multi-strategy HLA typing with ambiguity classification, novel allele detection, and clinical reporting.

**Version**: 2.1.0
**Status**: Research beta — validated on synthetic and real GIAB BAMs
**Accuracy**: 95% on synthetic BAMs (5 samples, 4 ancestries), 81% on real NA12878 300x BAM

## Requirements

**Python**: >= 3.10

**External tools** (must be on PATH):
- samtools >= 1.16
- minimap2 >= 2.24
- bowtie2 >= 2.5

**Optional**:
- megahit (for assembly fallback phase)

### Install external tools

```bash
# macOS (Homebrew)
brew install samtools minimap2 bowtie2

# Linux (conda)
conda install -c bioconda samtools minimap2 bowtie2

# Linux (apt)
sudo apt-get install samtools minimap2 bowtie2
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ANHIG/HLA-LA.git
cd HLA-LA

# Install Python package
pip install -e .

# Verify installation
hla-unified --version
# HLA-Unified 2.1.0

# Set up IMGT/HLA database (~500MB download)
hla-unified setup-db --out ./IMGTHLA
```

### Docker

```bash
# Build image
docker build -f Dockerfile.unified -t hla-unified:2.1.0 .

# Run
docker run -v /data:/data hla-unified:2.1.0 type \
  --bam /data/sample.bam --imgt-db /data/IMGTHLA --out /data/results
```

## Quick Start

```bash
# Type a sample from BAM
hla-unified type \
  --bam sample.bam \
  --imgt-db ./IMGTHLA \
  --out ./results \
  --threads 8 \
  --data-type short

# Type from FASTQ (paired-end)
hla-unified type \
  --r1 reads_R1.fastq.gz \
  --r2 reads_R2.fastq.gz \
  --imgt-db ./IMGTHLA \
  --out ./results

# View results
cat ./results/hla_types.tsv
```

## Usage Examples

### Basic HLA typing (WGS BAM)
```bash
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results
```

### With clinical profile (transplant)
```bash
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --profile transplant --clinical
```

### Exome sequencing
```bash
hla-unified type --bam exome.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type exome
```

### Long reads (PacBio HiFi)
```bash
hla-unified type --bam hifi.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type hifi
```

### Skip assembly (faster, no megahit required)
```bash
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --skip-assembly
```

## Validation on Real Data

### Download and type a GIAB reference sample
```bash
# Download NA12878 MHC region (~900MB, ~5 min)
mkdir -p validation/real_data/bams
samtools view -b -h -o validation/real_data/bams/NA12878.bam \
  'https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam' \
  chr6:28510120-33480577
samtools index validation/real_data/bams/NA12878.bam

# Run typing (~20 min for 300x)
hla-unified type \
  --bam validation/real_data/bams/NA12878.bam \
  --imgt-db ./IMGTHLA \
  --out validation/real_data/results/NA12878 \
  --threads 4 --data-type short --skip-assembly
```

### Automated multi-sample validation with truth comparison
```bash
python validation/real_data/run_real_validation.py \
  --samples NA12878,HG002,HG003,HG004 \
  --imgt-db ./IMGTHLA \
  --out validation/real_data/results \
  --region-only --threads 4
```

### Run synthetic BAM end-to-end test (no download needed)
```bash
python validation/real_data/run_synthetic_bam_test.py \
  --imgt-db ./IMGTHLA \
  --out validation/real_data/synth_results \
  --threads 4
```

## Pipeline Phases

| Phase | Method | Output |
|-------|--------|--------|
| 0 | Read extraction (MHC region + unmapped) | FASTQ |
| 1 | Fast pre-filter (minimap2, 150 candidates/locus) | Candidate alleles |
| 2 | Iterative refinement (bowtie2, 10 allele groups) | Refined candidates |
| 2.5 | Haplotype phasing (read-backed) | Phased bins |
| 3 | ILP genotyping (OptiType-style) + DRB3/4/5 CNV | Diploid pair/locus |
| 3.5 | K-mer genotyping (depth-aware, signal-explained scoring) | Primary calls |
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

## Benchmarking

```bash
# Run reference benchmark (10 samples, no BAMs needed)
python validation/run_validation.py

# Run synthetic BAM E2E test (5 samples, 4 ancestries)
python validation/real_data/run_synthetic_bam_test.py \
  --imgt-db ./IMGTHLA --out ./synth_results

# Run real-data validation (requires internet for GIAB download)
python validation/real_data/run_real_validation.py \
  --samples NA12878 --imgt-db ./IMGTHLA --out ./results --region-only
```

## Current Accuracy

| Test | Samples | Accuracy | Notes |
|------|---------|----------|-------|
| Reference benchmark | 10 (EUR/AFR/EAS/AMR) | 98.8% | Simulated results |
| Synthetic BAM E2E | 5 (EUR/AFR/EAS/AMR) | 95.0% | Reads from real IMGT alleles |
| Real BAM (NA12878) | 1 (EUR, 300x) | 81.2% | GIAB HiSeq 300x |
| Real BAM (HG002) | 1 (EUR/AJ, 300x) | 37.5% | Class II problematic |

## Known Limitations

- Class II accuracy on high-coverage (>100x) real BAMs is reduced due to multi-mapping noise from non-HLA MHC genes
- Very close alleles (differing by 1-2 SNPs, e.g., DRB1*04:01 vs 04:07) may be confused
- B*56:01 and other rare alleles require unrestricted k-mer search (slower)
- Assembly fallback requires megahit installation

## Dependencies

**Python packages** (installed automatically):
- pysam >= 0.21
- numpy >= 1.24
- scipy >= 1.10
- click >= 8.1
- PuLP >= 2.7
- biopython >= 1.81
- msgpack >= 1.0

**Development**:
```bash
pip install -e ".[dev]"      # pytest, ruff
pip install -e ".[benchmark]" # matplotlib, pandas
```

## Running Tests

```bash
# All tests (113 total)
pytest tests/

# Unit tests only
pytest tests/unit/

# Regression accuracy gates
pytest tests/regression/ -m regression

# Full validation
python validation/run_validation.py
```

## Citation

If you use HLA-Unified, please cite:
- HLA*LA: Dilthey et al., Bioinformatics 2019
- OptiType: Szolek et al., Bioinformatics 2014
- HLA-VBSeq: Nariai et al., BMC Genomics 2015

## License

MIT
