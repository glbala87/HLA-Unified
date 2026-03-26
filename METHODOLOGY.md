# HLA-Unified V2 — Methodology, Execution Guide & Development History

---

## Table of Contents

1. [Scientific Methodology](#1-scientific-methodology)
2. [Execution Guide](#2-execution-guide)
3. [Development History](#3-development-history)

---

## 1. Scientific Methodology

### 1.1 Overview

HLA-Unified V2 is a hybrid computational HLA typing pipeline that combines six established algorithmic approaches into a single multi-phase workflow:

| Algorithm Origin | Phase | What It Does |
|-----------------|-------|-------------|
| xHLA | Phase 1 | Fast read-to-allele pre-filtering via minimap2 |
| HLA-HD | Phase 2 | Iterative bowtie2 refinement by resolution level |
| OptiType | Phase 3 | Integer Linear Programming for optimal diploid pair |
| HLA-VBSeq | Phase 4 | Variational Bayes EM for calibrated posteriors |
| HLAforest | Phase 5a | K-mer-based orthogonal validation |
| HLAminer | Phase 5b | Targeted de novo assembly fallback |

V2 adds: haplotype phasing, DRB3/4/5 copy-number estimation, contamination screening, allele dropout detection, population frequency priors, ambiguity classification, novel allele annotation, and clinical reporting.

### 1.2 Pipeline Architecture

```
Input BAM/CRAM/FASTQ
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Read Extraction                                     │
│   Extract MHC-mapped reads (chr6:28.5-33.5 Mb) + unmapped   │
│   Auto-detect reference build (GRCh37/38, hg19/38, T2T)     │
│   Convert BAM → paired FASTQ via samtools                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Fast Pre-filter (xHLA-style)                        │
│   Align reads to combined IMGT/HLA CDS reference (minimap2)  │
│   Count reads per allele, rank by support                    │
│   Select top ~80 candidates per locus                        │
│   Modality-aware: minimap2 preset varies by data type        │
│     short → sr, pacbio → map-pb, hifi → map-hifi,           │
│     ont → map-ont, rna → splice                              │
└─────────────────────────────────────────────────────────────┘
        │  ~80 candidates/locus
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Iterative Refinement (HLA-HD-style)                 │
│   Round 1: Align to candidates (bowtie2 --very-sensitive)    │
│            Score by alignment quality, collapse to 2-digit   │
│            Keep top groups                                   │
│   Round 2: Re-align within top groups                        │
│            Score at 4-digit, select top ~20 per locus        │
│   For long reads: minimap2 replaces bowtie2                  │
└─────────────────────────────────────────────────────────────┘
        │  ~20 candidates/locus
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2.5: Haplotype Phasing                                 │
│   1. Detect heterozygous SNP positions from read pileup      │
│   2. Build read × het-site matrix (major=0, minor=1)         │
│   3. Cluster reads into 2 haplotype bins:                    │
│      • Short reads: greedy linkage clustering                │
│      • Long reads: signature-based direct clustering         │
│      • Fallback: spectral clustering (Laplacian eigenvector) │
│   4. Match each bin to best candidate allele by SNP profile  │
│   5. Feed phased allele pair as hint to ILP                  │
└─────────────────────────────────────────────────────────────┘
        │  phased allele hints
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: ILP Genotyping (OptiType-style)                     │
│                                                              │
│   Build read-allele matrix M[read, allele]:                  │
│     • Align reads to all remaining candidates                │
│     • Score = alignment score (AS tag), normalized per-row   │
│       to [0,1] to remove aligner-specific bias               │
│     • Include secondary alignments for multi-mapping reads   │
│                                                              │
│   Solve Integer Linear Program:                              │
│     maximize  Σ_i Σ_j M[i,j] × x[j]                        │
│     subject to:                                              │
│       Σ_j x[j] = 2  (diploid: exactly 2 alleles)            │
│       x[j] ∈ {0, 1}                                         │
│       y[i] ≤ Σ_j (M[i,j] × x[j])  (read explained)         │
│                                                              │
│   DRB3/4/5 Copy-Number Estimation:                           │
│     • Compare read depth at locus vs DRB1 (stable reference) │
│     • Depth ratio < 0.10 → ABSENT (0 copies, no call)       │
│     • Depth ratio 0.10-0.65 → HEMIZYGOUS (1 copy, 1 allele) │
│     • Depth ratio > 0.65 → DIPLOID (2 copies, standard)     │
│     • K-mer coverage as secondary signal                     │
│                                                              │
│   Contamination Screening:                                   │
│     • Count reads per allele at each locus                   │
│     • If ≥3 alleles have >10% read support at ≥2 loci →     │
│       FLAG: possible mixed-sample contamination              │
│                                                              │
│   Allele Dropout Detection:                                  │
│     • For homozygous calls, check:                           │
│       - Depth ratio vs reference (low → dropout risk)        │
│       - Het sites present despite homo call (suspicious)     │
│       - VB posterior confidence                              │
│     • Risk rated: NONE / LOW / MEDIUM / HIGH                 │
│                                                              │
│   Parallel execution: ThreadPoolExecutor across loci         │
└─────────────────────────────────────────────────────────────┘
        │  optimal diploid pair/locus
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Bayesian Confidence (HLA-VBSeq-style)               │
│                                                              │
│   Model: Dirichlet-Multinomial mixture                       │
│     π ~ Dir(α)     allele mixing proportions                 │
│     z_r ~ Cat(π)   read-to-allele assignment                 │
│     P(r|z=k) ∝ M[r,k]   read likelihood given allele        │
│                                                              │
│   Population frequency priors (V2):                          │
│     α_k = base_α + freq(allele_k) × N_alleles               │
│     108 built-in global frequencies from AFND                │
│     Common alleles (A*02:01=17%) get stronger prior          │
│     Rare/unknown alleles get floor prior (1e-5)              │
│                                                              │
│   Variational Bayes EM:                                      │
│     E-step: q(z_r=k) ∝ exp(E[log π_k] + log P(r|k))        │
│     M-step: α_q = α_0 + Σ_r q(z_r=k)                       │
│     Converge on ELBO (max 200 iterations, tol=1e-6)          │
│                                                              │
│   Pair posteriors:                                           │
│     Enumerate top-20 allele pairs by weight                  │
│     P(pair|data) = softmax of diploid log-likelihoods        │
│     log P(X|{a1,a2}) = Σ_r log[(P(r|a1)+P(r|a2))/2]        │
│                                                              │
│   Outputs: posterior, dosage, ranked alternatives, entropy   │
│   Parallel execution: ThreadPoolExecutor across loci         │
└─────────────────────────────────────────────────────────────┘
        │  calibrated posteriors + ranked pairs
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5a: K-mer Validation (HLAforest-style)                 │
│   Alignment-free orthogonal check:                           │
│   1. Extract k-mers (k=31) from called allele pair           │
│   2. Extract k-mers from reads at the locus                  │
│   3. Compute: proportion of allele k-mers covered by reads   │
│   4. Compute: fraction of read k-mers NOT in either allele   │
│   5. Chi-square uniformity test on k-mer coverage depth      │
│   6. Concordant if: coverage ≥ 90% AND unexpected ≤ 30%     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 5b: Assembly Fallback (HLAminer-style)                 │
│   Triggered when: VB confidence < 0.80, OR k-mer discordant,│
│   OR <50% reads explained, OR no VB/kmer and ILP weak        │
│                                                              │
│   1. Run assembler on locus reads:                           │
│      short → megahit, pacbio/ont → flye, hifi → hifiasm     │
│   2. Match contigs to known alleles:                         │
│      Fast screen: k-mer Jaccard similarity (k=21)            │
│      Refine top-5: minimap2 asm5 alignment                   │
│   3. If best match < 99.5% identity → flag as novel allele   │
│   4. Annotate mismatches: walk CIGAR to extract SNPs/indels  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 6: Post-processing (V2)                                │
│                                                              │
│   Ambiguity Classification:                                  │
│     Evaluate WHY a call is uncertain (not just that it is):  │
│     • UNAMBIGUOUS: posterior ≥ 0.99 AND gap ≥ 0.10           │
│     • COVERAGE_GAP: k-mer coverage < 70%                     │
│     • PHASE_BREAK: het sites but reads don't span            │
│     • CLOSE_ALLELES: top pairs differ by ≤3 SNPs             │
│     • EXON_ONLY: alleles differ only in introns (exome/RNA)  │
│     • LOW_DEPTH: < 10 reads at locus                         │
│     • HOMO_AMBIG: can't distinguish homo from close het      │
│     Each reason includes a resolution suggestion.            │
│                                                              │
│   Novel Allele Detection:                                    │
│     Proactive screening (not just assembly fallback):        │
│     Signal 1: >15% unexplained reads after ILP               │
│     Signal 2: >20% unexpected k-mers                         │
│     Signal 3: Assembly identity < 99.5%                      │
│     Confidence: HIGH (≥3 signals), MEDIUM (2), LOW (1)       │
│                                                              │
│   Novel Allele Annotation:                                   │
│     Map variants to gene model (exon/intron/UTR)             │
│     Classify: synonymous / non-synonymous / splice / frameshift│
│     Generate HGVS-like notation (e.g., A*02:01:g.750A>G)    │
│     Temporary designation (e.g., A*02:new)                   │
│                                                              │
│   Detailed Per-Locus QC:                                     │
│     23 metrics including haplotype balance, informative       │
│     positions, assembly N50/gene coverage, k-mer stats       │
│                                                              │
│   Confidence Tier:                                           │
│     HIGH:    posterior ≥ 0.99 AND ambiguity gap ≥ 0.10       │
│     MEDIUM:  posterior ≥ 0.90 AND ambiguity gap ≥ 0.05       │
│     LOW:     posterior ≥ 0.50                                 │
│     VERY_LOW: below 0.50                                     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Output Generation                                            │
│                                                              │
│   hla_types.tsv       Main results (TSV with provenance)     │
│   hla_types.json      Full JSON evidence trail               │
│   ambiguity.tsv       Ranked alternative diploid pairs       │
│   qc_report.json      Structured QC metrics                  │
│   qc_dashboard.html   Visual HTML dashboard                  │
│   clinical_summary.*  Clinical report (--clinical)           │
│   novel_alleles.json  HGVS annotations (if detected)        │
│   manifest.json       Environment snapshot (--strict)        │
│   imgt_lock.json      DB version lock (--strict)             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Decisions

**1. Hybrid engine** — No single approach works best for all data types. Pre-filtering (fast, lossy) narrows the search space; refinement (slower, precise) improves candidates; ILP (exact optimization) selects the optimal pair; VB (probabilistic) quantifies uncertainty; k-mer validation (alignment-free) catches systematic alignment errors; assembly (de novo) recovers novel sequences.

**2. Ambiguity is a feature** — Instead of forcing a single answer, the tool reports ranked alternatives with the *reason* for uncertainty. A transplant team needs to know whether ambiguity is resolvable (add more data) or inherent (close alleles).

**3. Modality-aware** — Short reads, long reads, exomes, and RNA-seq expose different parts of HLA biology. Each has different error profiles, coverage patterns, and achievable resolution. The tool uses modality-specific presets (7 types) rather than one-size-fits-all parameters.

**4. Database provenance** — Every call is tied to the exact IPD-IMGT/HLA release version. Lockfiles with SHA256 hashes enable exact reproduction of results across time and environments.

**5. Population priors** — The Dirichlet prior in the VB estimator is informed by global allele frequencies (108 alleles from AFND). Common alleles get a modest head start; rare alleles can still be called when the data is strong.

---

## 2. Execution Guide

### 2.1 Installation

```bash
# From the zip
unzip HLA-Unified.zip
cd HLA-Unified
pip install -e .

# Verify
hla-unified --version
# → hla-unified, version 2.0.0
```

**External tools** (install via conda or brew):
```bash
conda install -c bioconda samtools minimap2 bowtie2 megahit
# or
brew install samtools minimap2 bowtie2 megahit
```

### 2.2 Database Setup

```bash
# Download latest IMGT/HLA
hla-unified setup-db --out ./IMGTHLA

# Pin to a specific release (for reproducibility)
hla-unified setup-db --out ./IMGTHLA --release 3.56.0
```

### 2.3 Basic Typing

```bash
# Short-read WGS BAM (default)
hla-unified type \
  --bam sample.bam \
  --imgt-db ./IMGTHLA \
  --out ./results \
  --threads 8

# Paired FASTQ input
hla-unified type \
  --r1 sample_R1.fastq.gz \
  --r2 sample_R2.fastq.gz \
  --imgt-db ./IMGTHLA \
  --out ./results

# Exome
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type exome

# PacBio HiFi
hla-unified type --bam sample.hifi.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type hifi

# Oxford Nanopore
hla-unified type --bam sample.ont.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type ont

# RNA-seq
hla-unified type --bam sample.rna.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type rna

# Targeted HLA capture panel
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --data-type targeted_capture
```

### 2.4 Use-Case Profiles

```bash
# TRANSPLANT: conservative, clinical report, reproducibility manifest
# Reports only HIGH/MEDIUM confidence at classical transplant loci
# Generates clinical_summary.txt, manifest.json, imgt_lock.json
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --profile transplant --strict-reproducibility

# RESEARCH: all loci, max resolution, novel allele discovery
# Reports 15 loci including DRB3/4/5/DRA/E/F/G
# Accepts LOW confidence, reports novel alleles
hla-unified type --bam sample.bam --imgt-db ./IMGTHLA --out ./results \
  --profile research

# IMMUNO-ONCOLOGY: fast Class I for neoantigen prediction
# Only types A, B, C — minimal overhead
hla-unified type --bam tumor.bam --imgt-db ./IMGTHLA --out ./results \
  --profile immuno_onc
```

### 2.5 Advanced Options

```bash
# Specific loci only
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results \
  --loci A,B,C,DRB1

# Pin IMGT release version
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results \
  --imgt-release 3.56.0

# Control output resolution
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results \
  --output-resolution 2    # 2-field (protein level)
  # Options: 1, 2, 3, 4, G (G-group), max (assay limit)

# Generate clinical report without a profile
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results \
  --clinical

# Skip phases (faster, less accurate)
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results \
  --skip-refinement --skip-assembly --skip-kmer

# Verbose logging
hla-unified type --bam s.bam --imgt-db ./IMGTHLA --out ./results -vv
```

### 2.6 Validation & Benchmarking

```bash
# Compare one result against truth
hla-unified validate results/hla_types.tsv truth.tsv --resolution 2

# Benchmark suite on a dataset
# truth.tsv columns: SampleID, Locus, Allele1, Allele2 (optional: Ancestry, Depth)
hla-unified benchmark \
  --dataset truth.tsv \
  --bam-dir /path/to/bams/ \
  --imgt-db ./IMGTHLA \
  --out ./benchmark_results \
  --threads 16 \
  --resolution 2

# Evaluate pre-computed results (skip re-typing)
hla-unified benchmark \
  --dataset truth.tsv \
  --bam-dir /path/to/bams/ \
  --imgt-db ./IMGTHLA \
  --out ./benchmark_results \
  --skip-typing --results-dir ./existing_results
```

### 2.7 Nextflow (Batch Mode)

```bash
# Single sample
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --bam sample.bam \
  --imgt_db ./IMGTHLA \
  --outdir ./results

# Batch mode with sample sheet
# samples.csv: sample_id,bam_path,data_type
# NA12878,/data/NA12878.bam,short
# HG002,/data/HG002.hifi.bam,hifi
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --sample_sheet samples.csv \
  --imgt_db ./IMGTHLA \
  --outdir ./results

# With Docker
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --bam sample.bam \
  --imgt_db ./IMGTHLA \
  -with-docker hla-unified:2.0.0

# With transplant profile
nextflow run hla_unified/pipeline/nextflow_main.nf \
  --sample_sheet samples.csv \
  --imgt_db ./IMGTHLA \
  --profile_name transplant \
  --clinical true \
  --strict true
```

### 2.8 Docker

```bash
# Build
docker build -f Dockerfile.unified -t hla-unified:2.0.0 .

# Run
docker run -v /data:/data hla-unified:2.0.0 type \
  --bam /data/sample.bam \
  --imgt-db /data/IMGTHLA \
  --out /data/results

# Transplant mode
docker run -v /data:/data hla-unified:2.0.0 type \
  --bam /data/sample.bam \
  --imgt-db /data/IMGTHLA \
  --out /data/results \
  --profile transplant --strict-reproducibility
```

### 2.9 Output Files Reference

| File | Format | When Generated | Contents |
|------|--------|---------------|----------|
| `hla_types.tsv` | TSV | Always | Allele calls with confidence, ambiguity reason, scores, flags |
| `hla_types.json` | JSON | Always | Full evidence trail: calls, scores, alternatives, QC, ambiguity |
| `ambiguity.tsv` | TSV | Always | Ranked alternative diploid pairs per locus with posteriors |
| `qc_report.json` | JSON | Always | 23 per-locus QC metrics: haplotype balance, k-mer, phasing |
| `qc_dashboard.html` | HTML | Always | Standalone visual dashboard (no CDN dependencies) |
| `clinical_summary.txt` | Text | `--clinical` | Human-readable report with methodology and disclaimer |
| `clinical_summary.json` | JSON | `--clinical` | Machine-readable clinical report |
| `novel_alleles.json` | JSON | When detected | HGVS annotations, functional classification, gene region |
| `manifest.json` | JSON | `--strict-reproducibility` | Python/tool versions, OS, IMGT provenance |
| `imgt_lock.json` | JSON | `--strict-reproducibility` | IMGT release, commit, per-locus allele counts, SHA256 |

### 2.10 Interpreting Results

**TSV columns:**
```
Locus           HLA-A, HLA-B, etc.
Chromosome      1 or 2 (NOT phased across loci)
Allele          Called allele at output resolution
G_Group         G-group translation
GL_String       IMGT GL String format (HLA-A*02:01+HLA-A*01:01)
Confidence      HIGH / MEDIUM / LOW / VERY_LOW
Posterior       Calibrated posterior probability (0-1)
EvidenceScore   Fraction of reads explained by this pair
AmbiguityGap    Posterior difference: best pair - second best
AmbiguityReason Why the call may be uncertain (7 categories)
ReadsExplained  Number of reads supporting this pair
TotalReads      Total reads at this locus
KmerCovered     Fraction of allele k-mers found in reads
KmerConcordant  True if k-mer evidence supports the call
IsNovel         True if potential novel allele detected
Flags           Semicolon-separated warning flags
```

---

## 3. Development History

### 3.1 Starting Point — HLA-LA 1.0.4 Codebase

The project began with the existing HLA-LA 1.0.4 tool, which contained two implementations:

**Original C++/Perl (HLA*LA by Dilthey et al.):**
- Graph-based population reference graph (PRG) alignment
- BWA seed extraction → graph projection → Needleman-Wunsch extension
- G-group translation of calls
- 17 loci, TSV output with Q1 scores
- Files: `HLA-LA.cpp`, `HLA-LA.pl`, `Graph/`, `mapper/`, `hla/`

**Python reimplementation (hlala/):**
- Partial port of C++: graph, alignment, typing (~60-70% complete)
- Some key functions stubbed (k-mer coverage, Q1 score, unaccounted alleles)
- Files: `hlala/graph/`, `hlala/aligner/`, `hlala/hla/typer.py`

**Python multi-strategy pipeline (hla_unified/ V1):**
- 5-phase pipeline combining xHLA + HLA-HD + OptiType + VBSeq + HLAminer
- ~30 files, research-grade prototype
- Working ILP genotyper, VB confidence, k-mer validator, assembly fallback
- TSV + basic HTML output, G-group translation
- Score: 68/100 (solid architecture, prototype quality)

### 3.2 Phase 1 — Foundation (Config, Provenance, Docker)

**Problem:** Hardcoded thresholds scattered across files, no formal configuration, no version locking.

**Built:**
- `config/schema.py` — `PipelineConfig` dataclass (17 fields) with `from_cli()` factory, `LocusConfig` for 15 loci, `AssayPreset` (22 params), `UseCaseProfile`
- `config/presets.py` — 7 assay presets (short/exome/targeted_capture/pacbio/hifi/ont/rna) with tuned thresholds for each; 3 use-case profiles (transplant/research/immuno_onc)
- `config/manifest.py` — Runtime environment capture (Python/package/tool versions), IMGT lockfile with SHA256 hash, lockfile verification
- `Dockerfile.unified` — Multi-stage build with pinned tool versions (samtools 1.19, minimap2 2.26, bowtie2 2.5.3, megahit 1.2.9)
- Enhanced `reference/imgt_db.py` — Added `write_lockfile()`, `verify_lockfile()`, `get_allele_counts()`
- Bumped version to 2.0.0

### 3.3 Phase 2 — Core Inference (Ambiguity, Phasing, Posteriors)

**Problem:** V1 reported only top-1 call with a single confidence number. No explanation of WHY a call was uncertain. Phasing was short-read-only.

**Built:**
- `confidence/ambiguity_classifier.py` — `AmbiguityClassifier` with 7-reason taxonomy (UNAMBIGUOUS, COVERAGE_GAP, PHASE_BREAK, CLOSE_ALLELES, EXON_ONLY, LOW_DEPTH, HOMO_AMBIG). Each includes evidence dict and resolution suggestion.
- Enhanced `confidence/vb_estimator.py` — Increased `n_top_pairs` from 10→20, added `posterior_entropy()` method for ambiguity quantification
- Enhanced `phasing/haplotype_binner.py` — Added `_cluster_reads_longread()` for PacBio/ONT (signature-based direct clustering), `_cluster_reads_spectral()` (Laplacian eigenvector fallback for imbalanced bins), automatic routing by data type with spectral fallback at <10% balance

### 3.4 Phase 3 — QC & Output (Metrics, JSON, Clinical)

**Problem:** TSV-only output, no per-locus QC metrics beyond basic coverage, no clinical reporting.

**Built:**
- `qc/locus_metrics.py` — `DetailedLocusQC` with 23 metrics: haplotype balance, informative positions, exon coverage completeness, assembly N50/gene coverage, k-mer uniformity, read quality
- `output/writer.py` — `OutputWriter` producing TSV (with AmbiguityReason column), full JSON (evidence trail with score breakdown, alternatives, QC), ambiguity TSV
- `output/clinical.py` — `ClinicalReporter` generating JSON + plain text reports with DISCLAIMER, methodology statement, confidence-based locus filtering (withholds LOW/VERY_LOW), GL Strings
- Enhanced `qc/report.py` — Integrated `DetailedLocusQC` into QC report generation, added haplotype balance/informative positions/assembly metrics to HTML dashboard, updated QC JSON with `detailed_qc` section

### 3.5 Phase 4 — Novel Alleles & Presets

**Problem:** Novel allele detection only triggered for low-confidence calls (assembly fallback). No structured variant annotation.

**Built:**
- `novel/detector.py` — `NovelAlleleDetector` with proactive screening: 3 signals (unexplained reads >15%, unexpected k-mers >20%, assembly divergence >0.5%). Confidence rating: HIGH/MEDIUM/LOW.
- `novel/annotator.py` — `NovelAlleleAnnotator` mapping variants to gene model, classifying synonymous/non-synonymous/splice/frameshift, generating HGVS-like notation, temporary allele designations. Codon table for amino acid impact.

### 3.6 Phase 5 — Validation Framework

**Problem:** No benchmarking infrastructure, no way to compare across callers or evaluate at multiple depths.

**Built:**
- `benchmark/datasets.py` — `BenchmarkDataset.from_tsv()` with ancestry annotation
- `benchmark/metrics.py` — Order-independent `compare_diploid()` at configurable resolution, `compute_accuracy()`, `merge_accuracies()`, `BenchmarkReport` with per-ancestry stratification
- `benchmark/runner.py` — `BenchmarkRunner.run_dataset()` with parallel typing, discordant call tracking, JSON report generation
- `benchmark/downsampler.py` — `SyntheticDownsampler.run_depth_curve()` using samtools view -s
- `benchmark/consensus.py` — `CrossCallerConsensus` with parsers for 7 formats (HLA-Unified, HLA*LA, OptiType, xHLA, HLA-HD, arcasHLA, generic TSV), majority-vote consensus

### 3.7 Algorithmic Completeness Fixes

**Problem:** Audit revealed 6 critical/high gaps vs state-of-the-art HLA callers.

**Built:**
1. `genotyper/cnv.py` — `CopyNumberEstimator` for DRB3/4/5 (depth ratio vs DRB1, 3 states: ABSENT/HEMIZYGOUS/DIPLOID, k-mer override). Pipeline adjusts ILP constraint: 0 copies → no call, 1 → hemizygous, 2 → standard diploid.
2. Enhanced `genotyper/read_matrix.py` — Row-wise alignment score normalization to [0,1], removing minimap2/bowtie2/BWA scale bias
3. `qc/sample_qc.py` — `AlleleDropoutDetector` (depth ratio, het sites in homo calls, VB posterior → NONE/LOW/MEDIUM/HIGH risk) + `ContaminationDetector` (>2 alleles with >10% support at ≥2 loci)
4. Enhanced `confidence/vb_estimator.py` — Population frequency priors: `allele_frequencies` parameter sets Dirichlet alpha proportional to population frequency instead of uniform
5. `reference/frequencies.py` — `AlleleFrequencyDatabase` with 108 built-in global frequencies from AFND, TSV loader, allele-group fallback, population filtering
6. Enhanced `pipeline/runner.py` — `ThreadPoolExecutor` parallelization of ILP and VB across loci

### 3.8 Integration Wiring Fixes

**Problem:** Audit found new modules existed but weren't called from the pipeline.

**Fixed:**
- `haplotype_binner.py` line 133: Now routes to `_cluster_reads_longread()` for pacbio/hifi/ont, falls back to `_cluster_reads_spectral()` when bins are >90% imbalanced
- `pipeline/runner.py`: Now passes `data_type=self.data_type` to `phase_all_loci()`
- `pipeline/runner.py`: Now instantiates `LocusMetricsCalculator`, stores in `_last_locus_metrics`
- `__main__.py`: Now calls `NovelAlleleAnnotator.annotate()` for candidates with novel reports, writes `novel_alleles.json`
- `qc/report.py`: `generate_qc_report()` now accepts `detailed_metrics` and merges into `LocusQC`

### 3.9 Production Readiness

**Problem:** No tests for V2 modules, no allele frequencies, Nextflow V1 only, no README, pyproject not wired.

**Built:**
- `tests/unit/test_v2_modules.py` — 50 unit tests covering: PipelineConfig, AssayPresets, Manifest, CNV (7 tests), Dropout (4), Contamination (3), Ambiguity (5), Clinical (3), OutputWriter (2), NovelDetector (3), NovelAnnotator (2), BenchmarkMetrics (8), LocusMetrics (3)
- `tests/integration/test_pipeline_smoke.py` — 11 integration tests: OutputWriter end-to-end, Clinical transplant flow, Ambiguity classification, Novel detection→annotation chain, CNV→ILP integration, Frequency priors, Manifest generation
- `reference/frequencies.py` — 108 built-in allele frequencies, integrated into VB estimator via pipeline runner
- `pipeline/nextflow_main.nf` — Rewritten for V2: all flags, batch mode via `--sample_sheet` CSV, resource specs by data type, `errorStrategy 'retry'`, benchmark process
- `hla_unified/README.md` — Complete documentation of pipeline, presets, profiles, outputs, ambiguity table, CNV handling, benchmarking, Nextflow, Docker
- `pyproject.toml` — Now primary, installs `hla-unified` package with all dependencies

### 3.10 Final Codebase Statistics

```
Source files:     52 Python modules (9,137 lines)
Test files:       6 test files (61 tests, all passing)
Total files:      275 in repository (230 tracked by git)
Package size:     639 KB (zip)
External deps:    7 Python packages + 4 CLI tools
CLI commands:     5 (type, setup-db, build-panel, validate, benchmark)
CLI flags:        20+ (--profile, --clinical, --strict-reproducibility, etc.)
Output formats:   10 files (TSV, JSON, HTML, TXT, lockfiles)
Assay presets:    7 (short, exome, targeted_capture, pacbio, hifi, ont, rna)
Use-case profiles: 3 (transplant, research, immuno_onc)
Supported loci:   15 (A, B, C, DRB1, DQA1, DQB1, DPA1, DPB1, DRB3/4/5, DRA, E, F, G)
Ambiguity reasons: 7 (unambiguous, coverage_gap, phase_break, close_alleles, exon_only, low_depth, homo_ambig)
Allele frequencies: 108 built-in global (A, B, C, DRB1, DQB1, DQA1, DPA1, DPB1)
```
