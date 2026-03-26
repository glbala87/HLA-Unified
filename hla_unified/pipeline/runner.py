"""Unified pipeline orchestrator: wires all 5 phases together.

Pipeline flow:
  Input BAM/FASTQ
       |
  Phase 0: Read extraction (MHC region + unmapped)
       |
  Phase 1: Fast pre-filter (xHLA-style minimap2)
       |            ~50-100 candidates per locus
  Phase 2: Iterative refinement (HLA-HD-style)
       |            ~10-20 candidates per locus
  Phase 3: ILP genotyping (OptiType-style)
       |            optimal diploid pair per locus
  Phase 4: Bayesian confidence (VBSeq-style)
       |            calibrated posterior probabilities
  Phase 5: K-mer validation + assembly fallback
       |
  Output: TSV with allele calls, confidence, and validation flags
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam

from ..reference.loci import ALL_TYPING_LOCI, parse_allele_name, MHC_REGIONS
from ..reference.imgt_db import IMGTDatabase
from ..reference.g_groups import GGroupTranslator
from ..prefilter.read_extractor import ReadExtractor
from ..prefilter.fast_mapper import FastPrefilter
from ..refinement.iterative_refiner import IterativeRefiner
from ..genotyper.ilp_solver import solve_ilp, ILPResult
from ..genotyper.read_matrix import ReadAlleleMatrix, build_matrix_from_bam
from ..confidence.vb_estimator import VariationalBayesEstimator, ConfidenceResult
from ..kmer.validator import KmerValidator, KmerValidationResult
from ..assembly.targeted_assembler import TargetedAssembler, AssemblyResult
from ..utils.io import ensure_dir, write_fasta
from ..utils.external import run_cmd, run_pipeline, check_all_tools, ToolError

logger = logging.getLogger(__name__)


@dataclass
class AlternativeCall:
    """An alternative allele pair with its posterior probability."""
    allele1: str
    allele2: str
    posterior: float


@dataclass
class LocusScore:
    """Separated scoring: evidence + ambiguity + confidence."""
    evidence_score: float     # total alignment evidence (ILP objective / reads)
    ambiguity_gap: float      # posterior(best) - posterior(2nd best); 0=ambiguous, 1=certain
    confidence_posterior: float  # calibrated posterior probability
    confidence_tier: str      # HIGH / MEDIUM / LOW / VERY_LOW

    @staticmethod
    def compute_tier(posterior: float, ambiguity_gap: float) -> str:
        """Calibrated confidence tier using both posterior and ambiguity.

        Thresholds informed by benchmarking conventions:
        - HIGH: posterior >= 0.99 AND gap >= 0.10
        - MEDIUM: posterior >= 0.90 AND gap >= 0.05
        - LOW: posterior >= 0.50
        - VERY_LOW: below 0.50
        """
        if posterior >= 0.99 and ambiguity_gap >= 0.10:
            return "HIGH"
        elif posterior >= 0.90 and ambiguity_gap >= 0.05:
            return "MEDIUM"
        elif posterior >= 0.50:
            return "LOW"
        return "VERY_LOW"


@dataclass
class LocusCall:
    """Final HLA call for one locus."""
    locus: str
    allele1: str
    allele2: str
    g_group1: str
    g_group2: str
    confidence: str  # HIGH, MEDIUM, LOW, VERY_LOW
    posterior: float
    reads_explained: int
    total_reads: int
    kmer_covered: float
    kmer_concordant: bool
    is_novel: bool
    flags: list[str] = field(default_factory=list)
    # Explicit ambiguity model: ranked alternatives
    alternatives: list[AlternativeCall] = field(default_factory=list)
    # Separated scoring
    score: LocusScore | None = None
    # Novel allele annotation
    novel_summary: str = ""
    # GL String representation
    gl_string: str = ""


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    calls: dict[str, LocusCall]
    runtime_seconds: float
    phases_completed: list[str]
    imgt_release: str = ""
    imgt_commit: str = ""


class UnifiedPipeline:
    """Orchestrates the full multi-strategy HLA typing pipeline."""

    def __init__(
        self,
        imgt_db_path: str | Path,
        work_dir: str | Path,
        threads: int = 4,
        loci: list[str] | None = None,
        data_type: str = "short",
        output_resolution: int | str = "max",
        required_imgt_release: str | None = None,
        # Phase toggles
        skip_refinement: bool = False,
        skip_confidence: bool = False,
        skip_kmer: bool = False,
        skip_assembly: bool = False,
        # Phase parameters
        max_prefilter_candidates: int = 80,
        max_refined_candidates: int = 20,
        confidence_threshold: float = 0.90,
        assembly_confidence_trigger: float = 0.80,
    ) -> None:
        self.imgt_db = IMGTDatabase(imgt_db_path)
        self.work_dir = Path(work_dir)
        self.threads = threads
        self.loci = loci or list(ALL_TYPING_LOCI)
        self.data_type = data_type
        self.output_resolution = output_resolution
        self.required_imgt_release = required_imgt_release
        self.skip_refinement = skip_refinement
        self.skip_confidence = skip_confidence
        self.skip_kmer = skip_kmer
        self.skip_assembly = skip_assembly
        self.max_prefilter_candidates = max_prefilter_candidates
        self.max_refined_candidates = max_refined_candidates
        self.confidence_threshold = confidence_threshold
        self.assembly_trigger = assembly_confidence_trigger

    def run(
        self,
        input_path: str | Path,
        input_type: str = "bam",
        r2_path: str | Path | None = None,
        reference_build: str | None = None,
    ) -> PipelineResult:
        """Run the complete pipeline.

        Args:
            input_path: Path to BAM/CRAM or R1 FASTQ
            input_type: "bam", "cram", or "fastq"
            r2_path: Path to R2 FASTQ (for FASTQ input)
            reference_build: Reference genome build (auto-detected if None)
        """
        start_time = time.time()
        phases_completed = []

        ensure_dir(self.work_dir)

        # === Release version enforcement ===
        if self.required_imgt_release:
            actual = self.imgt_db.release_version
            if actual != self.required_imgt_release and actual != "unknown":
                raise ToolError(
                    f"IMGT/HLA release mismatch: required '{self.required_imgt_release}' "
                    f"but database is '{actual}'. "
                    f"Use --imgt-release to match, or re-run setup-db --release {self.required_imgt_release}"
                )
            logger.info("IMGT release verified: %s", actual)

        # Load assay-specific parameters
        from ..prefilter.fast_mapper import ASSAY_CONFIGS
        assay_cfg = ASSAY_CONFIGS.get(self.data_type, ASSAY_CONFIGS["short"])
        logger.info("Assay: %s (expected coverage ~%dx)",
                     assay_cfg["description"], assay_cfg["expected_coverage"])

        # Check required tools upfront
        required_tools = ["samtools", "minimap2"]
        if self.data_type in ("short", "exome", "targeted_capture") and not self.skip_refinement:
            required_tools.extend(["bowtie2", "bowtie2-build"])
        try:
            check_all_tools(required_tools)
        except ToolError as e:
            logger.error("Missing dependencies: %s", e)
            raise

        # === Phase 0: Read Extraction ===
        logger.info("=== Phase 0: Read Extraction ===")
        r1, r2 = self._extract_reads(
            input_path, input_type, r2_path, reference_build,
        )
        phases_completed.append("extraction")

        # === Phase 1: Fast Pre-filter ===
        logger.info("=== Phase 1: Fast Pre-filter (xHLA-style) ===")
        prefilter = FastPrefilter(
            threads=self.threads,
            max_candidates_per_locus=self.max_prefilter_candidates,
            data_type=self.data_type,
        )

        ref_fasta = self.work_dir / "imgt_combined.fa"
        self.imgt_db.build_combined_reference(
            self.loci, ref_fasta, genomic=False,
        )

        pf_results = prefilter.run(
            r1, r2, ref_fasta, self.work_dir / "prefilter", self.loci,
        )
        phases_completed.append("prefilter")

        # Collect candidate sequences
        all_candidates: dict[str, list[str]] = {}
        all_sequences: dict[str, str] = {}
        for locus in self.loci:
            pf = pf_results.per_locus.get(locus)
            if pf and pf.candidate_alleles:
                all_candidates[locus] = pf.candidate_alleles
                seqs = self.imgt_db.load_cds(locus)
                seqs.update(self.imgt_db.load_genomic(locus))
                all_sequences.update(seqs)
            else:
                all_candidates[locus] = []

        # === Phase 2: Iterative Refinement ===
        if not self.skip_refinement:
            logger.info("=== Phase 2: Iterative Refinement (HLA-HD-style) ===")
            refiner = IterativeRefiner(
                threads=self.threads, data_type=self.data_type,
            )
            ref_results = refiner.refine(
                r1, r2, all_candidates, all_sequences,
                self.work_dir / "refinement",
            )
            for locus, result in ref_results.items():
                if result.top_alleles:
                    all_candidates[locus] = result.top_alleles
            phases_completed.append("refinement")

        # === Phase 2.5: Haplotype Binning/Phasing ===
        phasing_results = {}
        logger.info("=== Phase 2.5: Haplotype Phasing ===")
        from ..phasing.haplotype_binner import HaplotypeBinner
        phaser = HaplotypeBinner()
        # We need a BAM aligned to candidates for phasing — build it now
        refined_bam = self._build_refined_alignment(
            r1, r2, all_candidates, all_sequences,
        )
        phasing_results = phaser.phase_all_loci(
            refined_bam, self.loci, all_candidates, all_sequences,
            data_type=self.data_type,
        )
        # Use phasing to inform allele selection: if phased, prefer
        # the alleles matched by each haplotype bin
        for locus, pr in phasing_results.items():
            if pr.is_phased and pr.bins[0].best_allele and pr.bins[1].best_allele:
                phased_pair = [pr.bins[0].best_allele, pr.bins[1].best_allele]
                # Ensure phased alleles are in candidate list
                for a in phased_pair:
                    if a not in all_candidates.get(locus, []):
                        all_candidates.setdefault(locus, []).append(a)
        phases_completed.append("phasing")

        # === Phase 3: ILP Genotyping ===
        logger.info("=== Phase 3: ILP Genotyping (OptiType-style) ===")
        # Rebuild alignment if candidates changed from phasing
        refined_bam = self._build_refined_alignment(
            r1, r2, all_candidates, all_sequences,
        )
        flat_candidates = [
            a for locus_alleles in all_candidates.values()
            for a in locus_alleles
        ]
        matrix = build_matrix_from_bam(refined_bam, flat_candidates)

        # Pre-compute per-locus sub-matrices (shared by ILP, VB, contamination)
        locus_matrices: dict[str, ReadAlleleMatrix] = {}
        for locus in self.loci:
            sub = matrix.subset_locus(locus)
            if sub.n_alleles > 0:
                locus_matrices[locus] = sub

        # === DRB3/4/5 Copy-Number Estimation ===
        from ..genotyper.cnv import CopyNumberEstimator, CNV_LOCI, adjust_ilp_for_cnv
        cnv_estimator = CopyNumberEstimator()
        reads_per_locus_count = {
            locus: sub.n_reads for locus, sub in locus_matrices.items()
        }
        cnv_estimates = cnv_estimator.estimate_all(
            self.loci, reads_per_locus_count,
        )

        # Parallel ILP genotyping across loci
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_ilp(locus: str) -> tuple[str, ILPResult]:
            sub = locus_matrices[locus]
            # Adjust allele count for CNV loci (DRB3/4/5)
            cnv = cnv_estimates.get(locus)
            if cnv and cnv.copy_number == 0:
                return locus, ILPResult(
                    locus=locus, allele1="", allele2="",
                    reads_explained=0, total_reads=sub.n_reads,
                    objective_value=0, is_homozygous=False,
                    solver_status="cnv_absent",
                )
            return locus, solve_ilp(sub, locus=locus)

        ilp_results: dict[str, ILPResult] = {}
        with ThreadPoolExecutor(max_workers=min(self.threads, len(locus_matrices))) as pool:
            futures = {pool.submit(_run_ilp, locus): locus for locus in locus_matrices}
            for future in as_completed(futures):
                locus, result = future.result()
                ilp_results[locus] = result

                # For hemizygous CNV loci, force single-allele call
                cnv = cnv_estimates.get(locus)
                if cnv and cnv.copy_number == 1 and result.allele1:
                    ilp_results[locus] = ILPResult(
                        locus=locus,
                        allele1=result.allele1,
                        allele2="",
                        reads_explained=result.reads_explained,
                        total_reads=result.total_reads,
                        objective_value=result.objective_value,
                        is_homozygous=False,
                        solver_status="cnv_hemizygous",
                    )
        phases_completed.append("ilp_genotyping")

        # === Phase 4: Bayesian Confidence (parallelized) ===
        vb_results: dict[str, ConfidenceResult] = {}
        if not self.skip_confidence:
            logger.info("=== Phase 4: Bayesian Confidence (VBSeq-style) ===")
            # Load population frequency priors
            allele_freq_dict = None
            try:
                from ..reference.frequencies import load_default_frequencies
                freq_db = load_default_frequencies()
                allele_freq_dict = {
                    name: freq_db.get_frequency(name)
                    for name in all_sequences.keys()
                }
                logger.info("Using population frequency priors (%d alleles)", len(allele_freq_dict))
            except Exception as e:
                logger.debug("Frequency priors unavailable: %s", e)
            vb = VariationalBayesEstimator(allele_frequencies=allele_freq_dict)

            def _run_vb(locus: str) -> tuple[str, ConfidenceResult]:
                return locus, vb.estimate(locus_matrices[locus], locus)

            with ThreadPoolExecutor(max_workers=min(self.threads, len(locus_matrices))) as pool:
                futures = {pool.submit(_run_vb, locus): locus for locus in locus_matrices}
                for future in as_completed(futures):
                    locus, result = future.result()
                    vb_results[locus] = result
            phases_completed.append("bayesian_confidence")

        # === Contamination Screening ===
        from ..qc.sample_qc import ContaminationDetector
        contam_detector = ContaminationDetector()
        contamination_result = contam_detector.screen(
            self.loci, locus_matrices, ilp_results,
        )
        if contamination_result.is_contaminated:
            logger.warning(
                "CONTAMINATION WARNING: %s",
                contamination_result.recommendation,
            )
        self._last_contamination = contamination_result
        self._last_cnv_estimates = cnv_estimates

        # === Allele Dropout Detection ===
        from ..qc.sample_qc import AlleleDropoutDetector
        dropout_detector = AlleleDropoutDetector()
        dropout_assessments = {}
        ref_reads = reads_per_locus_count.get("DRB1", 0)
        for locus in self.loci:
            ilp_r = ilp_results.get(locus)
            if ilp_r and ilp_r.is_homozygous:
                dropout_assessments[locus] = dropout_detector.assess(
                    locus=locus,
                    ilp_result=ilp_r,
                    vb_result=vb_results.get(locus),
                    phasing_result=phasing_results.get(locus),
                    locus_reads=reads_per_locus_count.get(locus, 0),
                    reference_locus_reads=ref_reads,
                )
                if dropout_assessments[locus].dropout_risk in ("HIGH", "MEDIUM"):
                    logger.warning(
                        "%s: %s dropout risk — %s",
                        locus,
                        dropout_assessments[locus].dropout_risk,
                        dropout_assessments[locus].recommendation,
                    )
        self._last_dropout_assessments = dropout_assessments

        # === Phase 5: K-mer Validation + Assembly ===
        kmer_results: dict[str, KmerValidationResult] = {}
        assembly_results: dict[str, AssemblyResult] = {}

        if not self.skip_kmer:
            logger.info("=== Phase 5a: K-mer Validation ===")
            kmer_val = KmerValidator()
            calls_for_kmer = {}
            for locus, ilp in ilp_results.items():
                if ilp.allele1 and ilp.allele2:
                    calls_for_kmer[locus] = (ilp.allele1, ilp.allele2)

            locus_reads = self._extract_locus_reads(refined_bam, self.loci)
            kmer_results = kmer_val.validate_all_loci(
                calls_for_kmer, all_sequences, locus_reads,
            )
            phases_completed.append("kmer_validation")

        if not self.skip_assembly:
            logger.info("=== Phase 5b: Assembly Fallback ===")
            assembler = TargetedAssembler(
                threads=self.threads, data_type=self.data_type,
            )
            for locus in self.loci:
                vb_r = vb_results.get(locus)
                kmer_r = kmer_results.get(locus)
                ilp_r = ilp_results.get(locus)

                # Trigger assembly if:
                # 1. VB confidence is low, OR
                # 2. K-mer validation is discordant, OR
                # 3. ILP explained few reads (< 50%), OR
                # 4. No VB/kmer results available but ILP looks weak
                needs_assembly = False
                if vb_r and vb_r.posterior_prob < self.assembly_trigger:
                    needs_assembly = True
                if kmer_r and not kmer_r.is_concordant:
                    needs_assembly = True
                if ilp_r and ilp_r.total_reads > 0:
                    explained_frac = ilp_r.reads_explained / ilp_r.total_reads
                    if explained_frac < 0.5:
                        needs_assembly = True
                # If both VB and kmer were skipped, check ILP quality only
                if not vb_r and not kmer_r and ilp_r:
                    if ilp_r.reads_explained < 10:
                        needs_assembly = True

                if needs_assembly:
                    logger.info("Triggering assembly for %s", locus)
                    locus_seqs = self.imgt_db.load_genomic(locus)
                    assembly_results[locus] = assembler.assemble_locus(
                        r1, r2, locus, locus_seqs,
                        self.work_dir / "assembly" / locus,
                    )
            phases_completed.append("assembly_fallback")

        # === Store intermediate results for V2 post-processing ===
        self._last_ilp_results = ilp_results
        self._last_vb_results = vb_results
        self._last_kmer_results = kmer_results
        self._last_assembly_results = assembly_results
        self._last_phasing_results = phasing_results

        # === Compile Final Results ===
        logger.info("=== Compiling Results ===")
        g_translator = GGroupTranslator()
        g_group_file = self.imgt_db.db_dir / "wmda" / "hla_nom_g.txt"
        if g_group_file.exists():
            g_translator.load(g_group_file)

        calls = self._compile_results(
            ilp_results, vb_results, kmer_results,
            assembly_results, g_translator,
        )

        runtime = time.time() - start_time
        provenance = self.imgt_db.provenance
        logger.info("Pipeline complete in %.1f seconds", runtime)

        result = PipelineResult(
            calls=calls,
            runtime_seconds=runtime,
            phases_completed=phases_completed,
            imgt_release=provenance.get("release", "unknown"),
            imgt_commit=provenance.get("git_commit", "unknown"),
        )

        # Write main output TSV (with provenance header)
        self._write_output(result, self.work_dir / "hla_types.tsv")

        # Write ambiguity report (ranked alternative calls)
        self._write_ambiguity_report(result, self.work_dir / "ambiguity.tsv")

        # === Compute Detailed Per-Locus QC Metrics ===
        from ..qc.locus_metrics import LocusMetricsCalculator
        metrics_calc = LocusMetricsCalculator()
        reads_per_locus = {
            locus: (
                ilp_results[locus].reads_explained if locus in ilp_results else 0,
                ilp_results[locus].total_reads if locus in ilp_results else 0,
            )
            for locus in self.loci
        }
        self._last_locus_metrics = metrics_calc.calculate_all(
            loci=self.loci,
            phasing_results=phasing_results,
            kmer_results=kmer_results,
            assembly_results=assembly_results,
            reads_per_locus=reads_per_locus,
        )

        # Write QC reports (JSON + HTML dashboard)
        from ..qc.report import generate_qc_report, write_qc_json, write_qc_html
        qc = generate_qc_report(
            result, provenance, self.data_type, str(input_path),
            phasing_results=phasing_results,
            kmer_results=kmer_results,
            assembly_results=assembly_results,
            detailed_metrics=self._last_locus_metrics,
        )
        write_qc_json(qc, self.work_dir / "qc_report.json")
        write_qc_html(qc, self.work_dir / "qc_dashboard.html")

        return result

    def _extract_reads(
        self,
        input_path: str | Path,
        input_type: str,
        r2_path: str | Path | None,
        reference_build: str | None,
    ) -> tuple[Path, Path | None]:
        """Phase 0: Extract HLA-relevant reads."""
        input_path = Path(input_path)

        if input_type == "fastq":
            r2 = Path(r2_path) if r2_path else None
            return input_path, r2

        extractor = ReadExtractor(threads=self.threads)

        if reference_build is None:
            reference_build = extractor.detect_reference(str(input_path))
            if reference_build:
                logger.info("Detected reference: %s", reference_build)
            else:
                logger.warning("Could not detect reference, defaulting to GRCh38")
                reference_build = "GRCh38"

        region_info = MHC_REGIONS.get(reference_build, ("chr6", 28510120, 33480577))
        region_str = f"{region_info[0]}:{region_info[1]}-{region_info[2]}"

        r1, r2 = extractor.extract_mhc_reads(
            str(input_path),
            self.work_dir / "extraction",
            region=region_str,
        )
        return r1, r2

    def _build_refined_alignment(
        self,
        r1: Path,
        r2: Path | None,
        candidates: dict[str, list[str]],
        sequences: dict[str, str],
    ) -> Path:
        """Align reads to refined candidate set for matrix building."""
        align_dir = self.work_dir / "refined_alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        ref_seqs = {}
        for locus_alleles in candidates.values():
            for a in locus_alleles:
                if a in sequences:
                    ref_seqs[a] = sequences[a]

        if not ref_seqs:
            raise ToolError("No candidate allele sequences found for alignment")

        ref_fasta = align_dir / "candidates.fa"
        write_fasta(ref_fasta, ref_seqs)

        output_bam = align_dir / "aligned.bam"

        if self.data_type == "short":
            # bowtie2 for short reads
            idx = align_dir / "bt2idx"
            run_cmd(
                ["bowtie2-build", "--quiet", str(ref_fasta), str(idx)],
                description="build bowtie2 index for refined alignment",
            )
            align_cmd = [
                "bowtie2", "-x", str(idx),
                "--very-sensitive", "-k", "50", "--no-unal",
                "-p", str(self.threads),
            ]
            if r2 and r2.exists():
                align_cmd.extend(["-1", str(r1), "-2", str(r2)])
            else:
                align_cmd.extend(["-U", str(r1)])
        else:
            # minimap2 for long reads / RNA-seq
            from ..prefilter.fast_mapper import MINIMAP2_PRESETS
            preset = MINIMAP2_PRESETS.get(self.data_type, "sr")
            align_cmd = [
                "minimap2", "-a", "-x", preset,
                "-t", str(self.threads),
                "--secondary=yes", "-N", "50",
                str(ref_fasta), str(r1),
            ]

        sort_cmd = ["samtools", "sort", "-@", "2", "-"]
        run_pipeline(
            [align_cmd, sort_cmd],
            output_path=output_bam,
            description="refined alignment for ILP matrix",
        )

        run_cmd(
            ["samtools", "index", str(output_bam)],
            description="index refined alignment BAM",
        )
        return output_bam

    def _extract_locus_reads(
        self, bam_path: Path, loci: list[str],
    ) -> dict[str, list[str]]:
        """Extract read sequences per locus from the aligned BAM."""
        locus_reads: dict[str, list[str]] = {locus: [] for locus in loci}

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if read.is_unmapped:
                    continue
                ref = read.reference_name
                if ref and read.query_sequence:
                    info = parse_allele_name(ref)
                    if info.locus in locus_reads:
                        locus_reads[info.locus].append(read.query_sequence)

        return locus_reads

    def _compile_results(
        self,
        ilp: dict[str, ILPResult],
        vb: dict[str, ConfidenceResult],
        kmer: dict[str, KmerValidationResult],
        assembly: dict[str, AssemblyResult],
        g_translator: GGroupTranslator,
    ) -> dict[str, LocusCall]:
        """Merge results from all phases into final calls."""
        from ..reference.loci import get_max_resolution, truncate_to_resolution

        calls = {}

        for locus in self.loci:
            ilp_r = ilp.get(locus)
            vb_r = vb.get(locus)
            kmer_r = kmer.get(locus)
            asm_r = assembly.get(locus)

            if not ilp_r or not ilp_r.allele1:
                calls[locus] = LocusCall(
                    locus=locus, allele1="", allele2="",
                    g_group1="", g_group2="",
                    confidence="VERY_LOW", posterior=0.0,
                    reads_explained=0, total_reads=0,
                    kmer_covered=0.0, kmer_concordant=False,
                    is_novel=False, flags=["no_call"],
                )
                continue

            # Prefer VB alleles if available
            a1 = vb_r.allele1 if vb_r else ilp_r.allele1
            a2 = vb_r.allele2 if vb_r else ilp_r.allele2

            # === Resolution control ===
            if self.output_resolution == "G":
                # G-group resolution
                g1, _ = g_translator.translate(a1)
                g2, _ = g_translator.translate(a2)
                a1, a2 = g1, g2
            elif self.output_resolution == "max":
                max_res = get_max_resolution(locus, self.data_type)
                a1 = truncate_to_resolution(a1, max_res)
                a2 = truncate_to_resolution(a2, max_res)
            elif isinstance(self.output_resolution, int):
                a1 = truncate_to_resolution(a1, self.output_resolution)
                a2 = truncate_to_resolution(a2, self.output_resolution)

            g1, _ = g_translator.translate(a1)
            g2, _ = g_translator.translate(a2)

            posterior = vb_r.posterior_prob if vb_r else 0.5

            # === Separated scoring ===
            evidence_score = (
                ilp_r.reads_explained / max(ilp_r.total_reads, 1)
            )
            # Ambiguity gap: posterior(best) - posterior(2nd best)
            ambiguity_gap = 1.0
            if vb_r and vb_r.alternative_pairs:
                second_best = vb_r.alternative_pairs[0][2] if vb_r.alternative_pairs else 0.0
                ambiguity_gap = posterior - second_best

            confidence = LocusScore.compute_tier(posterior, ambiguity_gap)
            score = LocusScore(
                evidence_score=round(evidence_score, 4),
                ambiguity_gap=round(ambiguity_gap, 4),
                confidence_posterior=round(posterior, 4),
                confidence_tier=confidence,
            )

            kmer_cov = kmer_r.proportion_kmers_covered if kmer_r else 0.0
            kmer_ok = kmer_r.is_concordant if kmer_r else True
            is_novel = asm_r.is_novel if asm_r else False

            novel_summary = ""
            if asm_r and asm_r.novel_report:
                novel_summary = asm_r.novel_report.summary

            flags = []
            if kmer_r and kmer_r.flags:
                flags.extend(kmer_r.flags)
            if is_novel:
                flags.append("potential_novel_allele")
            if ilp_r.is_homozygous:
                flags.append("homozygous")
            if not kmer_ok:
                flags.append("kmer_discordant")

            # Explicit ambiguity model: ranked alternatives
            alternatives = []
            if vb_r and vb_r.alternative_pairs:
                for alt_a1, alt_a2, alt_post in vb_r.alternative_pairs:
                    alternatives.append(AlternativeCall(
                        allele1=alt_a1, allele2=alt_a2, posterior=alt_post,
                    ))

            # GL String (Genotype List String, IMGT/HLA standard)
            gl = f"HLA-{a1}+HLA-{a2}"

            calls[locus] = LocusCall(
                locus=locus, allele1=a1, allele2=a2,
                g_group1=g1, g_group2=g2,
                confidence=confidence, posterior=posterior,
                reads_explained=ilp_r.reads_explained,
                total_reads=ilp_r.total_reads,
                kmer_covered=kmer_cov, kmer_concordant=kmer_ok,
                is_novel=is_novel, flags=flags,
                alternatives=alternatives,
                score=score,
                novel_summary=novel_summary,
                gl_string=gl,
            )

        return calls

    def _write_output(self, result: PipelineResult, path: Path) -> None:
        """Write final results to TSV with provenance header."""
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "Locus", "Chromosome", "Allele", "G_Group", "GL_String",
            "Confidence", "Posterior", "EvidenceScore", "AmbiguityGap",
            "ReadsExplained", "TotalReads",
            "KmerCovered", "KmerConcordant", "IsNovel", "Flags",
        ]

        with open(path, "w", newline="") as fh:
            # Provenance header: ties every call to exact IMGT release
            fh.write(f"## IPD-IMGT/HLA Release: {result.imgt_release}\n")
            fh.write(f"## IMGT Commit: {result.imgt_commit}\n")
            fh.write(f"## HLA-Unified Version: 2.0.0\n")
            fh.write(f"## Data Type: {self.data_type}\n")
            fh.write(f"## Output Resolution: {self.output_resolution}\n")

            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for locus in self.loci:
                call = result.calls.get(locus)
                if not call:
                    continue

                ev = call.score.evidence_score if call.score else 0.0
                ag = call.score.ambiguity_gap if call.score else 0.0

                for chrom, allele, ggroup in [
                    (1, call.allele1, call.g_group1),
                    (2, call.allele2, call.g_group2),
                ]:
                    writer.writerow({
                        "Locus": f"HLA-{locus}",
                        "Chromosome": chrom,
                        "Allele": allele,
                        "G_Group": ggroup,
                        "GL_String": call.gl_string,
                        "Confidence": call.confidence,
                        "Posterior": f"{call.posterior:.4f}",
                        "EvidenceScore": f"{ev:.4f}",
                        "AmbiguityGap": f"{ag:.4f}",
                        "ReadsExplained": call.reads_explained,
                        "TotalReads": call.total_reads,
                        "KmerCovered": f"{call.kmer_covered:.4f}",
                        "KmerConcordant": call.kmer_concordant,
                        "IsNovel": call.is_novel,
                        "Flags": ";".join(call.flags) if call.flags else "",
                    })

        logger.info("Results written to %s", path)

    def _write_ambiguity_report(
        self, result: PipelineResult, path: Path,
    ) -> None:
        """Write explicit ambiguity report: ranked alternative calls per locus."""
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "Locus", "Rank", "Allele1", "Allele2", "Posterior", "IsBestCall",
        ]

        with open(path, "w", newline="") as fh:
            fh.write(f"## IPD-IMGT/HLA Release: {result.imgt_release}\n")
            fh.write("## Ranked alternative diploid genotype calls per locus\n")

            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for locus in self.loci:
                call = result.calls.get(locus)
                if not call or not call.allele1:
                    continue

                # Rank 1: best call
                writer.writerow({
                    "Locus": f"HLA-{locus}",
                    "Rank": 1,
                    "Allele1": call.allele1,
                    "Allele2": call.allele2,
                    "Posterior": f"{call.posterior:.4f}",
                    "IsBestCall": True,
                })

                # Rank 2+: alternatives
                for rank, alt in enumerate(call.alternatives, start=2):
                    writer.writerow({
                        "Locus": f"HLA-{locus}",
                        "Rank": rank,
                        "Allele1": alt.allele1,
                        "Allele2": alt.allele2,
                        "Posterior": f"{alt.posterior:.4f}",
                        "IsBestCall": False,
                    })

        logger.info("Ambiguity report written to %s", path)
