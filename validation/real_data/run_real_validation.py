#!/usr/bin/env python3
"""Real-data validation for HLA-Unified V2.

Downloads GIAB reference BAMs, runs the full typing pipeline end-to-end,
and validates results against published consensus HLA types and cross-caller
concordance from HLA*LA, OptiType, xHLA, and arcasHLA.

Usage:
    # Quick: single sample, MHC region only (~100MB download)
    python run_real_validation.py --samples NA12878 --imgt-db IMGTHLA --out results --region-only

    # Full: all GIAB samples
    python run_real_validation.py --samples NA12878,HG002,HG003,HG004,NA19240 \
        --imgt-db IMGTHLA --out results --threads 8

    # Skip download if BAMs exist locally
    python run_real_validation.py --samples NA12878 --bam-dir /data/bams \
        --imgt-db IMGTHLA --out results

    # Evaluate pre-computed results only (no pipeline run)
    python run_real_validation.py --samples NA12878 --skip-typing \
        --results-dir results --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hla_unified.benchmark.datasets import BenchmarkDataset
from hla_unified.benchmark.runner import BenchmarkRunner
from hla_unified.benchmark.metrics import (
    compare_diploid, compute_accuracy, merge_accuracies,
)

VALIDATION_DIR = Path(__file__).resolve().parent
TRUTH_TSV = VALIDATION_DIR / "giab_truth.tsv"
PUBLISHED_DIR = VALIDATION_DIR / "published_calls"

logger = logging.getLogger(__name__)

# ── GIAB BAM download URLs (GRCh38) ─────────────────────────────────
# Verified NCBI FTP paths for GIAB reference samples
_GIAB_BASE = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data"
_AJ_BASE = f"{_GIAB_BASE}/AshkenazimTrio"
_MHC_REGION = {"region_chr": "chr6", "region_start": 28510120, "region_end": 33480577}

GIAB_BAMS = {
    "NA12878": {
        "url": f"{_GIAB_BASE}/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam",
        "index": f"{_GIAB_BASE}/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam.bai",
        **_MHC_REGION,
    },
    "HG002": {
        "url": f"{_AJ_BASE}/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.GRCh38.300x.bam",
        "index": f"{_AJ_BASE}/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.GRCh38.300x.bam.bai",
        **_MHC_REGION,
    },
    "HG003": {
        "url": f"{_AJ_BASE}/HG003_NA24149_father/NIST_HiSeq_HG003_Homogeneity-12389378/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG003.GRCh38.300x.bam",
        "index": f"{_AJ_BASE}/HG003_NA24149_father/NIST_HiSeq_HG003_Homogeneity-12389378/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG003.GRCh38.300x.bam.bai",
        **_MHC_REGION,
    },
    "HG004": {
        "url": f"{_AJ_BASE}/HG004_NA24143_mother/NIST_HiSeq_HG004_Homogeneity-14572558/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG004.GRCh38.300x.bam",
        "index": f"{_AJ_BASE}/HG004_NA24143_mother/NIST_HiSeq_HG004_Homogeneity-14572558/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG004.GRCh38.300x.bam.bai",
        **_MHC_REGION,
    },
    "NA19240": {
        # 1000 Genomes high-coverage (30x) — EBI mirror
        "url": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/data/YRI/NA19240/alignment/NA19240.alt_bwamem_GRCh38DH.20150718.YRI.high_coverage.cram",
        "index": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/data/YRI/NA19240/alignment/NA19240.alt_bwamem_GRCh38DH.20150718.YRI.high_coverage.cram.crai",
        "is_cram": True,
        **_MHC_REGION,
    },
}


def download_bam(
    sample_id: str, bam_dir: Path, region_only: bool = True,
) -> Path:
    """Download a GIAB BAM file (full or MHC region only).

    In region-only mode, uses samtools remote access (HTTP range requests)
    to download only the MHC region (~100MB instead of ~500GB).
    """
    info = GIAB_BAMS.get(sample_id)
    if not info:
        raise ValueError(f"Unknown sample: {sample_id}. Available: {list(GIAB_BAMS)}")

    bam_dir.mkdir(parents=True, exist_ok=True)
    bam_path = bam_dir / f"{sample_id}.bam"

    if bam_path.exists():
        size_mb = bam_path.stat().st_size / (1024 * 1024)
        logger.info("BAM already exists: %s (%.0f MB)", bam_path, size_mb)
        return bam_path

    is_cram = info.get("is_cram", False)
    remote_url = info["url"]
    region = f"{info['region_chr']}:{info['region_start']}-{info['region_end']}"

    if region_only:
        # samtools can fetch just the MHC region via HTTP range requests
        # This downloads ~100MB instead of the full 500GB BAM
        logger.info(
            "Extracting MHC region %s from remote %s for %s ...",
            region, "CRAM" if is_cram else "BAM", sample_id,
        )

        # samtools view works with remote BAM/CRAM if index is at URL.bai/.crai
        # For some servers we need to explicitly provide the index
        cmd = ["samtools", "view", "-b", "-h", "-o", str(bam_path)]

        # If CRAM, we may need a reference — but for MHC region extraction
        # to BAM output, samtools handles this via the embedded ref
        if is_cram:
            cmd.extend(["--output-fmt", "BAM"])

        cmd.extend([remote_url, region])

        try:
            logger.info("Running: %s", " ".join(cmd[-3:]))
            subprocess.run(cmd, check=True, timeout=3600)
        except subprocess.CalledProcessError:
            # Fallback: download index first, then extract
            logger.info("Direct remote access failed, downloading index first...")
            idx_suffix = ".crai" if is_cram else ".bai"
            idx_path = bam_dir / f"{sample_id}.remote{idx_suffix}"
            subprocess.run(
                ["wget", "-q", "-O", str(idx_path), info["index"]],
                check=True, timeout=600,
            )
            # Retry with local index hint
            env = {**subprocess.os.environ, "REF_PATH": ""}
            subprocess.run(cmd, check=True, timeout=3600, env=env)

        # Index the extracted BAM
        subprocess.run(["samtools", "index", str(bam_path)], check=True)

        size_mb = bam_path.stat().st_size / (1024 * 1024)
        logger.info("MHC region extracted: %s (%.0f MB)", bam_path, size_mb)
    else:
        # Full BAM/CRAM download
        logger.info("Downloading full file for %s (may be hundreds of GB)...", sample_id)
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(bam_path), remote_url],
            check=True, timeout=36000,
        )
        idx_suffix = ".crai" if is_cram else ".bai"
        idx_path = bam_dir / f"{sample_id}.bam{idx_suffix}"
        subprocess.run(
            ["wget", "-q", "-O", str(idx_path), info["index"]],
            check=True, timeout=600,
        )

    logger.info("BAM ready: %s", bam_path)
    return bam_path


def run_typing(
    sample_id: str,
    bam_path: Path,
    imgt_db: str,
    out_dir: Path,
    threads: int = 4,
) -> dict[str, tuple[str, str]]:
    """Run HLA-Unified V2 typing on a real BAM."""
    import shutil

    sample_out = out_dir / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    logger.info("Running HLA-Unified V2 typing on %s ...", sample_id)

    cmd = [
        sys.executable, "-m", "hla_unified", "type",
        "--bam", str(bam_path),
        "--imgt-db", imgt_db,
        "--out", str(sample_out),
        "--threads", str(threads),
        "--data-type", "short",
        "-vv",
    ]

    # Skip assembly if megahit not installed (optional phase)
    if not shutil.which("megahit"):
        cmd.append("--skip-assembly")
        logger.info("megahit not found, skipping assembly fallback phase")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=7200,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.error("Typing failed for %s:\nstdout: %s\nstderr: %s",
                      sample_id, result.stdout[-500:], result.stderr[-500:])
        return {}

    # Parse results
    return load_results(sample_out / "hla_types.tsv")


def load_results(tsv_path: Path) -> dict[str, tuple[str, str]]:
    """Load typing results from TSV."""
    calls: dict[str, dict[int, str]] = {}
    if not tsv_path.exists():
        return {}

    with open(tsv_path) as fh:
        lines = [l for l in fh if not l.startswith("#")]

    if not lines:
        return {}

    import io
    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter="\t")
    for row in reader:
        locus = row.get("Locus", "").replace("HLA-", "")
        chrom = int(row.get("Chromosome", 0))
        allele = row.get("Allele", "")
        calls.setdefault(locus, {})[chrom] = allele

    return {
        locus: (chroms.get(1, ""), chroms.get(2, ""))
        for locus, chroms in calls.items()
    }


def load_truth(
    truth_tsv: Path, samples: list[str],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Load truth set, filtered to requested samples."""
    truth: dict[str, dict[str, tuple[str, str]]] = {}
    with open(truth_tsv) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row["SampleID"]
            if sid not in samples:
                continue
            locus = row["Locus"].replace("HLA-", "")
            truth.setdefault(sid, {})[locus] = (row["Allele1"], row["Allele2"])
    return truth


def load_published_calls(
    published_dir: Path, samples: list[str],
) -> dict[str, dict[str, dict[str, tuple[str, str]]]]:
    """Load published caller results.

    Returns: {caller: {sample_id: {locus: (a1, a2)}}}
    """
    results: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}

    for tsv_path in published_dir.glob("*_published.tsv"):
        with open(tsv_path) as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                sid = row["SampleID"]
                if sid not in samples:
                    continue
                caller = row["Caller"]
                locus = row["Locus"].replace("HLA-", "")
                results.setdefault(caller, {}).setdefault(sid, {})[locus] = (
                    row["Allele1"], row["Allele2"],
                )

    return results


def validate_against_truth(
    calls: dict[str, tuple[str, str]],
    truth: dict[str, tuple[str, str]],
    sample_id: str,
    resolution: int = 2,
) -> dict:
    """Validate a single sample's calls against truth."""
    per_locus = compute_accuracy(calls, truth, resolution)

    total_alleles = sum(a.n_samples * 2 for a in per_locus.values())
    total_correct = sum(
        a.n_correct_both * 2 + a.n_correct_one for a in per_locus.values()
    )
    accuracy = total_correct / max(total_alleles, 1)

    discordant = []
    for locus, acc in per_locus.items():
        if acc.n_incorrect > 0 or acc.n_correct_one > 0:
            call = calls.get(locus, ("", ""))
            true = truth.get(locus, ("", ""))
            discordant.append({
                "locus": locus,
                "call": list(call),
                "truth": list(true),
                "match": compare_diploid(call, true, resolution),
            })

    return {
        "sample_id": sample_id,
        "accuracy": accuracy,
        "n_loci": len(per_locus),
        "n_correct_both": sum(a.n_correct_both for a in per_locus.values()),
        "n_correct_one": sum(a.n_correct_one for a in per_locus.values()),
        "n_incorrect": sum(a.n_incorrect for a in per_locus.values()),
        "n_no_call": sum(a.n_no_call for a in per_locus.values()),
        "discordant": discordant,
        "per_locus": {l: a.to_dict() for l, a in per_locus.items()},
    }


def cross_caller_concordance(
    unified_calls: dict[str, dict[str, tuple[str, str]]],
    published: dict[str, dict[str, dict[str, tuple[str, str]]]],
    truth: dict[str, dict[str, tuple[str, str]]],
    resolution: int = 2,
) -> dict:
    """Compare HLA-Unified accuracy against published callers.

    Only compares on loci that each caller actually reports, so callers
    like OptiType (Class I only) are not penalized for missing Class II.
    """
    caller_accuracies: dict[str, list[float]] = {"HLA-Unified": []}
    caller_loci_counts: dict[str, list[int]] = {"HLA-Unified": []}

    for sid, sid_truth in truth.items():
        # HLA-Unified accuracy (all loci)
        if sid in unified_calls:
            acc = compute_accuracy(unified_calls[sid], sid_truth, resolution)
            total = sum(a.n_samples * 2 for a in acc.values())
            correct = sum(a.n_correct_both * 2 + a.n_correct_one for a in acc.values())
            caller_accuracies["HLA-Unified"].append(correct / max(total, 1))
            caller_loci_counts["HLA-Unified"].append(len(acc))

        # Published callers — only compare on loci the caller reports
        for caller, caller_samples in published.items():
            if sid in caller_samples:
                caller_calls = caller_samples[sid]
                # Restrict truth to only loci this caller reports
                common_truth = {
                    l: t for l, t in sid_truth.items() if l in caller_calls
                }
                if not common_truth:
                    continue
                acc = compute_accuracy(caller_calls, common_truth, resolution)
                total = sum(a.n_samples * 2 for a in acc.values())
                correct = sum(a.n_correct_both * 2 + a.n_correct_one for a in acc.values())
                caller_accuracies.setdefault(caller, []).append(
                    correct / max(total, 1)
                )
                caller_loci_counts.setdefault(caller, []).append(len(acc))

    summary = {}
    for caller, accs in caller_accuracies.items():
        if accs:
            loci = caller_loci_counts.get(caller, [])
            avg_loci = sum(loci) / len(loci) if loci else 0
            summary[caller] = {
                "mean_accuracy": sum(accs) / len(accs),
                "n_samples": len(accs),
                "min_accuracy": min(accs),
                "max_accuracy": max(accs),
                "avg_loci_per_sample": round(avg_loci, 1),
            }

    return summary


def print_report(results: list[dict], concordance: dict, output_dir: Path) -> None:
    """Print and save the validation report."""
    print("\n" + "=" * 74)
    print("  HLA-Unified V2 — Real-Data Validation Report")
    print("=" * 74)

    # Per-sample results
    overall_correct = 0
    overall_total = 0

    for r in results:
        n_alleles = r["n_loci"] * 2
        n_correct = r["n_correct_both"] * 2 + r["n_correct_one"]
        overall_correct += n_correct
        overall_total += n_alleles

        status = "PASS" if r["accuracy"] >= 0.95 else "FAIL"
        print(f"\n  [{status}] {r['sample_id']}: {r['accuracy']:.1%} "
              f"({n_correct}/{n_alleles} alleles)")

        if r["discordant"]:
            for d in r["discordant"]:
                print(f"         HLA-{d['locus']}: "
                      f"call=({d['call'][0]}, {d['call'][1]}) "
                      f"truth=({d['truth'][0]}, {d['truth'][1]}) "
                      f"match={d['match']}")

    overall_acc = overall_correct / max(overall_total, 1)
    print(f"\n{'─' * 74}")
    print(f"  OVERALL: {overall_acc:.1%} ({overall_correct}/{overall_total} alleles)")
    threshold = "PASS" if overall_acc >= 0.95 else "FAIL"
    print(f"  Status:  {threshold} (threshold: >=95%)")

    # Cross-caller comparison
    if concordance:
        print(f"\n{'─' * 74}")
        print("  Cross-Caller Accuracy Comparison (2-field, common loci only)")
        print(f"  {'Caller':<20} {'Mean Acc':>10} {'Min':>8} {'Max':>8} {'Samples':>8} {'Loci':>6}")
        print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
        for caller in sorted(concordance, key=lambda c: concordance[c]["mean_accuracy"], reverse=True):
            c = concordance[caller]
            print(f"  {caller:<20} {c['mean_accuracy']:>9.1%} "
                  f"{c['min_accuracy']:>7.1%} {c['max_accuracy']:>7.1%} "
                  f"{c['n_samples']:>8} {c.get('avg_loci_per_sample', 0):>5.0f}")

    print(f"\n{'=' * 74}")

    # Save JSON report
    report = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "HLA-Unified V2",
        "data_source": "GIAB reference samples (real BAM data)",
        "resolution": 2,
        "overall_accuracy": round(overall_acc, 4),
        "overall_alleles_correct": overall_correct,
        "overall_alleles_total": overall_total,
        "pass_threshold": 0.95,
        "pass": overall_acc >= 0.95,
        "per_sample": results,
        "cross_caller_concordance": concordance,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "real_data_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved to: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-data validation for HLA-Unified V2",
    )
    parser.add_argument(
        "--samples", default="NA12878",
        help="Comma-separated sample IDs (default: NA12878)",
    )
    parser.add_argument(
        "--imgt-db", default="IMGTHLA",
        help="Path to IMGT/HLA database",
    )
    parser.add_argument(
        "--out", "-o", default="validation/real_data/results",
        help="Output directory",
    )
    parser.add_argument(
        "--bam-dir", default=None,
        help="Directory containing existing BAM files (skip download)",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Number of threads",
    )
    parser.add_argument(
        "--region-only", action="store_true",
        help="Download MHC region only (~100MB vs ~100GB per sample)",
    )
    parser.add_argument(
        "--skip-typing", action="store_true",
        help="Skip typing, evaluate existing results only",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory with pre-computed results (for --skip-typing)",
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=2,
        help="Comparison resolution (fields, default: 2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    samples = [s.strip() for s in args.samples.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bam_dir = Path(args.bam_dir) if args.bam_dir else out_dir / "bams"
    results_dir = Path(args.results_dir) if args.results_dir else out_dir

    # Load truth and published data
    truth = load_truth(TRUTH_TSV, samples)
    published = load_published_calls(PUBLISHED_DIR, samples)

    # Process each sample
    unified_calls: dict[str, dict[str, tuple[str, str]]] = {}
    sample_results: list[dict] = []

    for sid in samples:
        if sid not in truth:
            logger.warning("No truth data for %s, skipping", sid)
            continue

        if args.skip_typing:
            # Load pre-computed results
            result_tsv = results_dir / sid / "hla_types.tsv"
            calls = load_results(result_tsv)
            if not calls:
                logger.warning("No results found for %s at %s", sid, result_tsv)
                continue
        else:
            # Download BAM if needed
            bam_path = bam_dir / f"{sid}.bam"
            if not bam_path.exists():
                try:
                    bam_path = download_bam(sid, bam_dir, region_only=args.region_only)
                except Exception as e:
                    logger.error("Failed to download BAM for %s: %s", sid, e)
                    continue

            # Run typing
            calls = run_typing(sid, bam_path, args.imgt_db, results_dir, args.threads)
            if not calls:
                logger.error("Typing produced no results for %s", sid)
                continue

        unified_calls[sid] = calls

        # Validate against truth
        result = validate_against_truth(
            calls, truth[sid], sid, args.resolution,
        )
        sample_results.append(result)

    # Cross-caller concordance
    concordance = cross_caller_concordance(
        unified_calls, published, truth, args.resolution,
    )

    # Print and save report
    print_report(sample_results, concordance, out_dir)

    # Return non-zero if below threshold
    if sample_results:
        overall_correct = sum(r["n_correct_both"] * 2 + r["n_correct_one"] for r in sample_results)
        overall_total = sum(r["n_loci"] * 2 for r in sample_results)
        if overall_correct / max(overall_total, 1) < 0.95:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
