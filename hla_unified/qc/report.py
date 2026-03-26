"""Transparent QC: locus-level evidence dashboard and reporting.

Generates structured JSON QC report and standalone HTML dashboard
showing per-locus evidence, phase diagnostics, and confidence metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from ..pipeline.runner import PipelineResult, LocusCall

logger = logging.getLogger(__name__)


@dataclass
class LocusQC:
    """QC evidence for a single locus."""
    locus: str
    # Calls
    allele1: str
    allele2: str
    g_group1: str
    g_group2: str
    confidence: str
    posterior: float
    # Read evidence
    total_reads: int
    reads_explained: int
    reads_unexplained: int
    read_fraction_explained: float
    # K-mer evidence
    kmer_coverage: float
    kmer_concordant: bool
    kmer_flags: list[str]
    # Phasing evidence
    is_phased: bool
    n_het_sites: int
    phase_confidence: float
    # Novel allele
    is_novel: bool
    novel_summary: str
    # Flags
    all_flags: list[str]
    # Verdict
    pass_qc: bool
    # V2: Detailed metrics (haplotype balance, informative positions, assembly)
    haplotype_balance: float = 0.0
    allele1_depth: float = 0.0
    allele2_depth: float = 0.0
    n_informative_positions: int = 0
    assembly_n50: int = 0
    assembly_gene_coverage: float = 0.0
    qc_issues: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    """Complete QC report for a pipeline run."""
    # Metadata
    run_timestamp: str
    hla_unified_version: str
    imgt_release: str
    imgt_commit: str
    data_type: str
    input_file: str
    # Global metrics
    total_loci: int
    loci_passing_qc: int
    loci_high_confidence: int
    loci_medium_confidence: int
    loci_low_confidence: int
    # Per-locus QC
    locus_qc: dict[str, LocusQC]
    # Phase summary
    phases_completed: list[str]
    runtime_seconds: float


def generate_qc_report(
    result: PipelineResult,
    provenance: dict,
    data_type: str = "short",
    input_file: str = "",
    phasing_results: dict | None = None,
    kmer_results: dict | None = None,
    assembly_results: dict | None = None,
    detailed_metrics: dict | None = None,
) -> QCReport:
    """Generate a structured QC report from pipeline results."""
    from .. import __version__

    locus_qc_map: dict[str, LocusQC] = {}

    for locus, call in result.calls.items():
        # Read evidence
        unexplained = call.total_reads - call.reads_explained
        frac = call.reads_explained / max(call.total_reads, 1)

        # K-mer evidence
        kmer_flags = []
        if kmer_results and locus in kmer_results:
            kr = kmer_results[locus]
            kmer_flags = kr.flags if hasattr(kr, "flags") else []

        # Phasing
        is_phased = False
        n_het = 0
        phase_conf = 0.0
        if phasing_results and locus in phasing_results:
            pr = phasing_results[locus]
            is_phased = pr.is_phased
            n_het = pr.n_het_sites
            phase_conf = pr.phase_confidence

        # Novel allele
        novel_summary = ""
        if assembly_results and locus in assembly_results:
            ar = assembly_results[locus]
            if ar.novel_report:
                novel_summary = ar.novel_report.summary

        # V2: Merge detailed metrics if available
        hap_balance = 0.0
        a1_depth = 0.0
        a2_depth = 0.0
        n_info_pos = 0
        asm_n50 = 0
        asm_gene_cov = 0.0
        qc_issues: list[str] = []
        if detailed_metrics and locus in detailed_metrics:
            dm = detailed_metrics[locus]
            hap_balance = dm.haplotype_balance
            a1_depth = dm.allele1_depth
            a2_depth = dm.allele2_depth
            n_info_pos = dm.n_informative_positions
            asm_n50 = dm.assembly_n50
            asm_gene_cov = dm.assembly_gene_coverage
            qc_issues = dm.qc_issues

        # QC verdict
        pass_qc = (
            call.confidence in ("HIGH", "MEDIUM")
            and frac >= 0.5
            and call.kmer_concordant
            and len(qc_issues) == 0
        )

        locus_qc_map[locus] = LocusQC(
            locus=locus,
            allele1=call.allele1,
            allele2=call.allele2,
            g_group1=call.g_group1,
            g_group2=call.g_group2,
            confidence=call.confidence,
            posterior=call.posterior,
            total_reads=call.total_reads,
            reads_explained=call.reads_explained,
            reads_unexplained=unexplained,
            read_fraction_explained=round(frac, 4),
            kmer_coverage=call.kmer_covered,
            kmer_concordant=call.kmer_concordant,
            kmer_flags=kmer_flags,
            is_phased=is_phased,
            n_het_sites=n_het,
            phase_confidence=round(phase_conf, 4),
            is_novel=call.is_novel,
            novel_summary=novel_summary,
            all_flags=call.flags,
            pass_qc=pass_qc,
            haplotype_balance=round(hap_balance, 4),
            allele1_depth=round(a1_depth, 1),
            allele2_depth=round(a2_depth, 1),
            n_informative_positions=n_info_pos,
            assembly_n50=asm_n50,
            assembly_gene_coverage=round(asm_gene_cov, 4),
            qc_issues=qc_issues,
        )

    n_high = sum(1 for q in locus_qc_map.values() if q.confidence == "HIGH")
    n_med = sum(1 for q in locus_qc_map.values() if q.confidence == "MEDIUM")
    n_low = sum(1 for q in locus_qc_map.values() if q.confidence == "LOW")
    n_pass = sum(1 for q in locus_qc_map.values() if q.pass_qc)

    return QCReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        hla_unified_version=__version__,
        imgt_release=provenance.get("release", "unknown"),
        imgt_commit=provenance.get("git_commit", "unknown"),
        data_type=data_type,
        input_file=input_file,
        total_loci=len(locus_qc_map),
        loci_passing_qc=n_pass,
        loci_high_confidence=n_high,
        loci_medium_confidence=n_med,
        loci_low_confidence=n_low,
        locus_qc=locus_qc_map,
        phases_completed=result.phases_completed,
        runtime_seconds=result.runtime_seconds,
    )


def write_qc_json(report: QCReport, path: Path) -> None:
    """Write QC report as structured JSON."""
    data = {
        "metadata": {
            "run_timestamp": report.run_timestamp,
            "hla_unified_version": report.hla_unified_version,
            "imgt_release": report.imgt_release,
            "imgt_commit": report.imgt_commit,
            "data_type": report.data_type,
            "input_file": report.input_file,
        },
        "summary": {
            "total_loci": report.total_loci,
            "loci_passing_qc": report.loci_passing_qc,
            "loci_high_confidence": report.loci_high_confidence,
            "loci_medium_confidence": report.loci_medium_confidence,
            "loci_low_confidence": report.loci_low_confidence,
            "runtime_seconds": round(report.runtime_seconds, 1),
            "phases_completed": report.phases_completed,
        },
        "loci": {
            locus: {
                **asdict(qc),
                # V2 detailed metrics grouped for readability
                "detailed_qc": {
                    "haplotype_balance": qc.haplotype_balance,
                    "allele1_depth": qc.allele1_depth,
                    "allele2_depth": qc.allele2_depth,
                    "n_informative_positions": qc.n_informative_positions,
                    "assembly_n50": qc.assembly_n50,
                    "assembly_gene_coverage": qc.assembly_gene_coverage,
                    "qc_issues": qc.qc_issues,
                },
            }
            for locus, qc in report.locus_qc.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("QC JSON written to %s", path)


def write_qc_html(report: QCReport, path: Path) -> None:
    """Generate standalone HTML QC dashboard."""

    def _badge(text: str, color: str) -> str:
        return (
            f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:0.85em">{text}</span>'
        )

    def _conf_badge(conf: str) -> str:
        colors = {"HIGH": "#2e7d32", "MEDIUM": "#f57f17", "LOW": "#c62828"}
        return _badge(conf, colors.get(conf, "#666"))

    def _bool_badge(val: bool, true_text: str = "YES", false_text: str = "NO") -> str:
        return _badge(true_text, "#2e7d32") if val else _badge(false_text, "#c62828")

    # Build locus rows
    rows = []
    for locus in sorted(report.locus_qc.keys()):
        q = report.locus_qc[locus]
        flags_html = ", ".join(f"<code>{f}</code>" for f in q.all_flags) if q.all_flags else "&mdash;"
        novel_html = f'<span title="{q.novel_summary}">{_badge("NOVEL", "#6a1b9a")}</span>' if q.is_novel else ""

        balance_html = f"{q.haplotype_balance:.2f}" if q.haplotype_balance > 0 else "&mdash;"
        issues_html = ", ".join(f"<code>{i}</code>" for i in q.qc_issues) if q.qc_issues else "&mdash;"

        rows.append(f"""
        <tr>
            <td><strong>HLA-{q.locus}</strong></td>
            <td>{q.allele1}</td>
            <td>{q.allele2}</td>
            <td>{q.g_group1}</td>
            <td>{q.g_group2}</td>
            <td>{_conf_badge(q.confidence)}</td>
            <td>{q.posterior:.3f}</td>
            <td>{q.reads_explained}/{q.total_reads} ({q.read_fraction_explained:.0%})</td>
            <td>{balance_html}</td>
            <td>{q.kmer_coverage:.2%}</td>
            <td>{_bool_badge(q.kmer_concordant)}</td>
            <td>{_bool_badge(q.is_phased, "PHASED", "UNPHASED")} ({q.n_het_sites} het, {q.n_informative_positions} info)</td>
            <td>{_bool_badge(q.pass_qc, "PASS", "FAIL")}</td>
            <td>{novel_html}</td>
            <td style="font-size:0.8em">{flags_html}</td>
            <td style="font-size:0.75em">{issues_html}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>HLA-Unified QC Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 2em; background: #fafafa; }}
  h1 {{ color: #1565c0; }}
  .meta {{ background: #e3f2fd; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; }}
  .meta dt {{ font-weight: bold; display: inline; }}
  .meta dd {{ display: inline; margin-right: 2em; }}
  .summary {{ display: flex; gap: 1em; margin-bottom: 1.5em; }}
  .summary .card {{ background: #fff; padding: 1em 1.5em; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: center; }}
  .summary .card .number {{ font-size: 2em; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  th {{ background: #1565c0; color: #fff; padding: 8px 12px; text-align: left; font-size: 0.85em; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; font-size: 0.9em; }}
  tr:hover {{ background: #f5f5f5; }}
  .footer {{ margin-top: 2em; color: #888; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>HLA-Unified QC Dashboard</h1>

<div class="meta">
  <dl>
    <dt>IPD-IMGT/HLA Release:</dt><dd><strong>{report.imgt_release}</strong></dd>
    <dt>IMGT Commit:</dt><dd><code>{report.imgt_commit[:12]}</code></dd>
    <dt>Assay Type:</dt><dd>{report.data_type}</dd>
    <dt>Input:</dt><dd>{report.input_file}</dd>
    <dt>Runtime:</dt><dd>{report.runtime_seconds:.1f}s</dd>
    <dt>Phases:</dt><dd>{' &rarr; '.join(report.phases_completed)}</dd>
  </dl>
</div>

<div class="summary">
  <div class="card">
    <div class="number" style="color:#2e7d32">{report.loci_passing_qc}/{report.total_loci}</div>
    <div>Loci Passing QC</div>
  </div>
  <div class="card">
    <div class="number" style="color:#2e7d32">{report.loci_high_confidence}</div>
    <div>HIGH Confidence</div>
  </div>
  <div class="card">
    <div class="number" style="color:#f57f17">{report.loci_medium_confidence}</div>
    <div>MEDIUM Confidence</div>
  </div>
  <div class="card">
    <div class="number" style="color:#c62828">{report.loci_low_confidence}</div>
    <div>LOW Confidence</div>
  </div>
</div>

<h2>Per-Locus Evidence</h2>
<table>
<thead>
<tr>
  <th>Locus</th>
  <th>Allele 1</th>
  <th>Allele 2</th>
  <th>G-Group 1</th>
  <th>G-Group 2</th>
  <th>Confidence</th>
  <th>Posterior</th>
  <th>Reads</th>
  <th>Hap Bal</th>
  <th>K-mer Cov</th>
  <th>K-mer OK</th>
  <th>Phasing</th>
  <th>QC</th>
  <th>Novel</th>
  <th>Flags</th>
  <th>Issues</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>

<div class="footer">
  Generated by HLA-Unified v{report.hla_unified_version} at {report.run_timestamp}<br>
  Database: IPD-IMGT/HLA {report.imgt_release} (commit {report.imgt_commit[:12]})
</div>
</body>
</html>"""

    path.write_text(html)
    logger.info("QC HTML dashboard written to %s", path)
