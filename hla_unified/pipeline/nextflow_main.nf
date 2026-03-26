#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * HLA-Unified V2: Multi-strategy HLA typing pipeline (Nextflow DSL2)
 *
 * Supports:
 *   - Single sample (--bam) or batch mode (--sample_sheet)
 *   - All 7 assay types and 3 use-case profiles
 *   - V2 features: ambiguity classification, CNV, contamination, clinical
 *   - Optional benchmarking against truth set
 */

// === Parameters ===
params.bam             = null
params.r1              = null
params.r2              = null
params.sample_sheet    = null       // CSV: sample_id,bam_path,data_type
params.imgt_db         = null
params.outdir          = './results'
params.threads         = 4
params.reference       = null
params.loci            = 'A,B,C,DRB1,DQA1,DQB1,DPA1,DPB1'
params.data_type       = 'short'
params.output_resolution = 'max'
params.imgt_release    = null
params.profile_name    = null       // transplant, research, immuno_onc
params.clinical        = false
params.strict          = false      // --strict-reproducibility
params.skip_refinement = false
params.skip_assembly   = false
params.skip_kmer       = false
params.skip_confidence = false

// Benchmarking
params.truth_set       = null       // TSV for benchmark validation

// === Processes ===

process HLA_TYPE {
    tag "${sample_id}"
    cpus params.threads
    memory { params.data_type in ['pacbio','hifi','ont'] ? '16 GB' : '8 GB' }
    errorStrategy 'retry'
    maxRetries 2

    publishDir "${params.outdir}/${sample_id}", mode: 'copy'

    input:
    tuple val(sample_id), path(input_bam), val(data_type)

    output:
    tuple val(sample_id), path("hla_types.tsv"),  emit: tsv
    tuple val(sample_id), path("hla_types.json"), emit: json
    tuple val(sample_id), path("ambiguity.tsv"),  emit: ambiguity
    tuple val(sample_id), path("qc_report.json"), emit: qc_json
    tuple val(sample_id), path("qc_dashboard.html"), emit: qc_html
    tuple val(sample_id), path("clinical_summary.*"), emit: clinical, optional: true
    tuple val(sample_id), path("manifest.json"),  emit: manifest, optional: true
    tuple val(sample_id), path("novel_alleles.json"), emit: novel, optional: true

    script:
    def profile_flag = params.profile_name ? "--profile ${params.profile_name}" : ""
    def clinical_flag = params.clinical ? "--clinical" : ""
    def strict_flag = params.strict ? "--strict-reproducibility" : ""
    def ref_flag = params.reference ? "--reference ${params.reference}" : ""
    def imgt_flag = params.imgt_release ? "--imgt-release ${params.imgt_release}" : ""
    def skip_flags = [
        params.skip_refinement ? "--skip-refinement" : "",
        params.skip_confidence ? "--skip-confidence" : "",
        params.skip_kmer       ? "--skip-kmer"       : "",
        params.skip_assembly   ? "--skip-assembly"   : "",
    ].findAll { it }.join(" ")

    """
    python -m hla_unified type \\
        --bam ${input_bam} \\
        --imgt-db ${params.imgt_db} \\
        --out . \\
        --threads ${task.cpus} \\
        --data-type ${data_type} \\
        --loci ${params.loci} \\
        --output-resolution ${params.output_resolution} \\
        ${profile_flag} ${clinical_flag} ${strict_flag} \\
        ${ref_flag} ${imgt_flag} ${skip_flags} \\
        -vv
    """
}

process BENCHMARK {
    cpus params.threads
    memory '4 GB'

    publishDir "${params.outdir}/benchmark", mode: 'copy'

    input:
    path truth_set
    path results_dir

    output:
    path "benchmark_*.json", emit: report

    script:
    """
    python -m hla_unified benchmark \\
        --dataset ${truth_set} \\
        --bam-dir . \\
        --imgt-db ${params.imgt_db} \\
        --out . \\
        --threads ${task.cpus} \\
        --data-type ${params.data_type} \\
        --skip-typing \\
        --results-dir ${results_dir}
    """
}

// === Workflows ===

workflow SINGLE_SAMPLE {
    take:
    sample_ch  // tuple(sample_id, bam_path, data_type)

    main:
    HLA_TYPE(sample_ch)

    emit:
    tsv   = HLA_TYPE.out.tsv
    json  = HLA_TYPE.out.json
}

workflow BATCH {
    take:
    sheet_ch  // channel from sample sheet

    main:
    HLA_TYPE(sheet_ch)

    emit:
    tsv  = HLA_TYPE.out.tsv
    json = HLA_TYPE.out.json
}

// === Entry Point ===

workflow {
    if (params.sample_sheet) {
        // Batch mode: read sample sheet CSV
        Channel
            .fromPath(params.sample_sheet)
            .splitCsv(header: true)
            .map { row -> tuple(row.sample_id, file(row.bam_path), row.data_type ?: params.data_type) }
            .set { samples_ch }

        BATCH(samples_ch)

    } else if (params.bam) {
        // Single sample mode
        def sample_id = file(params.bam).getSimpleName()
        Channel
            .of(tuple(sample_id, file(params.bam), params.data_type))
            .set { single_ch }

        SINGLE_SAMPLE(single_ch)

    } else {
        error "Provide --bam for single sample or --sample_sheet for batch mode"
    }

    // Optional benchmarking
    if (params.truth_set) {
        BENCHMARK(
            file(params.truth_set),
            file(params.outdir),
        )
    }
}
