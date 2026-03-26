#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
 * HLA-LA: HLA typing from NGS data using a population reference graph
 * Nextflow DSL2 pipeline
 */

params.bam         = null
params.graph_dir   = null
params.sample_id   = "sample"
params.long_reads  = ""
params.threads     = 4
params.outdir      = "results"

// Validate required params
if (!params.bam) error "Please specify --bam"
if (!params.graph_dir) error "Please specify --graph_dir"


process DETECT_REFERENCE {
    tag "${params.sample_id}"

    input:
    path bam
    path bam_index
    path graph_dir

    output:
    path "ref_info.json"

    script:
    """
    hlala detect-reference \
        --bam ${bam} \
        --graph-dir ${graph_dir} \
        --output ref_info.json
    """
}

process EXTRACT_READS {
    tag "${params.sample_id}"
    cpus params.threads

    input:
    path bam
    path bam_index
    path ref_info

    output:
    path "extraction.bam"
    path "extraction.bam.bai"

    script:
    """
    hlala extract-reads \
        --bam ${bam} \
        --ref-info ${ref_info} \
        --output extraction.bam \
        --threads ${task.cpus}
    """
}

process BAM_TO_FASTQ {
    tag "${params.sample_id}"

    input:
    path extracted_bam
    path extracted_bai

    output:
    path "fastq/R_1.fastq"
    path "fastq/R_2.fastq"
    path "fastq/R_U.fastq"

    script:
    def lr_flag = params.long_reads ? "--long-reads ${params.long_reads}" : ""
    """
    hlala bam-to-fastq \
        --bam ${extracted_bam} \
        --output-dir fastq \
        ${lr_flag}
    """
}

process REMAP_TO_PRG {
    tag "${params.sample_id}"
    cpus params.threads

    input:
    path fastq1
    path fastq2
    path fastq_u
    path graph_dir

    output:
    path "remapped.bam"
    path "remapped.bam.bai"

    script:
    def ref = "${graph_dir}/extendedReferenceGenome/extendedReferenceGenome.fa"
    if (params.long_reads)
        """
        minimap2 -a -x map-${params.long_reads == 'ont2d' ? 'ont' : 'pb'} \
            -t ${task.cpus} ${ref} ${fastq_u} \
            | samtools sort -@ ${task.cpus} -o remapped.bam -
        samtools index remapped.bam
        """
    else
        """
        bwa mem -t ${task.cpus} ${ref} ${fastq1} ${fastq2} \
            | samtools sort -@ ${task.cpus} -o remapped.bam -
        samtools index remapped.bam
        """
}

process ALIGN_AND_TYPE {
    tag "${params.sample_id}"
    cpus params.threads
    publishDir "${params.outdir}/${params.sample_id}", mode: 'copy'

    input:
    path remapped_bam
    path remapped_bai
    path graph_dir

    output:
    path "typing/hla_types.txt"
    path "typing/"

    script:
    def lr_flag = params.long_reads ? "--long-reads ${params.long_reads}" : ""
    """
    hlala align-and-type \
        --bam ${remapped_bam} \
        --graph-dir ${graph_dir} \
        --sample-id ${params.sample_id} \
        --output-dir typing \
        --threads ${task.cpus} \
        ${lr_flag}
    """
}

process TRANSLATE_G_GROUPS {
    tag "${params.sample_id}"
    publishDir "${params.outdir}/${params.sample_id}", mode: 'copy'

    input:
    path hla_types
    path graph_dir

    output:
    path "hla_types_G.txt"

    script:
    """
    hlala translate-g-groups \
        --input ${hla_types} \
        --graph-dir ${graph_dir} \
        --output hla_types_G.txt
    """
}


workflow {
    // Input channels
    bam_ch   = Channel.fromPath(params.bam, checkIfExists: true)
    bai_ch   = Channel.fromPath("${params.bam}.bai", checkIfExists: false)
                .ifEmpty(Channel.fromPath("${params.bam}.crai", checkIfExists: false))
    graph_ch = Channel.fromPath(params.graph_dir, checkIfExists: true, type: 'dir')

    // Pipeline
    ref_info = DETECT_REFERENCE(bam_ch, bai_ch, graph_ch)

    (extracted_bam, extracted_bai) = EXTRACT_READS(bam_ch, bai_ch, ref_info)

    (fq1, fq2, fqu) = BAM_TO_FASTQ(extracted_bam, extracted_bai)

    (remapped_bam, remapped_bai) = REMAP_TO_PRG(fq1, fq2, fqu, graph_ch)

    (hla_types, typing_dir) = ALIGN_AND_TYPE(remapped_bam, remapped_bai, graph_ch)

    TRANSLATE_G_GROUPS(hla_types, graph_ch)
}
