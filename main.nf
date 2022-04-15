println "Occupancy Analysis Pipeline     "
println "================================="
println "Image            : ${params.image}"
println "Platform          : ${params.platform}"
println "Data              : ${params.data}"
println "Slide             : ${params.slide}"
println "Cycle start       : ${params.start}"
println "Cycle range       : ${params.range}"
println "Output            : ${params.output}"


ch = Channel.of( 'L01', 'L02', 'L03', 'L04' )
//ch = Channel.of( 'L01')
params.collect_dir = '/hwfssz8/MGI_CG_SZ/DATA/occu'

process occupancy_analysis {
    publishDir params.collect_dir, mode: 'copy' 

    input:
    each lane from ch
    

    output:
    val params.slide into occu_out

    //path("${params.output}/*_Summary.xlsx") into summary_ch


    //python /hwfssz8/MGI_BCC/USER/huangsixing/occupancy_analysis/occupancy_chip_wrapper.py -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range}

    ///share/app/singularity/3.8.1/bin/singularity exec -B /hwfssz8/MGI_BCC/USER/huangsixing/  ./occu.sif python /app/occupancy_chip_wrapper.py -d /hwfssz8/MGI_BCC/USER/huangsixing/Javier/V300098092_lite1.1 -l L01 -o /hwfssz8/MGI_BCC/USER/huangsixing/test -s V300098092 -c 1 -r 8

    //python ${params.script} -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range}


    script:
    """
    /share/app/singularity/3.8.1/bin/singularity exec -B $HOME  ${params.image} python /app/occupancy_chip_wrapper.py -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range} -p ${params.platform}

    """



    ///share/app/singularity/3.8.1/bin/singularity exec -B $HOME  ${params.image} python /app/occupancy_chip_wrapper.py -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range} -p ${params.platform}

    //"""
    //python /hwfssz8/MGI_CG_SZ/USER/huangsixing/nextflow/occupancy_analysis_chip_singu/test.py ${params.output}/${params.slide}/Lite/
    //"""

}

process copy_files {
    errorStrategy 'ignore'
    input:
    val slide from occu_out

    output:
    val slide into copy_out

    shell:
    """
    mkdir -p ${params.collect_dir}/\$(whoami)/${slide}
    cp ${params.output}/${slide}/Lite/*_Summary.xlsx ${params.collect_dir}/\$(whoami)/${slide}/ & cp ${params.output}/${slide}/v2/*_Summary.xlsx ${params.collect_dir}/\$(whoami)/${slide}/
    """
}
