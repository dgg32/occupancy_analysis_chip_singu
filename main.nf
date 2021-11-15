println "Occupancy Analysis Pipeline     "
println "================================="
println "Image            : ${params.image}"
println "Platform            : ${params.platform}"
//println "Script            : ${params.script}"
println "Data              : ${params.data}"
println "Slide             : ${params.slide}"
println "Cycle start       : ${params.start}"
println "Cycle range       : ${params.range}"
println "Output            : ${params.output}"


ch = Channel.of( 'L01', 'L02', 'L03', 'L04' )


process occupancy_analysis {

    input:
    each lane from ch

    //python /hwfssz8/MGI_BCC/USER/huangsixing/occupancy_analysis/occupancy_chip_wrapper.py -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range}

    ///share/app/singularity/3.8.1/bin/singularity exec -B /hwfssz8/MGI_BCC/USER/huangsixing/  ./occu.sif python /app/occupancy_chip_wrapper.py -d /hwfssz8/MGI_BCC/USER/huangsixing/Javier/V300098092_lite1.1 -l L01 -o /hwfssz8/MGI_BCC/USER/huangsixing/test -s V300098092 -c 1 -r 8

    //python ${params.script} -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range}


    script:
    """
    /share/app/singularity/3.8.1/bin/singularity exec -B $HOME  ${params.image} python /app/occupancy_chip_wrapper.py -d ${params.data} -l ${lane} -o ${params.output} -s ${params.slide} -c ${params.start} -r ${params.range} -p ${params.platform}
    """
}