#!/bin/bash -ue
/share/app/singularity/3.8.1/bin/singularity exec -B /hwfssz8/MGI_CG_SZ/USER/huangsixing/  /hwfssz8/MGI_CG_SZ/USER/huangsixing/sif/occu.sif python /app/occupancy_chip_wrapper.py -d /hwfssz8/MGI_CG_SZ/USER/huangsixing/test/IntData/V350037463/ -l L01 -o /hwfssz8/MGI_CG_SZ/USER/huangsixing/test -s V350037463 -c 1 -r 4 -p v2
