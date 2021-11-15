# Occupancy Analysis

This Readme file describes the steps for a deployment and execution of the CG's Occupancy Analysis (Occu) on a new infrastructure.


# Prerequisite

1. Singularity (>3.8.1). This container engine makes occu executable.
2. Nexflow (>21.04.3.5563). This workflow engine allows the occu to run locally, on SGE or on the Cloud.
3. Username and password for  the **bgisixing** docker register (Contact Sixing Huang).
# Steps

1. Build the sif image:
```console
singularity build --docker-login ./occu.sif  docker://bgisixing/occupancyanalysisdocker
```
You will need to enter the username and password for the register.

2. Git clone the nextflow repo
```console
git clone https://github.com/dgg32/occupancy_analysis_chip_singu
```
3. Modify the SGE settings in your nextflow.config so that occu can run on your local SGE
4. Run occu within the nextflow folder:
```console
nextflow run main.nf --image [path to occu.sif] --data [path to slide folder]  --output [path to an empty output folder] --slide [slide name] --start [start cycle] --range [cycle range] -profile sge
```
## Authors

*  **Sixing Huang** - *Concept and Coding*

