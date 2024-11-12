#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=1G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -o job_results

# Replace the following line with a program or command
APPTAINERENV_NSLOTS=${NSLOTS}
apptainer build --force containers/iris.sif $PROJECT_NAME/apptainer/iris.def;
