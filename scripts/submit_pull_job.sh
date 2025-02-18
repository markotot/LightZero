#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=1G
#$ -l h_rt=1:0:0
#$ -l rocky
#$ -cwd
#$ -j y
#$ -o job_results

APPTAINERENV_NSLOTS=${NSLOTS}
apptainer run --nv --env-file myenvs --env "JOB_TYPE=$JOB_TYPE" containers/iris.sif