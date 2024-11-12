#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=1G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -o job_results

module load python/3.10.14

# Replace the following line with a program or command
APPTAINERENV_NSLOTS=${NSLOTS}

apptainer run --nv --env-file myenvs --env "JOB_TYPE=$JOB_TYPE" containers/iris.sif