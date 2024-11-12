#!/bin/bash
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -o job_results



module load python/3.10.14

APPTAINERENV_NSLOTS=${NSLOTS}
apptainer run --nv --env-file myenvs --env "JOB_TYPE=\"pull_git\"" containers/iris.sif