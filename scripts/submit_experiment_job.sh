#!/bin/bash
#$ -pe smp 2
#$ -l h_vmem=8G
#$ -l gpu=1
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -o job_results

module load python/3.10.14

apptainer run --env-file myenvs --env "JOB_TYPE=$JOB_TYPE,RUN_NAME=$RUN_NAME,START_ID=$START_ID,END_ID=$END_ID,SEED=$SGE_TASK_ID" containers/iris.sif
