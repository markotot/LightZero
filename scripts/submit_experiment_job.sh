#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -cwd
#$ -j y
#$ -o job_results

APPTAINERENV_NSLOTS=${NSLOTS}
apptainer run --nv --env-file myenvs --env "JOB_TYPE=$JOB_TYPE,RUN_NAME=$RUN_NAME,NUM_SEEDS=$NUM_SEEDS,ENV_NAME=$ENV_NAME,WANDB_API_KEY=$WANDB_API_KEY" containers/iris.sif
