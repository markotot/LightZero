#!/bin/bash
#$ -pe smp 12
#$ -l h_vmem=6G
#$ -l gpu=1
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -o job_results

module load python/3.10.14

APPTAINERENV_NSLOTS=${NSLOTS}
nvidia-smi