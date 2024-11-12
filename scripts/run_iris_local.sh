#!/bin/bash
# Description: Run the evaluation of the Iris model on the Atari environment

source /home/marko/miniconda3/etc/profile.d/conda.sh
conda activate iris

NUM_SEEDS=$1
ENV_NAME=$2
WANDB_API_KEY=$3

wandb login $WANDB_API_KEY

echo "Evaluating Iris model on $NUM_SEEDS seeds on environment $ENV_NAME"
cd zoo/atari/entry/
for ((i=0; i<NUM_SEEDS; i++)); do
    python3 -m atari_eval_iris_model $i $ENV_NAME &
done
wait
