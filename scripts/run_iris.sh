#!/bin/bash

# Description: Run the evaluation of the Iris model on the Atari environment
cd zoo/atari/entry/
alias python3='/home/marko/miniconda3/envs/iris/bin/python'
NUM_SEEDS=$1
ENV_NAME=$2

echo "$PWD"
echo "Evaluating Iris model on $NUM_SEEDS seeds on environment $ENV_NAME"

for ((i=0; i<NUM_SEEDS; i++)); do
    echo "Evaluating seed $i on environment $ENV_NAME"
    python3 -m atari_eval_iris_model $i $ENV_NAME &
done

wait
echo "All seeds have been evaluated"