# Description: Run the evaluation of the Iris model on the Atari environment

#cd zoo/atari/entry/
NUM_SEEDS=$1
ENV_NAME=$2
WANDB_API_KEY=$3
wandb login $WANDB_API_KEY

#export PYTHONPATH="/data/home/acw549/LightZero:$PYTHONPATH"

echo "Evaluating Iris model on $NUM_SEEDS seeds on environment $ENV_NAME"

for ((i=0; i<NUM_SEEDS; i++)); do
    echo "Evaluating seed $i on environment $ENV_NAME"
    python3 -m zoo.atari.entry.atari_eval_iris_model $i $ENV_NAME
done

wait
echo "All seeds have been evaluated"