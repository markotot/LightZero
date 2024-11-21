# Description: Run the evaluation of the Iris model on the Atari environment

cd zoo/atari/entry/
START_SEED=$1
END_SEED=$2
ENV_NAME=$3
WANDB_API_KEY=$4
wandb login $WANDB_API_KEY

#export PYTHONPATH="/home/marko/pythonProject/LightZero:$PYTHONPATH"

echo "Evaluating Iris model on environment $ENV_NAME"

for ((i=START_SEED; i<END_SEED; i++)); do
    echo "Evaluating seed $i on environment $ENV_NAME"
    python3 -m atari_eval_iris_model $i $ENV_NAME &
done

wait
echo "All seeds have been evaluated"