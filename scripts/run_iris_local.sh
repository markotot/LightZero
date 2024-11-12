# Description: Run the evaluation of the Iris model on the Atari environment


source home/marko/miniconda3/etc/profile.d/conda.sh
conda activate iris
python3 --version

cd zoo/atari/entry/
NUM_SEEDS=$1
ENV_NAME=$2

echo "Evaluating Iris model on $NUM_SEEDS seeds on environment $ENV_NAME"

for ((i=0; i<NUM_SEEDS; i++)); do
    echo "Evaluating seed $i on environment $ENV_NAME"
    python3 -m atari_eval_iris_model $i $ENV_NAME &
done

wait
echo "All seeds have been evaluated"