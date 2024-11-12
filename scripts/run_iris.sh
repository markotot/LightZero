# Description: Run the evaluation of the Iris model on the Atari environment
cd zoo/atari/entry/

set NUM_SEEDS [lindex $argv 0]
set ENV_NAME [lindex $argv 1]

for ((i=0; i<=NUM_SEEDS; i++)); do
    python3 -m atari_eval_iris_model $i $environment &
done

wait
echo "All seeds have been evaluated"