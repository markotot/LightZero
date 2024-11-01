# Description: Run the evaluation of the Iris model on the Atari environment
cd zoo/atari/entry/

seeds=("0" "1" "2" "3" "4" "5")

for seed in "${seeds[@]}"; do
    python3 -m atari_eval_iris_model "$seed" &
done
