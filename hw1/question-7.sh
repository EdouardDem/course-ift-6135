#! /bin/bash

set -e

function run_experiment() {
    echo "Running experiment with mlp_ratio_tokens $1, mlp_ratio_channels $2. Saving to mlpmixer-$3"
    python main.py \
        --model mlpmixer \
        --mlp_ratio_tokens $1 \
        --mlp_ratio_channels $2 \
        --visualize \
        --epochs 40 \
        --model_config ./model_configs/mlpmixer.json \
        --logdir ./logs/question-7/mlpmixer-$3
}

# Grid search with :
# - mlp_ratio_tokens: 0.25, 0.5, 1
# - mlp_ratio_channels: 1, 2, 4
# Use loop to run all combinations
for tokens_ratio in 0.25 0.5 1; do
    for channels_ratio in 1 2 4; do
        # Créer un nom unique pour l'expérience
        name="t${tokens_ratio//./}_c${channels_ratio}"
        run_experiment $tokens_ratio $channels_ratio $name
    done
done 