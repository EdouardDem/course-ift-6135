#! /bin/bash

set -e

function run_experiment() {
    echo "Running experiment with optimizer $1, lr $2, weight decay $3. Saving to resnet18-$4"
    python main.py \
        --model resnet18 \
        --optimizer $1 \
        --lr $2 \
        --weight_decay $3 \
        --visualize \
        --model_config ./model_configs/resnet18.json \
        --logdir ./logs/question-5/resnet18-$4
}

# Grid search with :
# - optimizers: adam, sgd, momentum, adamw
# - lrs: 0.001, 0.0001, 0.00001
# - weight decays: 0.0005, 0.001, 0.005, 0.01
# Use loop to run all combinations
for optimizer in adam sgd momentum adamw; do
    for lr in 0.001 0.0001 0.00001; do
        for weight_decay in 0.0005 0.001 0.005 0.01; do
            name="${optimizer}_${lr}_${weight_decay}"
            run_experiment $optimizer $lr $weight_decay $name
        done
    done
done
