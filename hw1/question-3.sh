#! /bin/bash

set -e

function run_experiment() {
    echo "Running experiment with lr $1"
    python main.py --model resnet18 --optimizer adam --model_config ./model_configs/resnet18.json --lr $1 --visualize --epochs 40 --logdir ./logs/question-3/resnet18-0-$2
}

# run_experiment 0.1 1
# run_experiment 0.01 01
run_experiment 0.001 001
run_experiment 0.0001 0001
run_experiment 0.00001 00001
