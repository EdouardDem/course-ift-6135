#! /bin/bash

set -e

function run_experiment() {
    python main.py --model mlp --model_config ./model_configs/question-2/$1.json --logdir ./logs/question-2/$1
}

run_experiment mlp-relu
run_experiment mlp-tanh
run_experiment mlp-sigmoid


