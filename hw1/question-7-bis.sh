#! /bin/bash

set -e

function run_experiment() {
    python main.py --model mlpmixer --model_config ./model_configs/question-7-bis/$1.json --visualize --epochs 30 --logdir ./logs/question-7-bis/$1
}

run_experiment mlpmixer-e128
run_experiment mlpmixer-e256
run_experiment mlpmixer-e512
run_experiment mlpmixer-e1024
