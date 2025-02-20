#! /bin/bash

set -e

function run_experiment() {
    python main.py --model mlpmixer --model_config ./model_configs/question-4/$1.json --visualize --epochs 40 --logdir ./logs/question-4/$1
}

run_experiment mlpmixer-2
run_experiment mlpmixer-4
run_experiment mlpmixer-8
run_experiment mlpmixer-16
