#! /bin/bash

set -e

python main.py --model mlp --model_config ./model_configs/question-2/mlp-relu.json --logdir ./logs/question-2/mlp-relu

python main.py --model mlp --model_config ./model_configs/question-2/mlp-tanh.json --logdir ./logs/question-2/mlp-tanh

python main.py --model mlp --model_config ./model_configs/question-2/mlp-sigmoid.json --logdir ./logs/question-2/mlp-sigmoid

