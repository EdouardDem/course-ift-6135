#! /bin/bash

set -e

python question-8.py logs/question-2/mlp-relu/results.json "MLP ReLu"
python question-8.py logs/question-2/mlp-sigmoid/results.json "MLP Sigmoid"
python question-8.py logs/question-2/mlp-tanh/results.json "MLP Tanh"

python question-8.py logs/question-3/resnet18-0-001/results.json "ResNet18 lr=0.001"
python question-8.py logs/question-3/resnet18-0-0001/results.json "ResNet18 lr=0.0001"
python question-8.py logs/question-3/resnet18-0-00001/results.json "ResNet18 lr=0.00001"

python question-8.py logs/question-4/mlpmixer-2/results.json "MLPMixer Patch Size 2"
python question-8.py logs/question-4/mlpmixer-4/results.json "MLPMixer Patch Size 4"
python question-8.py logs/question-4/mlpmixer-8/results.json "MLPMixer Patch Size 8"
python question-8.py logs/question-4/mlpmixer-16/results.json "MLPMixer Patch Size 16"

