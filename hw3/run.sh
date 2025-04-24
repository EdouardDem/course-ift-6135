#! /bin/bash

# Remove saved models
rm *.pkl

# Run the training script
python q1_train_vae.py --epochs 20 --seed 42

# Run the plotting script
python q1_plot.py

# Run the generation script
python q1_generate.py
python q1_interpolate.py
python q1_disentangled.py

rm -rf images/q2 && mkdir -p images/q2
python q2_train.py --epochs 20 

rm -rf images/q3 && mkdir -p images/q3
python q3_train.py --epochs 20 
