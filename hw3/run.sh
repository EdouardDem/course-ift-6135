#! /bin/bash

# Run the training script
python q1_train_vae.py --epochs 20 

# Run the plotting script
python q1_plot.py

# Run the generation script
python q1_generate.py
python q1_interpolate.py
python q1_disentangled.py

python q2_train.py --epochs 20 

python q3_train.py --epochs 20 
