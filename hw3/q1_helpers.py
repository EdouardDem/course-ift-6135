#!/usr/bin/env python
# coding: utf-8

import torch
import os

# Import the VAE model class from the training script
from q1_train_vae import VAE

def load_trained_vae(save_path='results', model_name='vae_final.pt'):
    """
    Load a trained VAE model from the specified path.
    
    Args:
        save_path: Directory where the model is saved
        model_name: Name of the model file
    
    Returns:
        model: The loaded VAE model
        device: The device (CPU/GPU) the model is on
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    # Check if save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = VAE().to(device)
    
    # Check if the model file exists
    model_path = os.path.join(save_path, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device 