#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy as np

# Import the VAE model class from the training script
from q1_train_vae import VAE
from q1_helpers import load_trained_vae

def generate_samples(num_samples=10, save_path='results'):
    """
    Load the trained VAE model and generate samples.
    
    Args:
        num_samples: Number of samples to generate
        save_path: Directory to save the generated images
    """
    # Check if save directory exists, create if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load the trained model
    model, device = load_trained_vae(save_path)
    
    # Generate samples from the prior distribution
    with torch.no_grad():
        # Sample from normal distribution
        z = torch.randn(num_samples, 20).to(device)
        
        # Generate images from the latent vectors
        sample = model.decode(z)
        
        # Reshape and save the generated images
        sample = sample.view(num_samples, 1, 28, 28)
        save_image(sample, os.path.join(save_path, 'vae_samples.png'), nrow=5, normalize=True)
    
    print(f"Generated {num_samples} samples and saved to {save_path}/vae_samples.png")
    
    # Display individual samples with matplotlib for detailed inspection
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        img = sample[i].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'vae_samples_grid.png'), dpi=300)

if __name__ == "__main__":
    generate_samples() 