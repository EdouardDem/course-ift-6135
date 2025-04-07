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

def generate_disentangled_samples(num_samples=5, save_path='results'):
    """
    Load the trained VAE model and generate disentangled samples.
    
    Args:
        num_samples: Number of samples to generate
        save_path: Directory to save the generated images
    """
    # Check if save directory exists, create if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = VAE().to(device)
    
    # Check if the model file exists
    model_path = os.path.join(save_path, 'vae_final.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Generate samples from the prior distribution
    latent_dim = 20
    samples = []
    with torch.no_grad():
        # Sample from normal distribution
        source = torch.randn(num_samples, latent_dim).to(device)

        # Generate the original samples
        original = model.decode(source)
        original = original.view(num_samples, 1, 28, 28)
        samples.append(original)

        for i in range(latent_dim):
            # eps_factor = torch.randn(num_samples).to(device)
            eps_factor = 5
            z = source.clone()
            z[:, i] += eps_factor

            sample = model.decode(z)
            sample = sample.view(num_samples, 1, 28, 28)
            samples.append(sample)
        
        # Reshape and save the generated images
        sample = torch.cat(samples, dim=0)
        save_image(sample, os.path.join(save_path, 'disentangled_samples.png'), nrow=5, normalize=True)
    
    print(f"Generated {num_samples} disentangled samples and saved to {save_path}/disentangled_samples.png")
    
    # Display individual samples with matplotlib for detailed inspection
    # fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    # axes = axes.flatten()
    
    # for i, ax in enumerate(axes):
    #     img = sample[i].cpu().numpy().squeeze()
    #     ax.imshow(img, cmap='gray')
    #     ax.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, 'vae_samples_grid.png'), dpi=300)

if __name__ == "__main__":
    generate_disentangled_samples() 