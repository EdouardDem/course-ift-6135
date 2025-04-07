#!/usr/bin/env python
# coding: utf-8

import torch
import torch.utils.data
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import numpy as np

# Import the VAE model class from the training script
from q1_train_vae import VAE
from q1_helpers import load_trained_vae

def interpolate(x1, x2, alpha):
    return alpha * x1 + (1 - alpha) * x2

def generate_interpolation_comparison(save_path='results'):
    """
    Generate and compare interpolations in latent space and data space.
    
    Args:
        save_path: Directory to save the generated images
    """
    # Check if save directory exists, create if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Load the trained model
    model, device = load_trained_vae(save_path)
    
    # Generate two random points in latent space
    latent_dim = 20
    with torch.no_grad():
        # Sample two points from the prior
        z0 = torch.randn(1, latent_dim).to(device)
        z1 = torch.randn(1, latent_dim).to(device)
        
        # Generate corresponding images
        x0 = model.decode(z0)
        x1 = model.decode(z1)
        
        # Create interpolation points
        alphas = np.linspace(0, 1, 11)
        
        # Interpolate in latent space
        latent_interpolations = []
        for alpha in alphas:
            z_alpha = interpolate(z0, z1, alpha)
            x_alpha = model.decode(z_alpha)
            latent_interpolations.append(x_alpha)
        
        # Interpolate in data space
        data_interpolations = []
        for alpha in alphas:
            x_alpha = interpolate(x0, x1, alpha)
            data_interpolations.append(x_alpha)
        
        # Convert to tensors and reshape
        latent_interpolations = torch.cat(latent_interpolations, dim=0).view(-1, 1, 28, 28)
        data_interpolations = torch.cat(data_interpolations, dim=0).view(-1, 1, 28, 28)
        
        # Save the interpolations
        save_image(latent_interpolations, 
                  os.path.join(save_path, 'latent_space_interpolation.png'), 
                  nrow=len(alphas), normalize=True)
        save_image(data_interpolations, 
                  os.path.join(save_path, 'data_space_interpolation.png'), 
                  nrow=len(alphas), normalize=True)
        
        # Create a comparison plot
        fig = plt.figure(figsize=(15, 6))
        gs = GridSpec(2, 1, figure=fig)
        
        # Plot latent space interpolation
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(torch.cat([latent_interpolations[i] for i in range(len(alphas))], dim=2).squeeze().cpu().numpy(), 
                  cmap='gray')
        ax1.set_title('Latent Space Interpolation')
        ax1.axis('off')
        
        # Plot data space interpolation
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(torch.cat([data_interpolations[i] for i in range(len(alphas))], dim=2).squeeze().cpu().numpy(), 
                  cmap='gray')
        ax2.set_title('Data Space Interpolation')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'interpolation_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Created interpolation comparison at {save_path}/interpolation_comparison.png")

if __name__ == "__main__":
    generate_interpolation_comparison() 