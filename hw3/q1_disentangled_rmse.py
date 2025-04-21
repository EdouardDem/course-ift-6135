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
    
    # Load the trained model
    model, device = load_trained_vae(save_path)
    
    # Generate samples from the prior distribution
    latent_dim = 20
    eps_factor = 5
    samples = []
    with torch.no_grad():
        # Sample from normal distribution
        source = torch.randn(num_samples, latent_dim).to(device)

        # Generate the original samples
        original = model.decode(source)
        original = original.view(num_samples, 1, 28, 28)
        samples.append(original)

        for i in range(latent_dim):
            z = source.clone()
            z[:, i] += eps_factor

            sample = model.decode(z)
            sample = sample.view(num_samples, 1, 28, 28)
            samples.append(sample)
        
        # Reshape and save the generated images
        all_samples = torch.cat(samples, dim=0)
        save_image(all_samples, os.path.join(save_path, 'disentangled_samples.png'), nrow=5, normalize=True)
    
    print(f"Generated {num_samples} disentangled samples and saved to {save_path}/disentangled_samples.png")
    
    # ------------------------------------------------------------
    plot_grid(
        num_samples=num_samples,
        samples=samples,
        eps_factor=eps_factor,
        save_path=save_path,
        file_name='disentangled_grid_with_distances.png',
        fig_name='Disentangled Representations - Latent Traversals')
    
    plot_grid(
        num_samples=num_samples,
        samples=samples,
        eps_factor=eps_factor,
        save_path=save_path,
        start_index=0,
        end_index=10,
        file_name='disentangled_grid_with_distances_1_to_10.png',
        fig_name='Disentangled Representations - Latent Traversals (1 to 10)')
    
    plot_grid(
        num_samples=num_samples,
        samples=samples,
        eps_factor=eps_factor,
        save_path=save_path,
        start_index=10,
        end_index=20,
        file_name='disentangled_grid_with_distances_10_to_20.png',
        fig_name='Disentangled Representations - Latent Traversals (10 to 20)')


def plot_grid(num_samples, samples, eps_factor, save_path, file_name, fig_name, start_index=0, end_index=20):

    # ------------------------------------------------------------
    # Compute RMSE between samples and source
    rmse_values = []
    for i in range(1, len(samples)):  # Skip the first one (source)
        # Calculate pixel-wise squared differences for each sample compared to original
        squared_diff = (samples[i] - samples[0]) ** 2
        # Calculate RMSE for each sample in the batch (mean over all pixels and then sqrt)
        rmse = torch.sqrt(squared_diff.view(num_samples, -1).mean(dim=1))
        # Take the mean RMSE across all samples in the batch
        mean_rmse = rmse.mean().cpu().numpy()
        rmse_values.append(mean_rmse)
    
    # Create grid for visualization
    num_rows = end_index-start_index+2
    height_ratios = [1] * num_rows
    height_ratios[0] = 0.5
    fig = plt.figure(figsize=(15, num_rows*2))
    gs = GridSpec(num_rows, 7, figure=fig, width_ratios=[0.5, 1, 1, 1, 1, 1, 2], height_ratios=height_ratios)
    fig.suptitle(fig_name, fontsize=16)
    
    # ------------------------------------------------------------
    # Add column headers
    grid_fs = 16
    for i in range(5):
        ax = fig.add_subplot(gs[0, i+1])
        ax.text(0.5, 0.5, f"Sample {i+1}", ha='center', va='center', fontsize=grid_fs)
        ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5, f"Perturbed dim \n(ε = {eps_factor})", ha='center', va='center', fontsize=grid_fs)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 6])
    ax.text(0.5, 0.5, "RMSE", ha='center', va='center', fontsize=grid_fs)
    ax.axis('off')
    
    # ------------------------------------------------------------
    # Plot source row (original samples)
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.5, "Source", ha='center', va='center', fontsize=grid_fs)
    ax.axis('off')

    for i in range(5):
        ax = fig.add_subplot(gs[1, i +1])
        img = samples[0][i].cpu().numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    # ------------------------------------------------------------
    # Add empty cell for distance (no distance for source)
    ax = fig.add_subplot(gs[1, 6])
    ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=grid_fs)
    ax.axis('off')

    # ------------------------------------------------------------
    # Plot each latent dimension traversal
    max_rmse = max(rmse_values) if rmse_values else 0.5
    for dim in range(start_index, end_index):
        row_idx = dim + 2  # +2 because of title and source rows

        ax = fig.add_subplot(gs[row_idx - start_index, 0])
        ax.text(0.5, 0.5, f"Dim {dim+1}", ha='center', va='center', fontsize=grid_fs)
        ax.axis('off')
        
        # Plot samples for this dimension
        for i in range(5):
            ax = fig.add_subplot(gs[row_idx - start_index, i + 1])
            img = samples[dim+1][i].cpu().numpy().squeeze()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        
        # Plot RMSE metric
        rmse = rmse_values[dim]
        ax = fig.add_subplot(gs[row_idx - start_index, 6])
        
        # Create a horizontal bar chart for RMSE
        ax.barh([0], rmse, align='center', alpha=0.9, color='skyblue')
        ax.set_xlim([0, min(max_rmse * 1.2, 0.5)])  # Adjust based on your data
        ax.set_yticks([])  # Hide y-axis ticks
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    plt.savefig(os.path.join(save_path, file_name), dpi=300, bbox_inches='tight')
    print(f"Created grid visualization with RMSE at {save_path}/{file_name}")


if __name__ == "__main__":
    generate_disentangled_samples() 