#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_losses():
    """
    Load and plot the training and validation (test) losses saved by q1_train_vae.py
    """
    # Check if results directory exists
    if not os.path.exists('results'):
        print("Error: Results directory not found. Please run q1_train_vae.py first.")
        return

    # Check if loss files exist
    if not os.path.exists('results/train_losses.npy') or not os.path.exists('results/test_losses.npy'):
        print("Error: Loss files not found. Please run q1_train_vae.py first.")
        return

    # Load losses
    train_losses = np.load('results/train_losses.npy')
    test_losses = np.load('results/test_losses.npy')

    # Create epochs array
    epochs = np.arange(1, len(train_losses) + 1)

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss (negative ELBO)')
    plt.title('VAE Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig('results/vae_losses.png', dpi=300, bbox_inches='tight')
    
    # Print final losses
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {test_losses[-1]:.4f}")
    
    # Show the plot
    # plt.show()

if __name__ == "__main__":
    plot_losses() 