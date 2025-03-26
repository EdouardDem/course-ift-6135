from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase

# Configuration
models = ["lstm", "gpt"]  # Models to plot
seeds = ["0", "42"]       # Seeds to average over
num_layers_values = [1, 2, 3]  # L values
embedding_sizes = [64, 128, 256]  # d values (will be displayed in log2 scale)

def plot_metrics_by_layer_count(model_name, num_layers, seeds, figsize=(15, 12)):
    """
    Plot training and validation metrics for a given model and number of layers (L),
    with different embedding sizes (d) shown using a color gradient.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    num_layers : int
        Number of layers (L)
    seeds : list
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (15, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q5-a', exist_ok=True)
    
    # Create a 2x2 subplot grid for the four metrics
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} Model - L={num_layers} Layers', fontsize=16)
    
    # Metrics and positions in the grid
    grid_positions = {
        ('train', 'loss'): (0, 0),     # Training Loss: top-left
        ('test', 'loss'): (0, 1),      # Validation Loss: top-right  
        ('train', 'accuracy'): (1, 0),  # Training Accuracy: bottom-left
        ('test', 'accuracy'): (1, 1)    # Validation Accuracy: bottom-right
    }
    
    # Titles for each subplot
    titles = {
        ('train', 'loss'): 'Training Loss',
        ('test', 'loss'): 'Validation Loss',
        ('train', 'accuracy'): 'Training Accuracy',
        ('test', 'accuracy'): 'Validation Accuracy'
    }
    
    # Y-axis labels
    y_labels = {
        'loss': 'Loss',
        'accuracy': 'Accuracy'
    }
    
    # Setup colormap for different embedding sizes
    # We'll use log2 scale for embedding sizes as requested
    log2_embedding_sizes = [np.log2(size) for size in embedding_sizes]
    norm = mcolors.Normalize(vmin=min(log2_embedding_sizes), vmax=max(log2_embedding_sizes))
    cmap = plt.cm.viridis
    
    # Load data for each embedding size
    embedding_results = {}
    steps = None  # Will store the steps for x-axis
    
    for d in embedding_sizes:
        if model_name == "lstm":
            # For LSTM, both embedding_size and hidden_size are set to d
            base_dir = Path(f"logs/q5/model={model_name}-optimizer=adamw-n_steps=10000-num_layers={num_layers}-embedding_size={d}-hidden_size={d}")
        else:  # model_name == "gpt"
            base_dir = Path(f"logs/q5/model={model_name}-optimizer=adamw-n_steps=10000-num_layers={num_layers}-embedding_size={d}")
        
        if not base_dir.exists():
            print(f"Directory not found: {base_dir}")
            continue
        
        try:
            results = load_and_combine_results(base_dir, seeds)
            if results is not None:
                embedding_results[d] = results
                # Get steps for x-axis (use the first data we find)
                if steps is None and 'all_steps' in results:
                    steps = results['all_steps'][0]
            else:
                print(f"No results found for {model_name} with L={num_layers}, d={d}")
        except Exception as e:
            print(f"Error loading data for {model_name} with L={num_layers}, d={d}: {e}")
    
    # Check if we have data for at least one embedding size
    if not embedding_results or steps is None:
        print(f"No data available for {model_name} with L={num_layers}. Cannot create plot.")
        plt.close(fig)
        return False
    
    # Plot each metric on its corresponding subplot
    for (dataset, metric), (row, col) in grid_positions.items():
        ax = axs[row, col]
        
        # Set title and labels
        ax.set_title(titles[(dataset, metric)])
        ax.set_xlabel('Training Steps (t)')
        ax.set_ylabel(y_labels[metric])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot data for each embedding size
        for d, results in embedding_results.items():
            if dataset in results and metric in results[dataset]:
                y_mean = results[dataset][metric]['mean']
                y_std = results[dataset][metric]['std']
                
                # Use color based on log2 scale of embedding size
                color = cmap(norm(np.log2(d)))
                ax.plot(steps, y_mean, color=color, label=f'd={d}')
                ax.fill_between(steps, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)
    
    # Add a single legend to the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), title="Embedding Size (d)")
    
    # Adjust layout before adding colorbar
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust for the suptitle and colorbar
    
    # Add horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label('Embedding Size (d) - $\\log_2$ scale')
    
    # Set ticks to show actual embedding sizes, but positioned according to log2 scale
    cb.set_ticks(log2_embedding_sizes)
    cb.set_ticklabels(embedding_sizes)
    
    # Save figure
    fig.savefig(f'results/q5-a/{model_name}_L{num_layers}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Metrics plot saved as results/q5-a/{model_name}_L{num_layers}.png")
    return True

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q5-a', exist_ok=True)
    
    # Track whether any plots were successfully created
    success = False
    
    # Generate metric plots for each model and layer count
    for model in models:
        print(f"\nProcessing {model.upper()} model...")
        for L in num_layers_values:
            print(f"Generating plots for {model.upper()} model with L={L}...")
            if plot_metrics_by_layer_count(model, L, seeds):
                success = True
    
    print("\nAll plots have been saved to the results/q5-a/ directory.") 