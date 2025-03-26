from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import math

# Configuration
models = ["lstm", "gpt"]  # Models to plot
seeds = ["0", "42"]       # Seeds to average over
batch_sizes = [32, 64, 128, 256, 512]  # B values

def load_all_results():
    """
    Load results for all model configurations.
    
    Returns:
    --------
    dict : Dictionary with all results organized by model and batch size
    """
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for batch_size in batch_sizes:
            # Set up the correct base directory
            base_dir = Path(f"logs/q6/model={model_name}-optimizer=adamw-n_steps=20001-train_batch_size={batch_size}")
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            try:
                results = load_and_combine_results(base_dir, seeds)
                if results is not None:
                    all_results[model_name][batch_size] = results
                else:
                    print(f"No results found for {model_name} with batch_size={batch_size}")
            except Exception as e:
                print(f"Error loading data for {model_name} with batch_size={batch_size}: {e}")
    
    return all_results

def plot_metrics_over_time(all_results, figsize=(12, 12)):
    """
    Plot metrics as a function of time, with batch size on a color bar.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with results for different models and batch sizes
    figsize : tuple, optional
        Figure size, default is (12, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q6-a', exist_ok=True)
    
    # Metrics mapping
    metrics_split_mapping = [
        ('train', 'loss', 'Training Loss', 0, 0, True),
        ('test', 'loss', 'Validation Loss', 0, 1, True),
        ('train', 'accuracy', 'Training Accuracy', 1, 0, False),
        ('test', 'accuracy', 'Validation Accuracy', 1, 1, False)
    ]
    
    # Set up colormap for batch sizes (log2 scale)
    min_batch = min(batch_sizes)
    max_batch = max(batch_sizes)
    norm = mcolors.LogNorm(vmin=min_batch, vmax=max_batch)
    cmap = plt.cm.viridis
    
    for model_name in all_results:
        # Create figure with more space at the bottom for colorbar
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{model_name.upper()} Model - Metrics vs Training Steps', fontsize=16)
        
        # For each metric, create a subplot
        for split, metric, metric_label, row, col, use_log_scale in metrics_split_mapping:
            ax = axs[row, col]
            ax.set_title(metric_label)
            ax.set_xlabel('Training Steps (t)')
            ax.set_ylabel(metric_label)
            
            if use_log_scale:
                ax.set_yscale('log')
            
            # For each batch size, plot the metric
            for batch_size in sorted(all_results[model_name].keys()):
                results = all_results[model_name][batch_size]
                
                # Check if required data exists
                if split not in results or metric not in results[split]:
                    print(f"Missing {split} {metric} data for {model_name} with batch_size={batch_size}")
                    continue
                
                # Get all steps (x-axis values)
                if 'all_steps' in results and len(results['all_steps']) > 0:
                    steps = results['all_steps'][0]  # Use first seed's steps
                else:
                    print(f"No steps data for {model_name} with batch_size={batch_size}")
                    continue
                
                # Get metric values
                metric_data = results[split][metric]
                y_mean = metric_data["mean"]  # Use mean across seeds
                y_std = metric_data["std"]    # Use standard deviation across seeds
                
                if len(steps) != len(y_mean) or len(steps) != len(y_std):
                    print(f"Steps and values length mismatch for {model_name} with batch_size={batch_size}")
                    print(f"Steps: {len(steps)}, Mean: {len(y_mean)}, Std: {len(y_std)}")
                    # Try to trim to the shortest length
                    min_len = min(len(steps), len(y_mean), len(y_std))
                    steps = steps[:min_len]
                    y_mean = y_mean[:min_len]
                    y_std = y_std[:min_len]
                
                # Plot with color based on batch size
                color = cmap(norm(batch_size))
                line, = ax.plot(steps, y_mean, '-', color=color, label=f'B={batch_size}')
                
                # Add fill_between for standard deviation
                ax.fill_between(steps, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15)
            
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a legend in the first subplot to explain the batch sizes
        legend_elements = [plt.Line2D([0], [0], color=cmap(norm(b)), lw=2, label=f'B={b}') 
                          for b in sorted(batch_sizes) if b in all_results[model_name]]
        if legend_elements:
            axs[0, 0].legend(handles=legend_elements, loc='best')
        
        # Adjust layout before adding colorbar
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Add horizontal colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Batch Size (B) - $\\log_2$ scale')
        
        # Set up colorbar ticks in log2 scale
        cbar.set_ticks(batch_sizes)
        cbar.set_ticklabels([str(b) for b in batch_sizes])
        
        # Save figure
        fig.savefig(f'results/q6-a/{model_name}_metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Model {model_name} metrics plot saved as results/q6-a/{model_name}_metrics_over_time.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q6-a', exist_ok=True)
    
    print("Loading all model results...")
    all_results = load_all_results()
    
    print("Plotting metrics over time...")
    plot_metrics_over_time(all_results)
    
    print("\nAll plots have been saved to the results/q6-a/ directory.") 