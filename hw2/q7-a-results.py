from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Configuration
models = ["lstm"]  # Only LSTM model in q7
seeds = ["0", "42"]  # Seeds to average over
weight_decay_values_string = ["0.25", "0.5", "0.75", "1"]  # weight_decay values as strings
weight_decay_values = [0.25, 0.5, 0.75, 1.0]  # weight_decay values

def load_all_results():
    """
    Load results for all model configurations.
    
    Returns:
    --------
    dict : Dictionary with all results organized by model and weight_decay
    """
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for weight_decay in weight_decay_values:
            # Set up the correct base directory
            weight_decay_string = weight_decay_values_string[weight_decay_values.index(weight_decay)]
            base_dir = Path(f"logs/q7/model={model_name}-optimizer=adamw-n_steps=40001-weight_decay={weight_decay_string}")
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            try:
                results = load_and_combine_results(base_dir, seeds)
                if results is not None:
                    all_results[model_name][float(weight_decay)] = results
                else:
                    print(f"No results found for {model_name} with weight_decay={weight_decay}")
            except Exception as e:
                print(f"Error loading data for {model_name} with weight_decay={weight_decay}: {e}")
    
    return all_results

def plot_metrics_over_time(all_results, figsize=(10, 18)):
    """
    Plot metrics as a function of time, with weight_decay on a color bar.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with results for different models and weight_decay values
    figsize : tuple, optional
        Figure size, default is (10, 18)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q7-a', exist_ok=True)
    
    # Define metrics to plot
    metric_pairs = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('l2_norm', 'L2 Norm $\\|\\theta^{(t)}\\|_2$')
    ]
    
    for model_name in all_results:
        # Create a figure for each metric pair (train vs val)
        for metric_key, metric_label in metric_pairs:
            # Set up colormap for weight_decay values
            norm = mcolors.Normalize(vmin=min(weight_decay_values), vmax=max(weight_decay_values))
            cmap = plt.cm.viridis
            
            # Special handling for l2_norm - use a single plot
            if metric_key == 'l2_norm':
                # Create a single figure for L2 norm
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                fig.suptitle(f'{model_name.upper()} Model - {metric_label} vs Training Steps', fontsize=16)
                
                ax.set_title(f'{metric_label}')
                ax.set_xlabel('Training Steps (t)')
                ax.set_ylabel(metric_label)
                
                # For each weight_decay value, plot the metric
                for weight_decay in sorted(all_results[model_name].keys()):
                    results = all_results[model_name][weight_decay]
                    
                    # Get steps for x-axis
                    if 'all_steps' not in results or len(results['all_steps']) == 0:
                        print(f"No steps data for {model_name} with weight_decay={weight_decay}")
                        continue
                    
                    steps = results['all_steps'][0]  # Use first seed's steps
                    
                    # L2 norm is stored differently
                    if 'l2_norm' in results.get('train', {}):
                        values = results['train']['l2_norm']['mean']
                        std_values = results['train']['l2_norm']['std']
                    else:
                        print(f"No L2 norm data for {model_name} with weight_decay={weight_decay}")
                        continue
                    
                    # Make sure lengths match
                    if len(steps) != len(values):
                        min_len = min(len(steps), len(values))
                        steps = steps[:min_len]
                        values = values[:min_len]
                        std_values = std_values[:min_len] if len(std_values) >= min_len else std_values[:len(std_values)]
                    
                    # Plot with color based on weight_decay
                    color = cmap(norm(weight_decay))
                    line, = ax.plot(steps, values, '-', color=color, label=f'weight_decay={weight_decay}')
                    
                    # Add fill_between for standard deviation
                    ax.fill_between(steps, values - std_values, values + std_values, color=color, alpha=0.15)
                
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(loc='best')
            else:
                # For other metrics (loss, accuracy), use the original 1x2 grid for train/test
                fig, axs = plt.subplots(1, 2, figsize=figsize[::-1])  # Swap width and height for 1x2 grid
                fig.suptitle(f'{model_name.upper()} Model - {metric_label} vs Training Steps', fontsize=16)
                
                # Define the splits for regular metrics
                splits = ['train', 'test']
                split_labels = ['Training', 'Validation']
                
                # For each subplot (train and validation)
                for idx, (split, split_label) in enumerate(zip(splits, split_labels)):
                    ax = axs[idx]
                    ax.set_title(f'{split_label} {metric_label}')
                    ax.set_xlabel('Training Steps (t)')
                    ax.set_ylabel(metric_label)
                    
                    # Use log scale for loss
                    if metric_key == 'loss':
                        ax.set_yscale('log')
                    
                    # For each weight_decay value, plot the metric
                    for weight_decay in sorted(all_results[model_name].keys()):
                        results = all_results[model_name][weight_decay]
                        
                        # Get steps for x-axis
                        if 'all_steps' not in results or len(results['all_steps']) == 0:
                            print(f"No steps data for {model_name} with weight_decay={weight_decay}")
                            continue
                        
                        steps = results['all_steps'][0]  # Use first seed's steps
                        
                        # Regular metrics (loss and accuracy)
                        if split not in results or metric_key not in results[split]:
                            print(f"Missing {split} {metric_key} data for {model_name} with weight_decay={weight_decay}")
                            continue
                        
                        values = results[split][metric_key]['mean']
                        std_values = results[split][metric_key]['std']
                        
                        # Make sure lengths match
                        if len(steps) != len(values):
                            min_len = min(len(steps), len(values))
                            steps = steps[:min_len]
                            values = values[:min_len]
                            std_values = std_values[:min_len] if len(std_values) >= min_len else std_values[:len(std_values)]
                        
                        # Plot with color based on weight_decay
                        color = cmap(norm(weight_decay))
                        line, = ax.plot(steps, values, '-', color=color, label=f'weight_decay={weight_decay}')
                        
                        # Add fill_between for standard deviation
                        ax.fill_between(steps, values - std_values, values + std_values, color=color, alpha=0.15)
                    
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add legend to the first subplot for each metric pair
                    if idx == 0:
                        ax.legend(loc='best')
            
            # Adjust layout before adding colorbar
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Add horizontal colorbar at the bottom
            cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Weight Decay')
            
            # Set up colorbar ticks
            cbar.set_ticks(weight_decay_values)
            cbar.set_ticklabels([str(wd) for wd in weight_decay_values])
            
            # Save the figure
            metric_filename = 'l2-norm' if metric_key == 'l2_norm' else metric_key
            fig.savefig(f'results/q7-a/{model_name}_{metric_filename}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"{model_name.upper()} {metric_label} plot saved as results/q7-a/{model_name}_{metric_filename}.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q7-a', exist_ok=True)
    
    print("Loading all model results...")
    all_results = load_all_results()
    
    print("Plotting metrics over time...")
    plot_metrics_over_time(all_results)
    
    print("\nAll plots have been saved to the results/q7-a/ directory.") 