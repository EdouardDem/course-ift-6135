from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = ["0", "42"]  # Define seeds to use

def plot_metrics_grid(model_name, seeds=["0", "42"], figsize=(15, 12)):
    """
    Plot all metrics for each r_train value for a given model in a 2x2 grid.
    Combines all seeds into a single plot with mean and standard deviation.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list, optional
        List of seeds to average over, default is ["0", "42"]
    figsize : tuple, optional
        Figure size, default is (15, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q3-a', exist_ok=True)
    
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} Model Metrics by r_train', fontsize=16)
    
    # Define the metrics and their positions in the grid
    metrics = [
        ('train.loss', 0, 0, 'Training Loss', False),
        ('test.loss', 0, 1, 'Validation Loss', False),
        ('train.accuracy', 1, 0, 'Training Accuracy', False),
        ('test.accuracy', 1, 1, 'Validation Accuracy', False)
    ]
    
    # Set up colors for different r_train values
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_trains)))
    
    # Keep track of which r_train values have data (for legend)
    valid_r_trains = []
    
    # Process each r_train value
    for idx, r_train in enumerate(r_trains):
        base_dir = Path(f"logs/q3/model={model_name}-optimizer=adamw-n_steps=10000-r_train={r_train}")
        
        # Skip if directory doesn't exist
        if not base_dir.exists():
            print(f"Directory not found: {base_dir}")
            continue
            
        try:
            # Use load_and_combine_results to get combined statistics across seeds
            results = load_and_combine_results(base_dir, seeds)
            
            if results is None:
                print(f"No results found for {model_name} with r_train={r_train}")
                continue
                
            # Get steps for x-axis (use the first seed's steps)
            if 'all_steps' in results and len(results['all_steps']) > 0:
                steps = results['all_steps'][0]
            else:
                print(f"No steps data for {model_name} with r_train={r_train}")
                continue
                
            # Flag to check if we found data for this r_train
            has_data_for_this_r = False
                
            # Plot each metric
            for metric, row, col, label, use_log_scale in metrics:
                dataset, measure = metric.split('.')
                ax = axs[row, col]
                
                # Extract mean and std values
                if dataset in results and measure in results[dataset]:
                    y_mean = results[dataset][measure]['mean']
                    y_std = results[dataset][measure]['std']
                    
                    # Ensure lengths match
                    if len(steps) != len(y_mean):
                        min_len = min(len(steps), len(y_mean))
                        steps_plot = steps[:min_len]
                        y_mean = y_mean[:min_len]
                        y_std = y_std[:min_len] if len(y_std) >= min_len else y_std[:len(y_std)]
                    else:
                        steps_plot = steps
                    
                    # Plot the mean
                    ax.plot(steps_plot, y_mean, color=colors[idx], label=f'r_train={r_train}')
                    
                    # Add error shading for standard deviation
                    ax.fill_between(steps_plot, y_mean - y_std, y_mean + y_std, color=colors[idx], alpha=0.15)
                    
                    has_data_for_this_r = True
                else:
                    print(f"No data found for metric={metric}, model={model_name}, r_train={r_train}")
            
            if has_data_for_this_r and r_train not in valid_r_trains:
                valid_r_trains.append(r_train)
                
        except Exception as e:
            print(f"Error processing {model_name} with r_train={r_train}: {e}")
    
    # Set up each subplot
    for metric, row, col, label, use_log_scale in metrics:
        ax = axs[row, col]
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set log scale for loss plots
        if use_log_scale:
            ax.set_yscale('log')
    
    # Add a single legend for all subplots
    if valid_r_trains:
        handles = [plt.Line2D([0], [0], color=colors[r_trains.index(r)], label=f'r_train={r}')
                  for r in sorted(valid_r_trains)]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    fig.savefig(f'results/q3-a/{model_name}_metrics_grid.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if valid_r_trains:
        print(f"Grid plot saved as results/q3-a/{model_name}_metrics_grid.png")
        return True
    else:
        print(f"No data to plot for {model_name}")
        return False

if __name__ == "__main__":
    print("Generating grid plots for model metrics...")
    
    # Generate grid plot for GPT model
    print("\nGenerating GPT metrics grid plot...")
    plot_metrics_grid('gpt')
    
    # Generate grid plot for LSTM model
    print("\nGenerating LSTM metrics grid plot...")
    plot_metrics_grid('lstm')
    
    print("\nAll plots have been saved to the results/q3-a/ directory.")




