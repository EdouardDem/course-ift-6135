from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = ["0", "42"]  # Define seeds to use

def plot_metric_by_r_train(metric, label, model_name, seeds=["0", "42"], figsize=(10, 6)):
    """
    Plot a specific metric for each r_train value for a given model.
    Combines all seeds into a single plot with mean and standard deviation.
    
    Parameters:
    -----------
    metric : str
        The metric to plot (e.g., 'train.loss', 'test.accuracy')
    label : str
        The label for the y-axis and part of the plot title
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list, optional
        List of seeds to average over, default is ["0", "42"]
    figsize : tuple, optional
        Figure size, default is (10, 6)
    """
    # Split metric into dataset and measure
    dataset, measure = metric.split('.')
    
    # Create results directory if it doesn't exist
    os.makedirs('results/q3-a', exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Set up colors for different r_train values
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_trains)))
    
    has_data = False  # Flag to track if any data was plotted
    
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
                
            # Extract mean and std values
            if dataset in results and measure in results[dataset]:
                steps = results['all_steps'][0]  # Use the first seed's steps for x-axis
                y_mean = results[dataset][measure]['mean']
                y_std = results[dataset][measure]['std']
                
                # Plot the mean
                plt.plot(steps, y_mean, color=colors[idx], label=f'r_train={r_train}')
                
                # Add error shading for standard deviation
                plt.fill_between(steps, y_mean - y_std, y_mean + y_std, color=colors[idx], alpha=0.3)
                
                has_data = True
            else:
                print(f"No data found for metric={metric}, model={model_name}, r_train={r_train}")
        except Exception as e:
            print(f"Error processing {model_name} with r_train={r_train}: {e}")
    
    if not has_data:
        plt.close()
        print(f"No data to plot for {model_name} with metric {metric}")
        return
        
    plt.xlabel('Training Steps')
    plt.ylabel(label)
    plt.title(f'{model_name.upper()} - {label} by r_train (averaged over seeds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    safe_metric = metric.replace('.', '_')
    plt.savefig(f'results/q3-a/{model_name}_{safe_metric}_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved as results/q3-a/{model_name}_{safe_metric}_combined.png")

if __name__ == "__main__":
    # Plot training and validation metrics for the GPT model
    print("Generating plots for GPT model...")
    plot_metric_by_r_train('train.loss', 'Training Loss', 'gpt')
    plot_metric_by_r_train('test.loss', 'Validation Loss', 'gpt')
    plot_metric_by_r_train('train.accuracy', 'Training Accuracy', 'gpt')
    plot_metric_by_r_train('test.accuracy', 'Validation Accuracy', 'gpt')
    
    # Generate plots for LSTM model
    print("\nGenerating plots for LSTM model...")
    plot_metric_by_r_train('train.loss', 'Training Loss', 'lstm')
    plot_metric_by_r_train('test.loss', 'Validation Loss', 'lstm')
    plot_metric_by_r_train('train.accuracy', 'Training Accuracy', 'lstm')
    plot_metric_by_r_train('test.accuracy', 'Validation Accuracy', 'lstm')
    
    print("\nAll plots have been saved to the results/q3-a/ directory.")




