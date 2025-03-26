from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

# Define seeds to use
seeds = ["0", "42"]

def plot_model_metrics_grid(model_name, figsize=(15, 6), log_scale=False):
    """
    Plot train and validation metrics for a specific model in a 1x2 grid.
    Shows accuracy and loss with mean values and standard deviation error bands.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    figsize : tuple, optional
        Figure size, default is (15, 6)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q1', exist_ok=True)
    
    # Define base directory for the model
    base_dir = Path(f"logs/q1/model={model_name}-optimizer=adamw-n_steps=10000")
    
    # Skip if directory doesn't exist
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Set up the plot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} Model Performance Metrics', fontsize=16)
    
    try:
        # Load and combine results from all seeds
        results = load_and_combine_results(base_dir, seeds)
        
        if results is None:
            print(f"No results found for {model_name}")
            plt.close(fig)
            return
        
        # Extract steps for x-axis (use the first seed's steps)
        steps = results['all_steps'][0]
        
        # Metrics to plot with corresponding axis index and title
        metrics = [
            ('accuracy', 0, 'Accuracy'),
            ('loss', 1, 'Loss')
        ]
        
        for metric_type, ax_idx, title in metrics:
            ax = axs[ax_idx]
            
            # Plot training data
            if 'train' in results and metric_type in results['train']:
                train_mean = results['train'][metric_type]['mean']
                train_std = results['train'][metric_type]['std']
                
                ax.plot(steps, train_mean, 'b-', label=f'Train {title}')
                ax.fill_between(steps, train_mean - train_std, train_mean + train_std, 
                                color='blue', alpha=0.15)
            else:
                print(f"No training {metric_type} data found for {model_name}")
            
            # Plot validation data
            if 'test' in results and metric_type in results['test']:
                val_mean = results['test'][metric_type]['mean']
                val_std = results['test'][metric_type]['std']
                
                ax.plot(steps, val_mean, 'r-', label=f'Validation {title}')
                ax.fill_between(steps, val_mean - val_std, val_mean + val_std, 
                                color='red', alpha=0.15)
            else:
                print(f"No validation {metric_type} data found for {model_name}")
            
            # Set up the plot
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(title)
            ax.set_title(f'{title}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Use log scale for loss
            if log_scale and metric_type == 'loss':
                ax.set_yscale('log')
        
        # Adjust layout and save the plot
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        fig.savefig(f'results/q1/{model_name}_metrics_grid.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Grid plot saved as results/q1/{model_name}_metrics_grid.png")
        
    except Exception as e:
        print(f"Error processing {model_name} metrics: {e}")
        plt.close(fig)

if __name__ == "__main__":
    print("Generating grid plots for model performance metrics...")
    
    # Generate grid plot for GPT (accuracy + loss)
    print("\nGenerating GPT metrics grid plot...")
    plot_model_metrics_grid('gpt')
    
    # Generate grid plot for LSTM (accuracy + loss)
    print("\nGenerating LSTM metrics grid plot...")
    plot_model_metrics_grid('lstm')
    
    print("\nAll plots have been saved to the results/q1/ directory.") 