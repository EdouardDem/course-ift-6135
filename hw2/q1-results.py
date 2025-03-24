from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

# Define seeds to use
seeds = ["0", "42"]

def plot_model_metrics(model_name, metric_type, figsize=(10, 6)):
    """
    Plot train and validation metrics for a specific model.
    Shows mean values with standard deviation error bands.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    metric_type : str
        The type of metric to plot ('loss' or 'accuracy')
    figsize : tuple, optional
        Figure size, default is (10, 6)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q1', exist_ok=True)
    
    # Define base directory for the model
    base_dir = Path(f"logs/q1/model={model_name}-optimizer=adamw-n_steps=10000")
    
    # Skip if directory doesn't exist
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    try:
        # Load and combine results from all seeds
        results = load_and_combine_results(base_dir, seeds)
        
        if results is None:
            print(f"No results found for {model_name}")
            return
        
        # Extract steps for x-axis (use the first seed's steps)
        steps = results['all_steps'][0]
        
        # Plot training data
        if 'train' in results and metric_type in results['train']:
            train_mean = results['train'][metric_type]['mean']
            train_std = results['train'][metric_type]['std']
            
            plt.plot(steps, train_mean, 'b-', label=f'Train {metric_type.capitalize()}')
            plt.fill_between(steps, train_mean - train_std, train_mean + train_std, 
                             color='blue', alpha=0.3)
        else:
            print(f"No training {metric_type} data found for {model_name}")
        
        # Plot validation data
        if 'test' in results and metric_type in results['test']:
            val_mean = results['test'][metric_type]['mean']
            val_std = results['test'][metric_type]['std']
            
            plt.plot(steps, val_mean, 'r-', label=f'Validation {metric_type.capitalize()}')
            plt.fill_between(steps, val_mean - val_std, val_mean + val_std, 
                             color='red', alpha=0.3)
        else:
            print(f"No validation {metric_type} data found for {model_name}")
        
        # Set up the plot
        plt.xlabel('Training Steps')
        plt.ylabel(metric_type.capitalize())
        plt.title(f'{model_name.upper()} - {metric_type.capitalize()} (averaged over seeds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot
        plt.savefig(f'results/q1/{model_name}_{metric_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved as results/q1/{model_name}_{metric_type}.png")
        
    except Exception as e:
        print(f"Error processing {model_name} {metric_type}: {e}")
        plt.close()

if __name__ == "__main__":
    print("Generating plots for model performance metrics...")
    
    # Generate plot for GPT accuracy (train + validation)
    print("\nGenerating GPT accuracy plot...")
    plot_model_metrics('gpt', 'accuracy')
    
    # Generate plot for GPT loss (train + validation)
    print("\nGenerating GPT loss plot...")
    plot_model_metrics('gpt', 'loss')
    
    # Generate plot for LSTM accuracy (train + validation)
    print("\nGenerating LSTM accuracy plot...")
    plot_model_metrics('lstm', 'accuracy')
    
    # Generate plot for LSTM loss (train + validation)
    print("\nGenerating LSTM loss plot...")
    plot_model_metrics('lstm', 'loss')
    
    print("\nAll plots have been saved to the results/q1/ directory.") 