from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def load_results(model_name, r_train, seed):
    """
    Load the content of the pth file in the results directory
    """
    base_dir = Path("logs/q3")
    results_dir = base_dir / f"model={model_name}-optimizer=adamw-n_steps=10000-r_train={r_train}/seed={seed}/0/"
    pth_file = results_dir / f"test.pth"
    try:    
        return torch.load(pth_file)
    except FileNotFoundError:
        print(f"File {pth_file} not found")
        return None

def plot_metric_by_r_train(metric, label, model_name, seeds=[0, 42], figsize=(10, 6)):
    """
    Plot a specific metric for each r_train value for a given model.
    Creates a separate plot for each seed.
    
    Parameters:
    -----------
    metric : str
        The metric to plot (e.g., 'train.loss', 'test.accuracy')
    label : str
        The label for the y-axis and part of the plot title
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list, optional
        List of seeds to plot separately, default is [0, 42]
    figsize : tuple, optional
        Figure size, default is (10, 6)
    """
    # Split metric into dataset and measure
    dataset, measure = metric.split('.')
    
    # Create results directory if it doesn't exist
    os.makedirs('results/q3', exist_ok=True)
    
    # Generate plots for each seed separately
    for seed in seeds:
        plt.figure(figsize=figsize)
        
        # Set up colors for different r_train values
        colors = plt.cm.viridis(np.linspace(0, 1, len(r_trains)))
        
        has_data = False  # Flag to track if any data was plotted
        
        for idx, r_train in enumerate(r_trains):
            # Load results for this seed and r_train
            results = load_results(model_name, r_train, seed)
            if results is None:
                continue
                
            # Check if both the dataset and measure exist in the results
            if dataset in results and measure in results[dataset]:
                steps = results['all_steps']
                values = results[dataset][measure]
                
                # Plot the values
                plt.plot(steps, values, color=colors[idx], label=f'r_train={r_train}')
                has_data = True
            else:
                print(f"No data found for metric={metric}, model={model_name}, r_train={r_train}, seed={seed}")
        
        if not has_data:
            plt.close()
            print(f"No data to plot for seed={seed}")
            continue
            
        plt.xlabel('Training Steps')
        plt.ylabel(label)
        plt.title(f'{model_name.upper()} - {label} by r_train (Seed {seed})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the figure with seed in the filename
        safe_metric = metric.replace('.', '_')
        plt.savefig(f'results/q3/{model_name}_{safe_metric}_seed{seed}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved as results/q3/{model_name}_{safe_metric}_seed{seed}.png")

# Example usage:
# plot_metric_by_r_train('train.loss', 'Training Loss', 'gpt')
# plot_metric_by_r_train('test.accuracy', 'Validation Accuracy', 'gpt')

if __name__ == "__main__":
    # Plot training and validation metrics for the GPT model
    print("Generating plots for GPT model...")
    plot_metric_by_r_train('train.loss', 'Training Loss', 'gpt')
    plot_metric_by_r_train('test.loss', 'Validation Loss', 'gpt')
    plot_metric_by_r_train('train.accuracy', 'Training Accuracy', 'gpt')
    plot_metric_by_r_train('test.accuracy', 'Validation Accuracy', 'gpt')
    
    # Uncomment to generate plots for LSTM model if available
    print("\nGenerating plots for LSTM model...")
    plot_metric_by_r_train('train.loss', 'Training Loss', 'lstm')
    plot_metric_by_r_train('test.loss', 'Validation Loss', 'lstm')
    plot_metric_by_r_train('train.accuracy', 'Training Accuracy', 'lstm')
    plot_metric_by_r_train('test.accuracy', 'Validation Accuracy', 'lstm')
    
    print("\nAll plots have been saved to the results/q3/ directory.")




