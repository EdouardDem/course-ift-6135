from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

# Configuration
models = ["lstm", "gpt"]  # Models to plot
seeds = ["0", "42"]       # Seeds to average over
operation_orders = {
    "binary": "2",
    "ternary": "3"
}

def plot_operation_metrics(model_name, operation, seeds, figsize=(12, 10)):
    """
    Plot training and validation metrics (loss and accuracy) over time for a given model
    and operation order (binary or ternary).
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    operation : str
        The operation order ('binary' or 'ternary')
    seeds : list
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (12, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Set up the base directory
    operation_order = operation_orders[operation]
    base_dir = Path(f"logs/q4/model={model_name}-optimizer=adamw-n_steps=10000-operation_orders={operation_order}-p=11")
    
    # Skip if directory doesn't exist
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return None
    
    try:
        # Use load_and_combine_results to get combined statistics across seeds
        results = load_and_combine_results(base_dir, seeds)
        
        if results is None:
            print(f"No results found for {model_name} with {operation} operations")
            return None
        
        # Create a 2x2 subplot for loss and accuracy, training and validation
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        operation_name = "Binary" if operation == "binary" else "Ternary"
        fig.suptitle(f'{model_name.upper()} Model - {operation_name} Operations', fontsize=16)
        
        # Get steps for x-axis
        steps = results['all_steps'][0]  # Use the first seed's steps
        
        # Plot training loss
        if 'train' in results and 'loss' in results['train']:
            y_mean = results['train']['loss']['mean']
            y_std = results['train']['loss']['std']
            axs[0, 0].plot(steps, y_mean, 'b-', label='Train Loss')
            axs[0, 0].fill_between(steps, y_mean - y_std, y_mean + y_std, color='b', alpha=0.3)
            axs[0, 0].set_title('Training Loss')
            axs[0, 0].set_xlabel('Training Steps (t)')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].grid(True, linestyle='--', alpha=0.7)
            axs[0, 0].legend()
        
        # Plot validation loss
        if 'test' in results and 'loss' in results['test']:
            y_mean = results['test']['loss']['mean']
            y_std = results['test']['loss']['std']
            axs[0, 1].plot(steps, y_mean, 'r-', label='Validation Loss')
            axs[0, 1].fill_between(steps, y_mean - y_std, y_mean + y_std, color='r', alpha=0.3)
            axs[0, 1].set_title('Validation Loss')
            axs[0, 1].set_xlabel('Training Steps (t)')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)
            axs[0, 1].legend()
        
        # Plot training accuracy
        if 'train' in results and 'accuracy' in results['train']:
            y_mean = results['train']['accuracy']['mean']
            y_std = results['train']['accuracy']['std']
            axs[1, 0].plot(steps, y_mean, 'g-', label='Train Accuracy')
            axs[1, 0].fill_between(steps, y_mean - y_std, y_mean + y_std, color='g', alpha=0.3)
            axs[1, 0].set_title('Training Accuracy')
            axs[1, 0].set_xlabel('Training Steps (t)')
            axs[1, 0].set_ylabel('Accuracy')
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
            axs[1, 0].legend()
        
        # Plot validation accuracy
        if 'test' in results and 'accuracy' in results['test']:
            y_mean = results['test']['accuracy']['mean']
            y_std = results['test']['accuracy']['std']
            axs[1, 1].plot(steps, y_mean, 'm-', label='Validation Accuracy')
            axs[1, 1].fill_between(steps, y_mean - y_std, y_mean + y_std, color='m', alpha=0.3)
            axs[1, 1].set_title('Validation Accuracy')
            axs[1, 1].set_xlabel('Training Steps (t)')
            axs[1, 1].set_ylabel('Accuracy')
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
            axs[1, 1].legend()

        # Adjust layout and save figure
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        fig.savefig(f'results/q4-b/{model_name}_{operation}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Metrics plot saved as results/q4-b/{model_name}_{operation}.png")
        
        return results
    
    except Exception as e:
        print(f"Error processing {model_name} with {operation} operations: {e}")
        return None


def plot_binary_vs_ternary(model_name, seeds, figsize=(15, 12)):
    """
    Create a single grid comparing binary vs ternary operations for a specific model.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (15, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Load data for binary and ternary operations
    binary_results = plot_operation_metrics(model_name, "binary", seeds)
    ternary_results = plot_operation_metrics(model_name, "ternary", seeds)
    
    if binary_results is None and ternary_results is None:
        print(f"No data available for {model_name} with either binary or ternary operations")
        return
    
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} Model: Binary vs Ternary Operations', fontsize=16)
    
    # Metrics and positions in the grid
    grid_positions = {
        ('train', 'loss'): (0, 0),      # Training Loss: top-left
        ('test', 'loss'): (0, 1),       # Validation Loss: top-right  
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
    
    # Colors for operations
    colors = {'binary': 'blue', 'ternary': 'red'}
    
    # Results dictionary
    operation_results = {
        'binary': binary_results,
        'ternary': ternary_results
    }
    
    # Plot each metric on its corresponding subplot
    for (dataset, metric), (row, col) in grid_positions.items():
        ax = axs[row, col]
        
        # Set title and labels
        ax.set_title(titles[(dataset, metric)])
        ax.set_xlabel('Training Steps (t)')
        ax.set_ylabel(y_labels[metric])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        has_data = False
        
        # Plot data for each operation
        for operation, results in operation_results.items():
            if results is not None and dataset in results and metric in results[dataset]:
                steps = results['all_steps'][0]
                y_mean = results[dataset][metric]['mean']
                y_std = results[dataset][metric]['std']
                
                operation_name = "Binary" if operation == "binary" else "Ternary"
                ax.plot(steps, y_mean, color=colors[operation], label=f'{operation_name}')
                ax.fill_between(steps, y_mean - y_std, y_mean + y_std, color=colors[operation], alpha=0.3)
                has_data = True
        
        # Add legend if we have data
        if has_data:
            ax.legend()
    
    # Adjust layout and save figure
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    fig.savefig(f'results/q4-b/{model_name}_binary_vs_ternary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Binary vs Ternary comparison plot saved as results/q4-b/{model_name}_binary_vs_ternary.png")


def plot_all_comparisons(seeds, figsize=(15, 10)):
    """
    Create comprehensive comparison plots of binary vs ternary operations for all metrics and models.
    
    Parameters:
    -----------
    seeds : list
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (15, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Metrics to compare
    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy')
    ]
    
    # Datasets to compare
    datasets = [
        ('train', 'Training'),
        ('test', 'Validation')
    ]
    
    # Setup model and operation data
    all_results = {}
    
    # Load all data
    for model_name in models:
        all_results[model_name] = {}
        for operation in operation_orders:
            operation_order = operation_orders[operation]
            base_dir = Path(f"logs/q4/model={model_name}-optimizer=adamw-n_steps=10000-operation_orders={operation_order}-p=11")
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            try:
                results = load_and_combine_results(base_dir, seeds)
                if results is not None:
                    all_results[model_name][operation] = results
                else:
                    print(f"No results found for {model_name} with {operation} operations")
            except Exception as e:
                print(f"Error loading data for {model_name} with {operation} operations: {e}")
    
    # For each metric and dataset, create a comparison plot showing all models and operations
    for metric, metric_label in metrics:
        for dataset, dataset_label in datasets:
            # Create figure with 1x2 subplots (one for each model)
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(f'{dataset_label} {metric_label}: Binary vs Ternary Operations', fontsize=16)
            
            # Colors for operations
            colors = {'binary': 'blue', 'ternary': 'red'}
            
            # Process each model
            for i, model_name in enumerate(models):
                ax = axs[i]
                ax.set_title(f'{model_name.upper()} Model')
                ax.set_xlabel('Training Steps (t)')
                ax.set_ylabel(metric_label)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                has_data = False
                
                # Plot each operation for this model
                if model_name in all_results:
                    for operation, results in all_results[model_name].items():
                        if dataset in results and metric in results[dataset]:
                            steps = results['all_steps'][0]
                            y_mean = results[dataset][metric]['mean']
                            y_std = results[dataset][metric]['std']
                            
                            operation_name = "Binary" if operation == "binary" else "Ternary"
                            ax.plot(steps, y_mean, color=colors[operation], label=f'{operation_name}')
                            ax.fill_between(steps, y_mean - y_std, y_mean + y_std, color=colors[operation], alpha=0.3)
                            has_data = True
                
                # Add legend if we have data
                if has_data:
                    ax.legend()
            
            # Adjust layout and save figure
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
            fig.savefig(f'results/q4-b/comparison_{dataset}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Comparison plot saved as results/q4-b/comparison_{dataset}_{metric}.png")


if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Plot individual operation metrics for each model
    for model in models:
        for operation in operation_orders:
            print(f"\nGenerating plots for {model.upper()} model with {operation} operations...")
            plot_operation_metrics(model, operation, seeds)
    
    # Plot binary vs ternary comparisons for each model
    for model in models:
        print(f"\nGenerating binary vs ternary comparison for {model.upper()} model...")
        plot_binary_vs_ternary(model, seeds)
    
    # Plot comprehensive comparisons across all models and operations
    print("\nGenerating comprehensive comparison plots...")
    plot_all_comparisons(seeds)
    
    print("\nAll plots have been saved to the results/q4-b/ directory.") 