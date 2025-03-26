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

def load_results(model_name, seeds):
    """
    Load results for a given model with both binary and ternary operations.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list
        List of seeds to average over
    
    Returns:
    --------
    dict or None
        The loaded results or None if loading fails
    """
    # Set up the base directory with combined operation_orders=2,3
    base_dir = Path(f"logs/q4/model={model_name}-optimizer=adamw-n_steps=10000-operation_orders=2,3-p=11-reduction=none")
    
    # Skip if directory doesn't exist
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return None
    
    try:
        # Use load_and_combine_results to get combined statistics across seeds
        results = load_and_combine_results(base_dir, seeds)
        
        if results is None:
            print(f"No results found for {model_name}")
            return None
        
        return results
    
    except Exception as e:
        print(f"Error loading data for {model_name}: {e}")
        return None

def plot_operation_metrics(model_name, results, figsize=(12, 10)):
    """
    Plot training and validation metrics (loss and accuracy) over time for a given model,
    separated by operation order (binary and ternary).
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    results : dict
        The loaded results containing operation-specific metrics
    figsize : tuple, optional
        Figure size, default is (12, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    if results is None:
        print(f"No results available for {model_name}")
        return
    
    # Create a 2x2 subplot for loss and accuracy, separated by operation order
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name.upper()} Model - Binary vs Ternary Operations', fontsize=16)
    
    # Get steps for x-axis
    steps = results['all_steps'][0] if 'all_steps' in results and len(results['all_steps']) > 0 else []
    
    if not steps:
        print(f"No steps data found for {model_name}")
        return
    
    # Plot training loss by operation order
    ax = axs[0, 0]
    ax.set_title('Training Loss by Operation Order')
    ax.set_xlabel('Training Steps (t)')
    ax.set_ylabel('Loss')
    
    # Plot binary (order 2) loss
    if 'train' in results and 'loss_by_order_2' in results['train']:
        y_mean = results['train']['loss_by_order_2']['mean']
        y_std = results['train']['loss_by_order_2']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'b-', label='Binary (Order 2)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='b', alpha=0.3)
    
    # Plot ternary (order 3) loss
    if 'train' in results and 'loss_by_order_3' in results['train']:
        y_mean = results['train']['loss_by_order_3']['mean']
        y_std = results['train']['loss_by_order_3']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'r-', label='Ternary (Order 3)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='r', alpha=0.3)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot validation loss by operation order
    ax = axs[0, 1]
    ax.set_title('Validation Loss by Operation Order')
    ax.set_xlabel('Training Steps (t)')
    ax.set_ylabel('Loss')
    
    # Plot binary (order 2) loss
    if 'test' in results and 'loss_by_order_2' in results['test']:
        y_mean = results['test']['loss_by_order_2']['mean']
        y_std = results['test']['loss_by_order_2']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'b-', label='Binary (Order 2)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='b', alpha=0.3)
    
    # Plot ternary (order 3) loss
    if 'test' in results and 'loss_by_order_3' in results['test']:
        y_mean = results['test']['loss_by_order_3']['mean']
        y_std = results['test']['loss_by_order_3']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'r-', label='Ternary (Order 3)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='r', alpha=0.3)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot training accuracy by operation order
    ax = axs[1, 0]
    ax.set_title('Training Accuracy by Operation Order')
    ax.set_xlabel('Training Steps (t)')
    ax.set_ylabel('Accuracy')
    
    # Plot binary (order 2) accuracy
    if 'train' in results and 'acc_by_order_2' in results['train']:
        y_mean = results['train']['acc_by_order_2']['mean']
        y_std = results['train']['acc_by_order_2']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'b-', label='Binary (Order 2)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='b', alpha=0.3)
    
    # Plot ternary (order 3) accuracy
    if 'train' in results and 'acc_by_order_3' in results['train']:
        y_mean = results['train']['acc_by_order_3']['mean']
        y_std = results['train']['acc_by_order_3']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'r-', label='Ternary (Order 3)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='r', alpha=0.3)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot validation accuracy by operation order
    ax = axs[1, 1]
    ax.set_title('Validation Accuracy by Operation Order')
    ax.set_xlabel('Training Steps (t)')
    ax.set_ylabel('Accuracy')
    
    # Plot binary (order 2) accuracy
    if 'test' in results and 'acc_by_order_2' in results['test']:
        y_mean = results['test']['acc_by_order_2']['mean']
        y_std = results['test']['acc_by_order_2']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'b-', label='Binary (Order 2)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='b', alpha=0.3)
    
    # Plot ternary (order 3) accuracy
    if 'test' in results and 'acc_by_order_3' in results['test']:
        y_mean = results['test']['acc_by_order_3']['mean']
        y_std = results['test']['acc_by_order_3']['std']
        # Ensure lengths match
        min_len = min(len(steps), len(y_mean))
        ax.plot(steps[:min_len], y_mean[:min_len], 'r-', label='Ternary (Order 3)')
        ax.fill_between(steps[:min_len], 
                        y_mean[:min_len] - y_std[:min_len], 
                        y_mean[:min_len] + y_std[:min_len], 
                        color='r', alpha=0.3)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout and save figure
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    fig.savefig(f'results/q4-b/{model_name}_order_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Order comparison plot saved as results/q4-b/{model_name}_order_comparison.png")

def plot_all_models_comparison(model_results, figsize=(15, 12)):
    """
    Create comparison plots showing both models and both operation orders.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with results for each model
    figsize : tuple, optional
        Figure size, default is (15, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Create a 2x2 subplot for loss and accuracy, training and validation
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('LSTM vs GPT: Binary and Ternary Operations', fontsize=16)
    
    # Subplot titles and metrics
    subplot_configs = [
        (0, 0, 'Training Loss', 'train', 'loss'),
        (0, 1, 'Validation Loss', 'test', 'loss'),
        (1, 0, 'Training Accuracy', 'train', 'acc'),
        (1, 1, 'Validation Accuracy', 'test', 'acc')
    ]
    
    # Model and operation styles
    model_styles = {
        'lstm': {'linestyle': '-', 'marker': 'o'},
        'gpt': {'linestyle': '--', 'marker': 's'}
    }
    
    operation_colors = {
        '2': 'blue',   # Binary
        '3': 'red'     # Ternary
    }
    
    # Plot each metric
    for row, col, title, split, metric_base in subplot_configs:
        ax = axs[row, col]
        ax.set_title(title)
        ax.set_xlabel('Training Steps (t)')
        ax.set_ylabel('Loss' if 'Loss' in title else 'Accuracy')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        has_data = False
        
        # For each model
        for model_name, results in model_results.items():
            if results is None:
                continue
                
            steps = results['all_steps'][0] if 'all_steps' in results and len(results['all_steps']) > 0 else []
            
            if not steps:
                continue
            
            # For each operation order
            for order in ['2', '3']:
                metric_key = f"{metric_base}_by_order_{order}"
                
                if split in results and metric_key in results[split]:
                    y_mean = results[split][metric_key]['mean']
                    y_std = results[split][metric_key]['std']
                    
                    # Ensure lengths match
                    min_len = min(len(steps), len(y_mean))
                    
                    # Plot with model and operation specific styles
                    label = f"{model_name.upper()} - {'Binary' if order == '2' else 'Ternary'}"
                    color = operation_colors[order]
                    linestyle = model_styles[model_name]['linestyle']
                    marker = model_styles[model_name]['marker']
                    
                    ax.plot(steps[:min_len], y_mean[:min_len], 
                           color=color, linestyle=linestyle, marker=marker, markevery=len(steps)//10, 
                           label=label, alpha=0.8)
                    
                    ax.fill_between(steps[:min_len], 
                                   y_mean[:min_len] - y_std[:min_len], 
                                   y_mean[:min_len] + y_std[:min_len], 
                                   color=color, alpha=0.2)
                    
                    has_data = True
        
        # Add legend if we have data
        if has_data:
            ax.legend(loc='best')
    
    # Adjust layout and save figure
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    fig.savefig(f'results/q4-b/all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All models comparison plot saved as results/q4-b/all_models_comparison.png")

def plot_metrics_by_operation_order(model_results, figsize=(15, 10)):
    """
    Create separate plots for binary and ternary operations comparing both models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with results for each model
    figsize : tuple, optional
        Figure size, default is (15, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    # Operations to compare
    operations = {
        '2': 'Binary',
        '3': 'Ternary'
    }
    
    # For each operation
    for order, operation_name in operations.items():
        # Create a 2x2 subplot for loss and accuracy, training and validation
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{operation_name} Operations: LSTM vs GPT', fontsize=16)
        
        # Subplot titles and metrics
        subplot_configs = [
            (0, 0, f'Training Loss - {operation_name}', 'train', f'loss_by_order_{order}'),
            (0, 1, f'Validation Loss - {operation_name}', 'test', f'loss_by_order_{order}'),
            (1, 0, f'Training Accuracy - {operation_name}', 'train', f'acc_by_order_{order}'),
            (1, 1, f'Validation Accuracy - {operation_name}', 'test', f'acc_by_order_{order}')
        ]
        
        # Colors for models
        model_colors = {
            'lstm': 'blue',
            'gpt': 'red'
        }
        
        # Plot each metric
        for row, col, title, split, metric in subplot_configs:
            ax = axs[row, col]
            ax.set_title(title)
            ax.set_xlabel('Training Steps (t)')
            ax.set_ylabel('Loss' if 'Loss' in title else 'Accuracy')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            has_data = False
            
            # For each model
            for model_name, results in model_results.items():
                if results is None:
                    continue
                    
                steps = results['all_steps'][0] if 'all_steps' in results and len(results['all_steps']) > 0 else []
                
                if not steps:
                    continue
                
                if split in results and metric in results[split]:
                    y_mean = results[split][metric]['mean']
                    y_std = results[split][metric]['std']
                    
                    # Ensure lengths match
                    min_len = min(len(steps), len(y_mean))
                    
                    # Plot with model specific color
                    color = model_colors[model_name]
                    ax.plot(steps[:min_len], y_mean[:min_len], color=color, label=model_name.upper())
                    ax.fill_between(steps[:min_len], 
                                   y_mean[:min_len] - y_std[:min_len], 
                                   y_mean[:min_len] + y_std[:min_len], 
                                   color=color, alpha=0.3)
                    
                    has_data = True
            
            # Add legend if we have data
            if has_data:
                ax.legend()
        
        # Adjust layout and save figure
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        fig.savefig(f'results/q4-b/{operation_name.lower()}_operations.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"{operation_name} operations comparison plot saved as results/q4-b/{operation_name.lower()}_operations.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q4-b', exist_ok=True)
    
    print("Loading results for all models...")
    model_results = {}
    for model in models:
        print(f"Loading data for {model.upper()} model...")
        model_results[model] = load_results(model, seeds)
    
    print("\nGenerating per-model plots...")
    for model in models:
        if model_results[model] is not None:
            print(f"Generating plots for {model.upper()} model...")
            plot_operation_metrics(model, model_results[model])
    
    print("\nGenerating all models comparison plot...")
    plot_all_models_comparison(model_results)
    
    print("\nGenerating operation-specific comparison plots...")
    plot_metrics_by_operation_order(model_results)
    
    print("\nAll plots have been saved to the results/q4-b/ directory.") 