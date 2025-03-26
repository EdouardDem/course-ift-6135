from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results

# Configuration
models = ["lstm", "gpt"]  # Models to plot
seeds = ["0", "42"]       # Seeds to average over

def plot_metrics_over_time(model_name, seeds, figsize=(12, 10)):
    """
    Plot training and validation metrics (loss and accuracy) over time for a given model configuration.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    seeds : list, optional
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (12, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-a', exist_ok=True)
    
    # Set up the base directory
    base_dir = Path(f"logs/q4/model={model_name}-optimizer=adamw-n_steps=10000-operation_orders=2,3-p=11")
    
    # Skip if directory doesn't exist
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    try:
        # Use load_and_combine_results to get combined statistics across seeds
        results = load_and_combine_results(base_dir, seeds)
        
        if results is None:
            print(f"No results found for {model_name}")
            return
        
        # Create a 2x2 subplot for loss and accuracy, training and validation
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{model_name.upper()} Model (optimizer=adamw, operation_orders=[2,3] , p=11)', fontsize=16)
        
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
        fig.savefig(f'results/q4-a/{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Metrics plot saved as results/q4-a/{model_name}.png")
    
    except Exception as e:
        print(f"Error processing {model_name}: {e}")


def plot_model_comparison(seeds, figsize=(15, 10)):
    """
    Create plots comparing LSTM and GPT models side by side for each metric.
    
    Parameters:
    -----------
    seeds : list, optional
        List of seeds to average over
    figsize : tuple, optional
        Figure size, default is (15, 10)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q4-a', exist_ok=True)
    
    # Metrics to compare
    metrics = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
    ]
    
    # Datasets to compare
    datasets = [
        ('train', 'Training'),
        ('test', 'Validation')
    ]
    
    for metric, metric_label in metrics:
        for dataset, dataset_label in datasets:
            fig, ax = plt.subplots(figsize=figsize)
            
            colors = {'lstm': 'blue', 'gpt': 'red'}
            has_data = False
            
            for model_name in models:
                base_dir = Path(f"logs/q4/model={model_name}-optimizer=adamw-n_steps=10000-operation_orders=2,3-p=11")
                
                if not base_dir.exists():
                    print(f"Directory not found: {base_dir}")
                    continue
                
                try:
                    results = load_and_combine_results(base_dir, seeds)
                    
                    if results is None:
                        print(f"No results found for {model_name}")
                        continue
                    
                    if dataset in results and metric in results[dataset]:
                        steps = results['all_steps'][0]
                        y_mean = results[dataset][metric]['mean']
                        y_std = results[dataset][metric]['std']
                        
                        ax.plot(steps, y_mean, color=colors[model_name], label=f'{model_name.upper()}')
                        ax.fill_between(steps, y_mean - y_std, y_mean + y_std, color=colors[model_name], alpha=0.3)
                        has_data = True
                
                except Exception as e:
                    print(f"Error processing {model_name} for {dataset}.{metric}: {e}")
            
            if not has_data:
                plt.close(fig)
                continue
            
            ax.set_title(f'{dataset_label} {metric_label} Comparison: LSTM vs GPT')
            ax.set_xlabel('Training Steps (t)')
            ax.set_ylabel(metric_label)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save the comparison figure
            fig.tight_layout()
            fig.savefig(f'results/q4-a/comparison_{dataset}_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Comparison plot saved as results/q4-a/comparison_{dataset}_{metric}.png")


if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q4-a', exist_ok=True)
    
    # Plot individual model metrics
    for model in models:
        print(f"\nGenerating plots for {model.upper()} model...")
        plot_metrics_over_time(model, seeds)
    
    # Plot model comparisons
    print("\nGenerating model comparison plots...")
    plot_model_comparison(seeds)
    
    print("\nAll plots have been saved to the results/q4-a/ directory.") 