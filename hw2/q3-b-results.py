from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import get_extrema_performance_steps, load_and_combine_results

# Define r_train values and seeds
r_trains = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = ["0", "42"]  # Define seeds to use

def extract_extrema_metrics(model_name, r_trains, seeds):
    """
    Extract extrema metrics for a model across different r_train values.
    
    Parameters:
    -----------
    model_name : str
        The name of the model ('lstm' or 'gpt')
    r_trains : list
        List of r_train values to process
    seeds : list
        List of seeds to average over
    
    Returns:
    --------
    dict
        Dictionary with r_train values as keys and metric data as values
    """
    metrics_by_r_train = {}
    
    for r_train in r_trains:
        base_dir = Path(f"logs/q3/model={model_name}-optimizer=adamw-n_steps=10000-r_train={r_train}")
        
        # Skip if directory doesn't exist
        if not base_dir.exists():
            print(f"Directory not found: {base_dir}")
            continue
        
        try:
            # Load combined results for this r_train value
            results = load_and_combine_results(base_dir, seeds)
            
            if results is None or 'extrema' not in results:
                print(f"No extrema metrics found for {model_name} with r_train={r_train}")
                continue
            
            # Store the extrema metrics for this r_train
            metrics_by_r_train[r_train] = results['extrema']
            
        except Exception as e:
            print(f"Error processing {model_name} with r_train={r_train}: {e}")
    
    return metrics_by_r_train

def plot_metric_pair_grid(metric_type, model_names=['gpt', 'lstm'], log_scale=False, figsize=(15, 6)):
    """
    Plot train and validation metrics side by side in a 1x2 grid, as a function of r_train.
    
    Parameters:
    -----------
    metric_type : str
        The type of metric to plot: 'loss', 'accuracy' or 'steps'
    model_names : list, optional
        List of model names to include in the plot, default is ['gpt', 'lstm']
    log_scale : bool, optional
        Whether to use log scale for y-axis, default is False
    figsize : tuple, optional
        Figure size, default is (15, 6)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q3-b', exist_ok=True)
    
    # Set up metric pairs and labels based on metric_type
    if metric_type == 'loss':
        metric_pair = ('min_train_loss', 'min_test_loss')
        y_labels = ('$\\mathcal{L}_{\\text{train}}$', '$\\mathcal{L}_{\\text{val}}$')
        title = 'Loss vs $r_{train}$'
        filename = 'loss_grid'
    elif metric_type == 'accuracy':
        metric_pair = ('max_train_accuracy', 'max_test_accuracy')
        y_labels = ('$\\mathcal{A}_{\\text{train}}$', '$\\mathcal{A}_{\\text{val}}$')
        title = 'Accuracy vs $r_{train}$'
        filename = 'accuracy_grid'
    elif metric_type == 'loss_steps':
        metric_pair = ('min_train_loss_step', 'min_test_loss_step')
        y_labels = ('$t_f(\\mathcal{L}_{\\text{train}})$', '$t_f(\\mathcal{L}_{\\text{val}})$')
        title = 'Best Loss Time vs $r_{train}$'
        filename = 'loss_steps_grid'
    elif metric_type == 'accuracy_steps':
        metric_pair = ('max_train_accuracy_step', 'max_test_accuracy_step')
        y_labels = ('$t_f(\\mathcal{A}_{\\text{train}})$', '$t_f(\\mathcal{A}_{\\text{val}})$')
        title = 'Best Accuracy Time vs $r_{train}$'
        filename = 'accuracy_steps_grid'
    else:
        print(f"Unknown metric type: {metric_type}")
        return
    
    # Create figure with 1x2 grid
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Define colors and markers for different models
    colors = {'gpt': 'blue', 'lstm': 'red'}
    markers = {'gpt': 'o', 'lstm': 's'}
    
    # Track if we have any data to plot
    has_data = False
    
    # Plot each metric in its corresponding subplot
    for i, (metric_name, y_label) in enumerate(zip(metric_pair, y_labels)):
        ax = axs[i]
        ax.set_xlabel('$r_{train}$')
        ax.set_ylabel(y_label)
        ax.set_title(y_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if log_scale and metric_type == 'loss':
            ax.set_yscale('log')
            
        subplot_has_data = False
        
        for model_name in model_names:
            # Extract metrics for this model
            metrics_by_r_train = extract_extrema_metrics(model_name, r_trains, seeds)
            
            if not metrics_by_r_train:
                print(f"No data available for {model_name}")
                continue
            
            # Prepare data for plotting
            x_values = []
            y_values = []
            y_errors = []
            
            for r_train in sorted(metrics_by_r_train.keys()):
                if metric_name in metrics_by_r_train[r_train]:
                    x_values.append(r_train)
                    y_values.append(metrics_by_r_train[r_train][metric_name])
                    
                    # Check if std is available for this metric
                    std_key = f"{metric_name}_std"
                    if std_key in metrics_by_r_train[r_train]:
                        y_errors.append(metrics_by_r_train[r_train][std_key])
                    else:
                        y_errors.append(0)
            
            if x_values:
                subplot_has_data = True
                has_data = True
                ax.errorbar(x_values, y_values, yerr=y_errors, 
                          color=colors[model_name], marker=markers[model_name], 
                          capsize=5, label=f"{model_name.upper()}")
        
        if subplot_has_data:
            ax.legend()
    
    if not has_data:
        plt.close(fig)
        print(f"No data to plot for {metric_type} metrics")
        return
    
    # Adjust layout and save figure
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    log_suffix = '_log' if log_scale else ''
    plt.savefig(f'results/q3-b/{filename}{log_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Grid plot saved as results/q3-b/{filename}{log_suffix}.png")

if __name__ == "__main__":
    print("Generating comparative performance grid plots as a function of r_train...")
    
    # Plot loss metrics (log scale)
    print("\nPlotting loss metrics grid...")
    plot_metric_pair_grid('loss', log_scale=True)
    
    # Plot accuracy metrics
    print("\nPlotting accuracy metrics grid...")
    plot_metric_pair_grid('accuracy')
    
    # Plot time step metrics for loss
    print("\nPlotting best loss time grid...")
    plot_metric_pair_grid('loss_steps')
    
    # Plot time step metrics for accuracy
    print("\nPlotting best accuracy time grid...")
    plot_metric_pair_grid('accuracy_steps')
    
    print("\nAll plots have been saved to the results/q3-b/ directory.") 