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

def plot_extrema_metric(metric_name, y_label, model_names=['gpt', 'lstm'], log_scale=False, figsize=(10, 6)):
    """
    Plot a specific extrema metric as a function of r_train for different models.
    
    Parameters:
    -----------
    metric_name : str
        The metric to plot (e.g., 'min_train_loss', 'max_test_accuracy')
    y_label : str
        The label for the y-axis
    model_names : list, optional
        List of model names to include in the plot, default is ['gpt', 'lstm']
    log_scale : bool, optional
        Whether to use log scale for y-axis, default is False
    figsize : tuple, optional
        Figure size, default is (10, 6)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q3-b', exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Define colors and markers for different models
    colors = {'gpt': 'blue', 'lstm': 'red'}
    markers = {'gpt': 'o', 'lstm': 's'}
    
    # Track if we have any data to plot
    has_data = False
    
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
            has_data = True
            plt.errorbar(x_values, y_values, yerr=y_errors, 
                         color=colors[model_name], marker=markers[model_name], 
                         capsize=5, label=f"{model_name.upper()}")
    
    if not has_data:
        plt.close()
        print(f"No data to plot for metric {metric_name}")
        return
    
    # Configure the plot
    plt.xlabel('$r_{train}$')
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs $r_{{train}}$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set y-axis to log scale if requested
    if log_scale:
        plt.yscale('log')
    
    # Save the figure
    safe_metric = metric_name.replace('_', '-')
    log_suffix = '_log' if log_scale else ''
    plt.savefig(f'results/q3-b/{safe_metric}{log_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as results/q3-b/{safe_metric}{log_suffix}.png")

if __name__ == "__main__":
    print("Generating comparative performance plots as a function of r_train...")
    
    # Plot loss metrics (log scale)
    print("\nPlotting loss metrics...")
    plot_extrema_metric('min_train_loss', '$\\mathcal{L}_{\\text{train}}$', log_scale=True)
    plot_extrema_metric('min_test_loss', '$\\mathcal{L}_{\\text{val}}$', log_scale=True)
    
    # Plot accuracy metrics
    print("\nPlotting accuracy metrics...")
    plot_extrema_metric('max_train_accuracy', '$\\mathcal{A}_{\\text{train}}$')
    plot_extrema_metric('max_test_accuracy', '$\\mathcal{A}_{\\text{val}}$')
    
    # Plot time step metrics for loss
    print("\nPlotting time step metrics for loss...")
    plot_extrema_metric('min_train_loss_step', '$t_f(\\mathcal{L}_{\\text{train}})$')
    plot_extrema_metric('min_test_loss_step', '$t_f(\\mathcal{L}_{\\text{val}})$')
    
    # Plot time step metrics for accuracy
    print("\nPlotting time step metrics for accuracy...")
    plot_extrema_metric('max_train_accuracy_step', '$t_f(\\mathcal{A}_{\\text{train}})$')
    plot_extrema_metric('max_test_accuracy_step', '$t_f(\\mathcal{A}_{\\text{val}})$')
    
    print("\nAll plots have been saved to the results/q3-b/ directory.") 