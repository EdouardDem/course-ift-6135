from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import math

# Configuration
models = ["lstm", "gpt"]  # Models to plot
seeds = ["0", "42"]       # Seeds to average over
num_layers_values = [1, 2, 3]  # L values
embedding_sizes = [64, 128, 256]  # d values

# Actual parameter counts excluding embeddings (from log output)
params = {
    "lstm": {
        "1": {
            "64": 35364,
            "128": 136228,
            "256": 534564
        },
        "2": {
            "64": 68388,
            "128": 267812,
            "256": 1059876
        },
        "3": {  
            "64": 101412,
            "128": 399396,
            "256": 1585188
        }
    },
    "gpt": {
        "1": {
            "64": 52004,
            "128": 202276,
            "256": 797732
        },
        "2": {
            "64": 101668,
            "128": 399908,
            "256": 1586212
        },
        "3": {
            "64": 151332,
            "128": 597540,
            "256": 2374692
        }
    }
}

# Define a function to get the parameter count from the params dictionary
def get_parameter_count(model_type, num_layers, embedding_size):
    """
    Get the parameter count for a given model architecture from the params dictionary.
    
    Parameters:
    -----------
    model_type : str
        The model type ('lstm' or 'gpt')
    num_layers : int
        Number of layers (L)
    embedding_size : int
        Embedding size (d)
    
    Returns:
    --------
    int : Number of parameters (excluding embeddings)
    """
    return params[model_type][str(num_layers)][str(embedding_size)]

def load_all_results():
    """
    Load results for all model configurations.
    
    Returns:
    --------
    dict : Dictionary with all results organized by model, L, and d
    """
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for L in num_layers_values:
            all_results[model_name][L] = {}
            
            for d in embedding_sizes:
                # Get actual parameter count from the params dictionary
                num_params = get_parameter_count(model_name, L, d)
                
                # Set up the correct base directory
                if model_name == "lstm":
                    base_dir = Path(f"logs/q5/model={model_name}-optimizer=adamw-n_steps=10000-num_layers={L}-embedding_size={d}-hidden_size={d}")
                else:  # model_name == "gpt"
                    base_dir = Path(f"logs/q5/model={model_name}-optimizer=adamw-n_steps=10000-num_layers={L}-embedding_size={d}")
                
                if not base_dir.exists():
                    print(f"Directory not found: {base_dir}")
                    continue
                
                try:
                    results = load_and_combine_results(base_dir, seeds)
                    if results is not None:
                        # Store results along with parameter count
                        all_results[model_name][L][d] = {
                            'results': results,
                            'params': num_params
                        }
                    else:
                        print(f"No results found for {model_name} with L={L}, d={d}")
                except Exception as e:
                    print(f"Error loading data for {model_name} with L={L}, d={d}: {e}")
    
    return all_results

def extract_metrics(all_results):
    """
    Extract metrics from the pre-computed results for all configurations.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with all results organized by model, L, and d
    
    Returns:
    --------
    dict : Dictionary with extracted metrics
    """
    metrics_data = {}
    
    for model_name in all_results:
        metrics_data[model_name] = {
            'L': [],
            'd': [],
            'params': [],
            'final_train_loss': [],
            'final_val_loss': [],
            'final_train_acc': [],
            'final_val_acc': [],
            'time_to_final_train_loss': [],
            'time_to_final_val_loss': [],
            'time_to_final_train_acc': [],
            'time_to_final_val_acc': []
        }
        
        for L in all_results[model_name]:
            for d in all_results[model_name][L]:
                data = all_results[model_name][L][d]
                results = data['results']['extrema']
                params = data['params']
                
                # Skip if missing key metrics
                if not all(key in results for key in ['min_train_loss', 'min_test_loss', 
                                                     'max_train_accuracy', 'max_test_accuracy',
                                                     'min_train_loss_step', 'min_test_loss_step',
                                                     'max_train_accuracy_step', 'max_test_accuracy_step']):
                    print(f"Missing key metrics for {model_name} with L={L}, d={d}")
                    continue
                
                # Get final metrics (use min for loss, max for accuracy)
                final_train_loss = results['min_train_loss']
                final_val_loss = results['min_test_loss']
                final_train_acc = results['max_train_accuracy']
                final_val_acc = results['max_test_accuracy']
                
                # Get steps to reach these metrics
                time_to_final_train_loss = results['min_train_loss_step']
                time_to_final_val_loss = results['min_test_loss_step']
                time_to_final_train_acc = results['max_train_accuracy_step']
                time_to_final_val_acc = results['max_test_accuracy_step']
                
                # Store data
                metrics_data[model_name]['L'].append(L)
                metrics_data[model_name]['d'].append(d)
                metrics_data[model_name]['params'].append(params)
                metrics_data[model_name]['final_train_loss'].append(final_train_loss)
                metrics_data[model_name]['final_val_loss'].append(final_val_loss)
                metrics_data[model_name]['final_train_acc'].append(final_train_acc)
                metrics_data[model_name]['final_val_acc'].append(final_val_acc)
                metrics_data[model_name]['time_to_final_train_loss'].append(time_to_final_train_loss)
                metrics_data[model_name]['time_to_final_val_loss'].append(time_to_final_val_loss)
                metrics_data[model_name]['time_to_final_train_acc'].append(time_to_final_train_acc)
                metrics_data[model_name]['time_to_final_val_acc'].append(time_to_final_val_acc)
    
    return metrics_data

def plot_metrics_vs_architecture(metrics_data, figsize=(15, 20)):
    """
    Plot metrics as a function of architecture parameters (d and L).
    
    Parameters:
    -----------
    metrics_data : dict
        Dictionary with extracted metrics
    figsize : tuple, optional
        Figure size, default is (15, 20)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q5-b', exist_ok=True)
    
    for model_name in metrics_data:
        # Create a figure with 4 rows and 2 columns for the 8 metrics
        fig, axs = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle(f'{model_name.upper()} Model - Metrics vs Architecture Parameters', fontsize=16)
        
        # Prepare data
        L_values = np.array(metrics_data[model_name]['L'])
        d_values = np.array(metrics_data[model_name]['d'])
        
        # Prepare color map for L values
        unique_L = sorted(set(L_values))
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(unique_L), vmax=max(unique_L))
        
        # Get all unique d values and sort them for x-axis
        all_d = sorted(set(d_values))
        
        # Metrics to plot
        metrics = [
            ('final_train_loss', 'Training Loss', 0, 0, True),
            ('final_val_loss', 'Validation Loss', 0, 1, True),
            ('final_train_acc', 'Training Accuracy', 1, 0, False),
            ('final_val_acc', 'Validation Accuracy', 1, 1, False),
            ('time_to_final_train_loss', 'Time to Final Training Loss', 2, 0, False),
            ('time_to_final_val_loss', 'Time to Final Validation Loss', 2, 1, False),
            ('time_to_final_train_acc', 'Time to Final Training Accuracy', 3, 0, False),
            ('time_to_final_val_acc', 'Time to Final Validation Accuracy', 3, 1, False)
        ]
        
        # For each metric, create a plot with d on x-axis and color for L
        for metric_key, metric_label, row, col, use_log_scale in metrics:
            ax = axs[row, col]
            ax.set_title(metric_label)
            ax.set_xlabel('Embedding Size (d) - $\\log_2$ scale')
            ax.set_ylabel(metric_label)
            
            if use_log_scale:
                ax.set_yscale('log')
            
            # Set x-axis to log2 scale for embedding size
            ax.set_xscale('log', base=2)
            
            # Plot each data point
            for L in unique_L:
                # Filter data for this L value
                mask = (L_values == L)
                x = d_values[mask]
                y = np.array(metrics_data[model_name][metric_key])[mask]
                
                # Sort by d to ensure connected lines make sense
                sorted_indices = np.argsort(x)
                x = x[sorted_indices]
                y = y[sorted_indices]
                
                color = cmap(norm(L))
                ax.plot(x, y, 'o-', color=color, label=f'L={L}')
            
            # Set x-axis ticks to show actual d values
            ax.set_xticks(all_d)
            ax.set_xticklabels(all_d)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend if it's the first plot
            if row == 0 and col == 0:
                ax.legend(title="Number of Layers (L)")
        
        # Adjust layout and save
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        fig.savefig(f'results/q5-b/{model_name}_metrics_vs_architecture.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Architecture metrics plot saved as results/q5-b/{model_name}_metrics_vs_architecture.png")

def plot_metrics_vs_params(metrics_data, figsize=(15, 20)):
    """
    Plot metrics as a function of parameter count with a single curve per metric.
    
    Parameters:
    -----------
    metrics_data : dict
        Dictionary with extracted metrics
    figsize : tuple, optional
        Figure size, default is (15, 20)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q5-b', exist_ok=True)
    
    for model_name in metrics_data:
        # Create a figure with 4 rows and 2 columns for the 8 metrics
        fig, axs = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle(f'{model_name.upper()} Model - Metrics vs Parameter Count', fontsize=16)
        
        # Get parameter values and sort them
        param_values = np.array(metrics_data[model_name]['params'])
        
        # Metrics to plot
        metrics = [
            ('final_train_loss', 'Training Loss', 0, 0, True),
            ('final_val_loss', 'Validation Loss', 0, 1, True),
            ('final_train_acc', 'Training Accuracy', 1, 0, False),
            ('final_val_acc', 'Validation Accuracy', 1, 1, False),
            ('time_to_final_train_loss', 'Time to Final Training Loss', 2, 0, False),
            ('time_to_final_val_loss', 'Time to Final Validation Loss', 2, 1, False),
            ('time_to_final_train_acc', 'Time to Final Training Accuracy', 3, 0, False),
            ('time_to_final_val_acc', 'Time to Final Validation Accuracy', 3, 1, False)
        ]
        
        # For each metric, create a single plot against parameter count
        for metric_key, metric_label, row, col, use_log_scale in metrics:
            ax = axs[row, col]
            ax.set_title(metric_label)
            ax.set_xlabel('Parameter Count (excluding embeddings)')
            ax.set_ylabel(metric_label)
            
            # Get metric values
            y_values = np.array(metrics_data[model_name][metric_key])
            
            # Sort both arrays by parameter count
            sort_idx = np.argsort(param_values)
            x_sorted = param_values[sort_idx]
            y_sorted = y_values[sort_idx]
            
            # Set scales
            if use_log_scale:
                ax.set_yscale('log')
            ax.set_xscale('log')
            
            # Plot single line with markers
            ax.plot(x_sorted, y_sorted, 'o-', color='blue', markersize=8)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add value annotations
            for x, y in zip(x_sorted, y_sorted):
                ax.annotate(f'{y:.3f}', 
                          (x, y),
                          textcoords="offset points",
                          xytext=(0,10),
                          ha='center')
        
        # Adjust layout and save
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        fig.savefig(f'results/q5-b/{model_name}_metrics_vs_params.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Parameter count metrics plot saved as results/q5-b/{model_name}_metrics_vs_params.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q5-b', exist_ok=True)
    
    print("Loading all model results...")
    all_results = load_all_results()
    
    print("Extracting metrics from pre-computed results...")
    metrics_data = extract_metrics(all_results)
    
    print("Plotting metrics vs architecture parameters (d and L)...")
    plot_metrics_vs_architecture(metrics_data)
    
    print("Plotting metrics vs parameter count...")
    plot_metrics_vs_params(metrics_data)
    
    print("\nAll plots have been saved to the results/q5-b/ directory.") 