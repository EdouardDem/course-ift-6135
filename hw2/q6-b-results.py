from pathlib import Path
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
batch_sizes = [32, 64, 128, 256, 512]  # B values
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # α values

def load_all_results():
    """
    Load results for all model configurations.
    
    Returns:
    --------
    dict : Dictionary with all results organized by model, batch size, and alpha
    """
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for batch_size in batch_sizes:
            all_results[model_name][batch_size] = {}
            
            # Load the full steps data (alpha=1.0)
            base_dir = Path(f"logs/q6/model={model_name}-optimizer=adamw-n_steps=20001-train_batch_size={batch_size}")
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            try:
                results = load_and_combine_results(base_dir, seeds)
                if results is not None:
                    # Store the results
                    all_results[model_name][batch_size][1.0] = results
                    
                    # For the remaining alpha values, we'll use the same data but truncated
                    if 'all_steps' in results and len(results['all_steps']) > 0:
                        total_steps = len(results['all_steps'][0])
                        
                        # Create entries for other alpha values by truncating the data
                        for alpha in alpha_values[:-1]:  # Excluding 1.0 which we already processed
                            # Truncate the data according to alpha
                            truncated_results = truncate_results(results, alpha, total_steps)
                            all_results[model_name][batch_size][alpha] = truncated_results
                    else:
                        print(f"No steps data for {model_name} with batch_size={batch_size}")
                else:
                    print(f"No results found for {model_name} with batch_size={batch_size}")
            except Exception as e:
                print(f"Error loading data for {model_name} with batch_size={batch_size}: {e}")
    
    return all_results

def truncate_results(results, alpha, total_steps):
    """
    Truncate the results to simulate running for only alpha * total_steps.
    
    Parameters:
    -----------
    results : dict
        The original results dictionary
    alpha : float
        The ratio of steps to keep
    total_steps : int
        The total number of steps in the original data
    
    Returns:
    --------
    dict
        The truncated results
    """
    truncated = {}
    step_limit = int(alpha * total_steps)
    
    # Simple copying for non-time-series data
    if 'extrema' in results:
        # We need to recalculate extrema values for the truncated data
        truncated['extrema'] = {}
    
    # Copy and truncate time-series data
    for key in results:
        if key == 'all_steps':
            truncated[key] = [steps[:step_limit] for steps in results[key]]
        elif key in ['train', 'test']:
            truncated[key] = {}
            for metric in results[key]:
                truncated[key][metric] = {}
                for stat in results[key][metric]:
                    if stat in ['mean', 'std']:
                        # Truncate the time series
                        truncated[key][metric][stat] = results[key][metric][stat][:step_limit]
                    else:
                        # Copy other statistics
                        truncated[key][metric][stat] = results[key][metric][stat]
    
    # Calculate extrema values for the truncated data
    for dataset in ['train', 'test']:
        if dataset in truncated:
            # For loss, find minimum values
            if 'loss' in truncated[dataset]:
                mean_loss = truncated[dataset]['loss']['mean']
                min_idx = np.argmin(mean_loss)
                min_loss = mean_loss[min_idx]
                min_step = truncated['all_steps'][0][min_idx] if truncated['all_steps'] else 0
                
                truncated.setdefault('extrema', {})
                truncated['extrema'][f'min_{dataset}_loss'] = min_loss
                truncated['extrema'][f'min_{dataset}_loss_step'] = min_step
            
            # For accuracy, find maximum values
            if 'accuracy' in truncated[dataset]:
                mean_acc = truncated[dataset]['accuracy']['mean']
                max_idx = np.argmax(mean_acc)
                max_acc = mean_acc[max_idx]
                max_step = truncated['all_steps'][0][max_idx] if truncated['all_steps'] else 0
                
                truncated.setdefault('extrema', {})
                truncated['extrema'][f'max_{dataset}_accuracy'] = max_acc
                truncated['extrema'][f'max_{dataset}_accuracy_step'] = max_step
    
    return truncated

def extract_metrics_for_batch_alpha_grid(all_results):
    """
    Extract metrics (final values and times) for all models, batch sizes, and alphas.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary with all results
        
    Returns:
    --------
    dict
        Dictionary with extracted metrics
    """
    metrics_data = {}
    
    metric_keys = {
        'min_train_loss': 'Training Loss',
        'min_test_loss': 'Validation Loss',
        'max_train_accuracy': 'Training Accuracy',
        'max_test_accuracy': 'Validation Accuracy',
        'min_train_loss_step': 'Time to Best Training Loss',
        'min_test_loss_step': 'Time to Best Validation Loss',
        'max_train_accuracy_step': 'Time to Best Training Accuracy',
        'max_test_accuracy_step': 'Time to Best Validation Accuracy'
    }
    
    for model_name in all_results:
        metrics_data[model_name] = {key: {} for key in metric_keys}
        
        # Initialize arrays for each metric
        for metric in metric_keys:
            metrics_data[model_name][metric] = {
                'batch_sizes': [],
                'alpha_values': [],
                'values': []
            }
        
        # Extract metrics for each batch size and alpha
        for batch_size in sorted(all_results[model_name].keys()):
            for alpha in sorted(all_results[model_name][batch_size].keys()):
                results = all_results[model_name][batch_size][alpha]
                
                if 'extrema' not in results:
                    print(f"No extrema data for {model_name} with batch_size={batch_size}, alpha={alpha}")
                    continue
                
                # Extract and store each metric
                for metric in metric_keys:
                    if metric in results['extrema']:
                        metrics_data[model_name][metric]['batch_sizes'].append(batch_size)
                        metrics_data[model_name][metric]['alpha_values'].append(alpha)
                        metrics_data[model_name][metric]['values'].append(results['extrema'][metric])
    
    return metrics_data

def plot_metrics_vs_batch_alpha(metrics_data, figsize=(18, 15)):
    """
    Plot metrics as a function of batch size, with alpha values on a color bar.
    
    Parameters:
    -----------
    metrics_data : dict
        Dictionary with extracted metrics
    figsize : tuple, optional
        Figure size, default is (18, 15)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q6-b', exist_ok=True)
    
    # Metrics and their properties (name, position, log scale)
    metrics = [
        ('min_train_loss', 'Training Loss', 0, 0, True),
        ('min_test_loss', 'Validation Loss', 0, 1, True),
        ('max_train_accuracy', 'Training Accuracy', 1, 0, False),
        ('max_test_accuracy', 'Validation Accuracy', 1, 1, False),
        ('min_train_loss_step', 'Time to Best Training Loss', 2, 0, False),
        ('min_test_loss_step', 'Time to Best Validation Loss', 2, 1, False),
        ('max_train_accuracy_step', 'Time to Best Training Accuracy', 3, 0, False),
        ('max_test_accuracy_step', 'Time to Best Validation Accuracy', 3, 1, False)
    ]
    
    # Set up a figure with subplots for each model
    for model_name in metrics_data:
        fig, axs = plt.subplots(4, 2, figsize=figsize)
        fig.suptitle(f'{model_name.upper()} Model - Metrics vs Batch Size (B) by Alpha (α)', fontsize=16)
        
        # Get all alpha values for this model (for colormap)
        all_alpha_values = set()
        for metric, _, _, _, _ in metrics:
            all_alpha_values.update(metrics_data[model_name][metric]['alpha_values'])
        all_alpha_values = sorted(all_alpha_values)
        
        # Set up colormap for alpha values
        norm = mcolors.Normalize(vmin=min(all_alpha_values), vmax=max(all_alpha_values))
        cmap = plt.cm.viridis
        
        for metric_key, metric_label, row, col, use_log_scale in metrics:
            ax = axs[row, col]
            ax.set_title(metric_label)
            ax.set_xlabel('Batch Size (B)')
            ax.set_ylabel(metric_label)
            
            if use_log_scale:
                ax.set_yscale('log')
            
            # Set x-axis to log scale for batch size
            ax.set_xscale('log', base=2)
            
            # Group data by alpha values
            alpha_groups = {}
            batch_sizes = metrics_data[model_name][metric_key]['batch_sizes']
            alpha_values = metrics_data[model_name][metric_key]['alpha_values']
            values = metrics_data[model_name][metric_key]['values']
            
            for b, a, v in zip(batch_sizes, alpha_values, values):
                alpha_groups.setdefault(a, {'batch_sizes': [], 'values': []})
                alpha_groups[a]['batch_sizes'].append(b)
                alpha_groups[a]['values'].append(v)
            
            # Plot each alpha group
            for alpha in sorted(alpha_groups.keys()):
                group = alpha_groups[alpha]
                # Sort by batch size
                idx = np.argsort(group['batch_sizes'])
                x = [group['batch_sizes'][i] for i in idx]
                y = [group['values'][i] for i in idx]
                
                color = cmap(norm(alpha))
                ax.plot(x, y, 'o-', color=color, label=f'α={alpha}')
            
            # Set custom x-ticks for batch sizes
            ax.set_xticks(sorted(set(batch_sizes)))
            ax.set_xticklabels([str(b) for b in sorted(set(batch_sizes))])
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a legend in the first subplot to explain alpha values
        legend_elements = [plt.Line2D([0], [0], color=cmap(norm(a)), lw=2, label=f'α={a}') 
                          for a in sorted(all_alpha_values)]
        if legend_elements:
            # Add legend to upper right subplot to avoid cluttering
            axs[0, 1].legend(handles=legend_elements, loc='best', title="Alpha (α)")
        
        # Adjust layout before adding colorbar
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Add horizontal colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Alpha (α) - Ratio of Training Steps')
        
        # Set up colorbar ticks
        cbar.set_ticks(sorted(all_alpha_values))
        cbar.set_ticklabels([str(a) for a in sorted(all_alpha_values)])
        
        # Save the figure
        fig.savefig(f'results/q6-b/{model_name}_metrics_vs_batch_alpha.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Metrics vs batch size and alpha plot saved as results/q6-b/{model_name}_metrics_vs_batch_alpha.png")

def plot_combined_validation_metrics(metrics_data, figsize=(15, 12)):
    """
    Plot validation metrics for both models in one figure for comparison.
    
    Parameters:
    -----------
    metrics_data : dict
        Dictionary with extracted metrics
    figsize : tuple, optional
        Figure size, default is (15, 12)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q6-b', exist_ok=True)
    
    # Set up a figure with 2x2 subplots for validation metrics
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('LSTM vs GPT - Validation Metrics by Batch Size (B) and Alpha (α)', fontsize=16)
    
    # Metrics to plot
    validation_metrics = [
        ('min_test_loss', 'Validation Loss', 0, 0, True),
        ('max_test_accuracy', 'Validation Accuracy', 0, 1, False),
        ('min_test_loss_step', 'Time to Best Validation Loss', 1, 0, False),
        ('max_test_accuracy_step', 'Time to Best Validation Accuracy', 1, 1, False)
    ]
    
    # Colors for models
    model_colors = {'lstm': 'tab:blue', 'gpt': 'tab:red'}
    
    # Get all alpha values (for colormap)
    all_alpha_values = set()
    for model_name in metrics_data:
        for metric, _, _, _, _ in validation_metrics:
            all_alpha_values.update(metrics_data[model_name][metric]['alpha_values'])
    all_alpha_values = sorted(all_alpha_values)
    
    # Set up alpha colormap
    alpha_norm = mcolors.Normalize(vmin=min(all_alpha_values), vmax=max(all_alpha_values))
    alpha_cmap = plt.cm.viridis
    
    # Plot each validation metric
    for metric_key, metric_label, row, col, use_log_scale in validation_metrics:
        ax = axs[row, col]
        ax.set_title(metric_label)
        ax.set_xlabel('Batch Size (B)')
        ax.set_ylabel(metric_label)
        
        if use_log_scale:
            ax.set_yscale('log')
        
        # Set x-axis to log scale for batch size
        ax.set_xscale('log', base=2)
        
        # Plot data for each model and alpha value
        for model_name in metrics_data:
            # Group data by alpha values
            alpha_groups = {}
            batch_sizes = metrics_data[model_name][metric_key]['batch_sizes']
            alpha_values = metrics_data[model_name][metric_key]['alpha_values']
            values = metrics_data[model_name][metric_key]['values']
            
            for b, a, v in zip(batch_sizes, alpha_values, values):
                alpha_groups.setdefault(a, {'batch_sizes': [], 'values': []})
                alpha_groups[a]['batch_sizes'].append(b)
                alpha_groups[a]['values'].append(v)
            
            # Plot each alpha group
            for alpha in sorted(alpha_groups.keys()):
                group = alpha_groups[alpha]
                # Sort by batch size
                idx = np.argsort(group['batch_sizes'])
                x = [group['batch_sizes'][i] for i in idx]
                y = [group['values'][i] for i in idx]
                
                # Calculate color by blending model color with alpha
                color = alpha_cmap(alpha_norm(alpha))
                linestyle = '-' if model_name == 'lstm' else '--'
                marker = 'o' if model_name == 'lstm' else 's'
                
                ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, 
                        alpha=0.7, label=f'{model_name.upper()}, α={alpha}')
        
        # Set custom x-ticks for batch sizes
        all_batch_sizes = set()
        for model_name in metrics_data:
            all_batch_sizes.update(metrics_data[model_name][metric_key]['batch_sizes'])
        ax.set_xticks(sorted(all_batch_sizes))
        ax.set_xticklabels([str(b) for b in sorted(all_batch_sizes)])
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend in the upper right subplot
    handles, labels = axs[0, 1].get_legend_handles_labels()
    # Only keep one instance of each model+alpha combo
    by_label = dict(zip(labels, handles))
    axs[0, 1].legend(by_label.values(), by_label.keys(), loc='best')
    
    # Adjust layout before adding colorbar
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Add horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=alpha_norm, cmap=alpha_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Alpha (α) - Ratio of Training Steps')
    
    # Set up colorbar ticks
    cbar.set_ticks(sorted(all_alpha_values))
    cbar.set_ticklabels([str(a) for a in sorted(all_alpha_values)])
    
    # Save the figure
    fig.savefig(f'results/q6-b/combined_validation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Combined validation metrics plot saved as results/q6-b/combined_validation_metrics.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q6-b', exist_ok=True)
    
    print("Loading and processing results...")
    all_results = load_all_results()
    
    print("Extracting metrics for batch size and alpha analysis...")
    metrics_data = extract_metrics_for_batch_alpha_grid(all_results)
    
    print("Plotting metrics vs batch size and alpha for each model...")
    plot_metrics_vs_batch_alpha(metrics_data)
    
    print("Plotting combined validation metrics for comparison...")
    plot_combined_validation_metrics(metrics_data)
    
    print("\nAll plots have been saved to the results/q6-b/ directory.")
    print("\nAnalysis of generalization performance:")
    print("1. Validation Loss: Check results/q6-b/combined_validation_metrics.png to see if validation loss decreases as B and/or α increases.")
    print("2. Validation Accuracy: Check results/q6-b/combined_validation_metrics.png to see if validation accuracy improves as B and/or α increases.") 