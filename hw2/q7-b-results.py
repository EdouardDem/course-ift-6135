from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from checkpointing import load_and_combine_results
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

# Configuration
models = ["lstm"]  # Only LSTM model in q7
seeds = ["0", "42"]  # Seeds to average over
weight_decay_values_string = ["0.25", "0.5", "0.75", "1"]  # weight_decay values as strings
weight_decay_values = [0.25, 0.5, 0.75, 1.0]  # weight_decay values

def load_all_results():
    """
    Load results for all model configurations.
    
    Returns:
    --------
    dict : Dictionary with all results organized by model and weight_decay
    """
    all_results = {}
    
    for model_name in models:
        all_results[model_name] = {}
        
        for weight_decay in weight_decay_values:
            # Set up the correct base directory
            weight_decay_string = weight_decay_values_string[weight_decay_values.index(weight_decay)]
            base_dir = Path(f"logs/q7/model={model_name}-optimizer=adamw-n_steps=40001-weight_decay={weight_decay_string}")
            
            if not base_dir.exists():
                print(f"Directory not found: {base_dir}")
                continue
            
            try:
                results = load_and_combine_results(base_dir, seeds)
                if results is not None:
                    all_results[model_name][float(weight_decay)] = results
                else:
                    print(f"No results found for {model_name} with weight_decay={weight_decay}")
            except Exception as e:
                print(f"Error loading data for {model_name} with weight_decay={weight_decay}: {e}")
    
    return all_results

def extract_metrics_for_weight_decay(all_results):
    """
    Extract final metrics (loss, accuracy, and their times) for all models and weight_decay values.
    
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
    
    # Define the metrics we want to extract
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
                'weight_decay_values': [],
                'values': [],
                'std_values': []
            }
        
        # Extract metrics for each weight_decay value
        for weight_decay in sorted(all_results[model_name].keys()):
            results = all_results[model_name][weight_decay]
            
            # Calculate extrema if not already present
            if 'extrema' not in results:
                results['extrema'] = {}
                
                # For each split (train/test)
                for split in ['train', 'test']:
                    if split in results:
                        # Loss - find minimum
                        if 'loss' in results[split]:
                            loss = results[split]['loss']['mean']
                            loss_std = results[split]['loss']['std']
                            min_idx = np.argmin(loss)
                            min_loss = loss[min_idx]
                            min_loss_std = loss_std[min_idx]
                            min_step = results['all_steps'][0][min_idx] if 'all_steps' in results and len(results['all_steps']) > 0 else 0
                            
                            results['extrema'][f'min_{split}_loss'] = min_loss
                            results['extrema'][f'min_{split}_loss_std'] = min_loss_std
                            results['extrema'][f'min_{split}_loss_step'] = min_step
                        
                        # Accuracy - find maximum
                        if 'accuracy' in results[split]:
                            acc = results[split]['accuracy']['mean']
                            acc_std = results[split]['accuracy']['std']
                            max_idx = np.argmax(acc)
                            max_acc = acc[max_idx]
                            max_acc_std = acc_std[max_idx]
                            max_step = results['all_steps'][0][max_idx] if 'all_steps' in results and len(results['all_steps']) > 0 else 0
                            
                            results['extrema'][f'max_{split}_accuracy'] = max_acc
                            results['extrema'][f'max_{split}_accuracy_std'] = max_acc_std
                            results['extrema'][f'max_{split}_accuracy_step'] = max_step
            
            # Extract and store each metric
            for metric in metric_keys:
                std_metric = f"{metric}_std" if not metric.endswith('_step') else None
                
                if metric in results['extrema']:
                    metrics_data[model_name][metric]['weight_decay_values'].append(weight_decay)
                    metrics_data[model_name][metric]['values'].append(results['extrema'][metric])
                    
                    # Store standard deviation if available
                    if std_metric and std_metric in results['extrema']:
                        metrics_data[model_name][metric]['std_values'].append(results['extrema'][std_metric])
                    else:
                        # No std for time metrics
                        metrics_data[model_name][metric]['std_values'].append(0)
    
    return metrics_data

def plot_metrics_vs_weight_decay(metrics_data, figsize=(12, 4)):
    """
    Plot metrics as a function of weight_decay.
    
    Parameters:
    -----------
    metrics_data : dict
        Dictionary with extracted metrics
    figsize : tuple, optional
        Figure size, default is (12, 6)
    """
    # Create results directory if it doesn't exist
    os.makedirs('results/q7-b', exist_ok=True)
    
    # Metrics to plot - organized by figure
    metric_groups = [
        # Loss plot (train and val)
        [('min_train_loss', 'Training Loss', 0, True),
         ('min_test_loss', 'Validation Loss', 1, True)],
        
        # Accuracy plot (train and val)
        [('max_train_accuracy', 'Training Accuracy', 0, False),
         ('max_test_accuracy', 'Validation Accuracy', 1, False)],
        
        # Time to min loss plot (train and val)
        [('min_train_loss_step', 'Time to Best Training Loss', 0, False),
         ('min_test_loss_step', 'Time to Best Validation Loss', 1, False)],
        
        # Time to max accuracy plot (train and val)
        [('max_train_accuracy_step', 'Time to Best Training Accuracy', 0, False),
         ('max_test_accuracy_step', 'Time to Best Validation Accuracy', 1, False)]
    ]
    
    plot_titles = [
        'Loss vs Weight Decay',
        'Accuracy vs Weight Decay',
        'Time to Best Loss vs Weight Decay',
        'Time to Best Accuracy vs Weight Decay'
    ]
    
    # For each model
    for model_name in metrics_data:
        # Create a figure for each metric group (train and val in 1x2 grid)
        for group_idx, (metric_group, plot_title) in enumerate(zip(metric_groups, plot_titles)):
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(f'{model_name.upper()} Model - {plot_title}', fontsize=16)
            
            for metric_key, metric_label, col, use_log_scale in metric_group:
                ax = axs[col]
                ax.set_title(metric_label)
                ax.set_xlabel('Weight Decay')
                ax.set_ylabel(metric_label)
                
                # Use log scale for loss
                if use_log_scale:
                    ax.set_yscale('log')
                
                # Get data for this metric
                x = metrics_data[model_name][metric_key]['weight_decay_values']
                y = metrics_data[model_name][metric_key]['values']
                y_err = metrics_data[model_name][metric_key]['std_values']
                
                # Sort by weight_decay
                idx = np.argsort(x)
                x = [x[i] for i in idx]
                y = [y[i] for i in idx]
                y_err = [y_err[i] for i in idx]
                
                # Plot with error bars
                ax.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label=metric_label)
                
                # Set x-ticks to weight_decay values
                ax.set_xticks(sorted(set(x)))
                ax.set_xticklabels([str(wd) for wd in sorted(set(x))])
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout
            fig.tight_layout()
            
            # Save the figure
            filename = plot_title.lower().replace(' ', '_')
            fig.savefig(f'results/q7-b/{model_name}_{filename}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"{model_name.upper()} {plot_title} plot saved as results/q7-b/{model_name}_{filename}.png")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist
    os.makedirs('results/q7-b', exist_ok=True)
    
    print("Loading all model results...")
    all_results = load_all_results()
    
    print("Extracting metrics for weight_decay analysis...")
    metrics_data = extract_metrics_for_weight_decay(all_results)
    
    print("Plotting metrics vs weight_decay...")
    plot_metrics_vs_weight_decay(metrics_data)
    
    print("\nAll plots have been saved to the results/q7-b/ directory.") 