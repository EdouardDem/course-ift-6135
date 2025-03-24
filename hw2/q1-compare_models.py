import torch
import numpy as np
import os
import sys
import glob
from pathlib import Path
from checkpointing import get_all_checkpoints, get_extrema_performance_steps
from checkpointing import get_all_checkpoints_per_trials, get_extrema_performance_steps_per_trials
import pandas as pd
try:
    from tabulate import tabulate  # Required for pandas to_markdown
except ImportError:
    pass

def find_seed_dirs(base_dir):
    """Find all seed directories in a given base directory."""
    seed_dirs = []
    # Look for seed=X directories
    for seed_dir in Path(base_dir).glob("seed=*"):
        if seed_dir.is_dir():
            seed_dirs.append(seed_dir)
            
    return seed_dirs

def get_model_metrics_multi_seed(exp_dir, metric_file='test.pth'):
    """Load experiment results from multiple seeds and compute mean/std metrics."""
    # Find all seed directories
    seed_dirs = find_seed_dirs(exp_dir)
    
    if not seed_dirs:
        print(f"No seed directories found in {exp_dir}")
        return None
    
    print(f"Found {len(seed_dirs)} seed directories in {exp_dir}")
    
    # Gather all metrics from seeds
    all_metrics = {
        "min_train_loss": [],
        "min_test_loss": [],
        "max_train_accuracy": [],
        "max_test_accuracy": [],
        "min_train_loss_step": [],
        "min_test_loss_step": [],
        "max_train_accuracy_step": [],
        "max_test_accuracy_step": []
    }
    
    for seed_dir in seed_dirs:
        # Find the metric file (look in subdirectories if needed)
        metric_files = list(Path(seed_dir).glob(f"**/{metric_file}"))
        if not metric_files:
            print(f"No {metric_file} found in {seed_dir}")
            continue
            
        # Use the first metric file found
        stats_path = str(metric_files[0])
        try:
            statistics = torch.load(stats_path)
            metrics = get_extrema_performance_steps(statistics)
            
            # Add metrics to our aggregated dict
            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])
                    
        except Exception as e:
            print(f"Error processing {stats_path}: {e}")
    
    if not all(all_metrics.values()):
        print(f"No valid metrics found in any seed directory")
        return None
        
    # Calculate mean and std for each metric
    result = {}
    for key, values in all_metrics.items():
        if values:
            result[key] = np.mean(values)
            result[f"{key}_std"] = np.std(values)
    
    return result

def format_metric(value, std=None):
    """Format metric as mean±std if std is provided, otherwise just mean."""
    if std is not None:
        return f"{value:.4f}±{std:.4f}"
    return f"{value:.4f}"

def get_metric_with_std(metrics, metric_name, std_name=None):
    """Get a metric with its std if available."""
    value = metrics.get(metric_name)
    if value is None:
        return None
    
    if std_name and std_name in metrics:
        return format_metric(value, metrics[std_name])
    return format_metric(value)

def dataframe_to_markdown(df):
    """Convert a pandas DataFrame to a markdown table manually."""
    headers = df.columns
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row) + " |")
    
    return "\n".join([header_row, separator_row] + rows)

def main():
    # Path to experiment directories
    base_dir = Path("logs/q1")
    lstm_dir = base_dir / "model=lstm-optimizer=adamw-n_steps=10000"
    gpt_dir = base_dir / "model=gpt-optimizer=adamw-n_steps=10000"
    
    # Get metrics for both models
    lstm_metrics = get_model_metrics_multi_seed(lstm_dir, 'test.pth')
    gpt_metrics = get_model_metrics_multi_seed(gpt_dir, 'test.pth')
    
    if lstm_metrics is None or gpt_metrics is None:
        print("Could not load metrics for one or both models.")
        sys.exit(1)
    
    # Create table rows
    rows = []
    
    # Add loss and accuracy metrics
    metrics = [
        ("$\\mathcal{L}_{\\text{train}}$", "min_train_loss", "min_train_loss_std"),
        ("$\\mathcal{L}_{\\text{val}}$", "min_test_loss", "min_test_loss_std"),
        ("$\\mathcal{A}_{\\text{train}}$", "max_train_accuracy", "max_train_accuracy_std"),
        ("$\\mathcal{A}_{\\text{val}}$", "max_test_accuracy", "max_test_accuracy_std"),
        ("$t_f(\\mathcal{L}_{\\text{train}})$", "min_train_loss_step", "min_train_loss_step_std"),
        ("$t_f(\\mathcal{L}_{\\text{val}})$", "min_test_loss_step", "min_test_loss_step_std"),
        ("$t_f(\\mathcal{A}_{\\text{train}})$", "max_train_accuracy_step", "max_train_accuracy_step_std"),
        ("$t_f(\\mathcal{A}_{\\text{val}})$", "max_test_accuracy_step", "max_test_accuracy_step_std"),
    ]
    
    for label, metric, std_metric in metrics:
        lstm_value = get_metric_with_std(lstm_metrics, metric, std_metric)
        gpt_value = get_metric_with_std(gpt_metrics, metric, std_metric)
        rows.append([label, lstm_value, gpt_value])
    
    # Add delta time metrics
    # For multi-seed data, calculate mean and std of deltas
    lstm_delta_loss_mean = abs(lstm_metrics["min_train_loss_step"] - lstm_metrics["min_test_loss_step"])
    lstm_delta_loss_std = np.sqrt(lstm_metrics["min_train_loss_step_std"]**2 + lstm_metrics["min_test_loss_step_std"]**2)
    
    lstm_delta_acc_mean = abs(lstm_metrics["max_train_accuracy_step"] - lstm_metrics["max_test_accuracy_step"])
    lstm_delta_acc_std = np.sqrt(lstm_metrics["max_train_accuracy_step_std"]**2 + lstm_metrics["max_test_accuracy_step_std"]**2)
    
    gpt_delta_loss_mean = abs(gpt_metrics["min_train_loss_step"] - gpt_metrics["min_test_loss_step"])
    gpt_delta_loss_std = np.sqrt(gpt_metrics["min_train_loss_step_std"]**2 + gpt_metrics["min_test_loss_step_std"]**2)
    
    gpt_delta_acc_mean = abs(gpt_metrics["max_train_accuracy_step"] - gpt_metrics["max_test_accuracy_step"])
    gpt_delta_acc_std = np.sqrt(gpt_metrics["max_train_accuracy_step_std"]**2 + gpt_metrics["max_test_accuracy_step_std"]**2)
    
    rows.append(["$\\Delta t (\\mathcal{L})$", format_metric(lstm_delta_loss_mean, lstm_delta_loss_std), 
                format_metric(gpt_delta_loss_mean, gpt_delta_loss_std)])
    rows.append(["$\\Delta t (\\mathcal{A})$", format_metric(lstm_delta_acc_mean, lstm_delta_acc_std), 
                format_metric(gpt_delta_acc_mean, gpt_delta_acc_std)])
    
    # Create pandas DataFrame for nice formatting
    df = pd.DataFrame(rows, columns=["Metric", "LSTM", "GPT"])
    
    # Display the table
    print("\nComparison of LSTM and GPT Models:\n")
    print(df.to_string(index=False))
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/q1")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open(results_dir / "model_comparison_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {results_dir}/model_comparison_table.tex")
    
    # Save as Markdown table
    try:
        # Try using pandas' to_markdown method first (requires tabulate)
        markdown_table = df.to_markdown(index=False)
    except (AttributeError, ImportError):
        # Fall back to manual markdown table creation
        markdown_table = dataframe_to_markdown(df)
        
    with open(results_dir / "model_comparison_table.md", "w") as f:
        f.write("# Comparison of LSTM and GPT Models\n\n")
        f.write(markdown_table)
    print(f"Markdown table saved to {results_dir}/model_comparison_table.md")

if __name__ == "__main__":
    main() 