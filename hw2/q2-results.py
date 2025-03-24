import torch
import numpy as np
import os
import sys
import glob
from pathlib import Path
from checkpointing import load_and_combine_results
import pandas as pd
import tabulate

def pretty_to_string(value, precision=6):
    """
    Display as few decimal places as possible.
    """
    output = f"{value:.{precision}f}"
    # Remove trailing zeros
    output = output.rstrip("0")
    # Remove decimal point if there are no decimal places
    if "." in output:
        output = output.rstrip(".")
    return output

def format_metric(value, std=None):
    """
    Format metric as mean±std if std is provided, otherwise just mean.
    Show the appropriate number of decimal places.
    """
    if std is not None:
        return f"{pretty_to_string(value)} ±{pretty_to_string(std)}"
    return f"{pretty_to_string(value)}"

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
    
    # Define the seeds we want to include
    seeds = ["0", "42"]
    
    # Get metrics for both models, separated by seed
    lstm_metrics = load_and_combine_results(lstm_dir, seeds)
    gpt_metrics = load_and_combine_results(gpt_dir, seeds)
    
    if not lstm_metrics or not gpt_metrics:
        print("Could not load metrics for one or both models.")
        sys.exit(1)
    
    # Create table rows
    rows = []
    
    # List of metrics to include in the table
    metrics = [
        ("$\\mathcal{L}_{\\text{train}}$", "min_train_loss", "min_train_loss_std"),
        ("$\\mathcal{L}_{\\text{val}}$", "min_test_loss", "min_test_loss_std"),
        ("$\\mathcal{A}_{\\text{train}}$", "max_train_accuracy", "max_train_accuracy_std"),
        ("$\\mathcal{A}_{\\text{val}}$", "max_test_accuracy", "max_test_accuracy_std"),
        ("$t_f(\\mathcal{L}_{\\text{train}})$", "min_train_loss_step", "min_train_loss_step_std"),
        ("$t_f(\\mathcal{L}_{\\text{val}})$", "min_test_loss_step", "min_test_loss_step_std"),
        ("$t_f(\\mathcal{A}_{\\text{train}})$", "max_train_accuracy_step", "max_train_accuracy_step_std"),
        ("$t_f(\\mathcal{A}_{\\text{val}})$", "max_test_accuracy_step", "max_test_accuracy_step_std"),
        ("$\\Delta t (\\mathcal{L})$", "delta_step_loss", "delta_step_loss_std"),
        ("$\\Delta t (\\mathcal{A})$", "delta_step_accuracy", "delta_step_accuracy_std"),
    ]
    
    for label, metric, std in metrics:
        row = [label]

        print(f"Processing {label} with metric {metric} and std {std}")
        
        # Add LSTM metrics for each seed
        value = lstm_metrics["extrema"][metric]
        value_std = lstm_metrics["extrema"][std]
        row.append(format_metric(value, value_std))
        
        # Add GPT metrics for each seed
        value = gpt_metrics["extrema"][metric]
        value_std = gpt_metrics["extrema"][std]
        row.append(format_metric(value, value_std))
                
        rows.append(row)
    
    # Create pandas DataFrame with appropriate column names
    columns = ["Metric", "LSTM", "GPT"]
    df = pd.DataFrame(rows, columns=columns)
    
    # Display the table
    print("\nComparison of LSTM and GPT Models by Seed:\n")
    print(df.to_string(index=False))
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/q2")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open(results_dir / "model_comparison_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {results_dir}/model_comparison_table.tex")
    
    # Save as Markdown table
    markdown_table = df.to_markdown(index=False)
        
    with open(results_dir / "model_comparison_table.md", "w") as f:
        f.write("# Comparison of LSTM and GPT Models by Seed\n\n")
        f.write(markdown_table)
    print(f"Markdown table saved to {results_dir}/model_comparison_table.md")

if __name__ == "__main__":
    main() 