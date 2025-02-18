import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration
base_dir = "logs/question-5"
optimizers = ['adam', 'sgd', 'momentum', 'adamw']
learning_rates = ['0.001', '0.0001', '0.00001']
weight_decays = ['0.0005', '0.001', '0.005', '0.01']

# Définir une palette de couleurs distincte pour chaque lr
lr_colors = {
    '0.001': 'blue',
    '0.0001': 'red',
    '0.00001': 'green'
}

def load_results(logdir):
    """Load results from json file"""
    with open(os.path.join(logdir, 'results.json'), 'r') as f:
        return json.load(f)

def create_optimizer_plot(optimizer, metric='valid_accs'):
    """Create plot for a specific optimizer showing all lr and weight_decay combinations"""
    plt.figure(figsize=(12, 8))
    
    # Pour chaque learning rate
    for lr in learning_rates:
        base_color = lr_colors[lr]
        # Créer une palette de dégradés pour les weight_decays
        n_weights = len(weight_decays)
        color_palette = sns.light_palette(base_color, n_colors=n_weights+2)[1:-1]
        
        # Pour chaque weight decay
        for wd, color in zip(weight_decays, color_palette):
            exp_name = f"resnet18-{optimizer}_{lr}_{wd}"
            try:
                results = load_results(os.path.join(base_dir, exp_name))
                epochs = range(1, len(results[metric]) + 1)
                
                label = f"lr={lr}, wd={wd}"
                plt.plot(epochs, results[metric], color=color, label=label, linewidth=2)
            except FileNotFoundError:
                print(f"Warning: No results found for {exp_name}")
                continue

    plt.title(f'{optimizer.upper()} - {metric.replace("_", " ").title()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    
    # Ajuster la légende pour qu'elle soit lisible
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return plt.gcf()

def generate_optimizer_plots():
    """Generate plots for all optimizers"""
    metrics = ['train_losses', 'valid_losses', 'train_accs', 'valid_accs']
    
    # Créer le dossier de sauvegarde
    save_dir = "plots/question-5"
    os.makedirs(save_dir, exist_ok=True)
    
    # Pour chaque métrique
    for metric in metrics:
        # Pour chaque optimizer
        for optimizer in optimizers:
            fig = create_optimizer_plot(optimizer, metric)
            save_path = os.path.join(save_dir, f'{optimizer}_{metric}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        print(f"Generated plots for {metric}")

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    generate_optimizer_plots() 