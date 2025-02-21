import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration
base_dir = "logs/question-7"
tokens_ratios = ['0.25', '0.5', '1']
channels_ratios = ['1', '2', '4']

# Définir une palette de couleurs distincte pour chaque token_ratio
token_colors = {
    '0.25': 'blue',
    '0.5': 'red',
    '1': 'green'
}

def load_results(logdir):
    """Load results from json file"""
    with open(os.path.join(logdir, 'results.json'), 'r') as f:
        return json.load(f)

def create_ratio_plot(metric='valid_accs'):
    """Create plot showing all mlp_ratio combinations"""
    plt.figure(figsize=(12, 8))
    
    # Pour chaque ratio de tokens
    for token_ratio in tokens_ratios:
        base_color = token_colors[token_ratio]
        # Créer une palette de dégradés pour les channels_ratios
        n_channels = len(channels_ratios)
        color_palette = sns.light_palette(base_color, n_colors=n_channels+2)[1:-1]
        
        # Pour chaque ratio de channels
        for channel_ratio, color in zip(channels_ratios, color_palette):
            exp_name = f"mlpmixer-t{token_ratio.replace('.', '')}_c{channel_ratio}"
            try:
                results = load_results(os.path.join(base_dir, exp_name))
                epochs = range(1, len(results[metric]) + 1)
                
                label = f"tokens={token_ratio}, channels={channel_ratio}"
                plt.plot(epochs, results[metric], color=color, label=label, linewidth=2)
            except FileNotFoundError:
                print(f"Warning: No results found for {exp_name}")
                continue

    plt.title(f'MLPMixer - {metric.replace("_", " ").title()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    
    # Ajuster la légende pour qu'elle soit lisible
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return plt.gcf()

def generate_ratio_plots():
    """Generate plots for all metrics"""
    metrics = ['train_losses', 'valid_losses', 'train_accs', 'valid_accs']
    
    # Créer le dossier de sauvegarde
    save_dir = "plots/question-7"
    os.makedirs(save_dir, exist_ok=True)
    
    # Pour chaque métrique
    for metric in metrics:
        fig = create_ratio_plot(metric)
        save_path = os.path.join(save_dir, f'{metric}.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"Generated plot for {metric}")

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    generate_ratio_plots() 