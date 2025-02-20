import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configuration
base_dir = "logs/question-6"
embed_dims = [256, 512, 1024]
num_blocks = [2, 4, 6]
drop_rates = [0.0, 0.3, 0.5]

# Définir une palette de couleurs distincte pour chaque num_blocks
block_colors = {
    2: 'blue',
    4: 'red',
    6: 'green'
}

def load_results(logdir):
    """Load results from json file"""
    with open(os.path.join(logdir, 'results.json'), 'r') as f:
        return json.load(f)

def create_embed_dim_plot(embed_dim, metric='valid_accs'):
    """Create plot for a specific embed_dim showing all num_blocks and drop_rate combinations"""
    plt.figure(figsize=(12, 8))
    
    # Pour chaque nombre de blocs
    for blocks in num_blocks:
        base_color = block_colors[blocks]
        # Créer une palette de dégradés pour les drop_rates
        n_drops = len(drop_rates)
        color_palette = sns.light_palette(base_color, n_colors=n_drops+2)[1:-1]
        
        # Pour chaque drop rate
        for drop_rate, color in zip(drop_rates, color_palette):
            exp_name = f"mlpmixer_e{embed_dim}_b{blocks}_d{str(drop_rate).replace('.', '')}"
            try:
                results = load_results(os.path.join(base_dir, exp_name))
                epochs = range(1, len(results[metric]) + 1)
                
                label = f"blocks={blocks}, drop={drop_rate}"
                plt.plot(epochs, results[metric], color=color, label=label, linewidth=2)
            except FileNotFoundError:
                print(f"Warning: No results found for {exp_name}")
                continue

    plt.title(f'Embed Dim {embed_dim} - {metric.replace("_", " ").title()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    
    # Ajuster la légende pour qu'elle soit lisible
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return plt.gcf()

def generate_embed_dim_plots():
    """Generate plots for all embed_dims"""
    metrics = ['train_losses', 'valid_losses', 'train_accs', 'valid_accs']
    
    # Créer le dossier de sauvegarde
    save_dir = "plots/question-6"
    os.makedirs(save_dir, exist_ok=True)
    
    # Pour chaque métrique
    for metric in metrics:
        # Pour chaque dimension d'embedding
        for embed_dim in embed_dims:
            fig = create_embed_dim_plot(embed_dim, metric)
            save_path = os.path.join(save_dir, f'embed{embed_dim}_{metric}.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        print(f"Generated plots for {metric}")

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    generate_embed_dim_plots() 