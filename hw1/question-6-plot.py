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

def create_metrics_plot_epochs(epochs_num):
    """Create a plot showing all metrics for the specified epochs experiment
    Args:
        epochs_num: int, number of epochs (50 or 100)
    """
    plt.figure(figsize=(12, 8))
    
    # Charger les résultats
    exp_name = f"mlpmixer_e512_b4_d00_{epochs_num}epochs"
    try:
        results = load_results(os.path.join(base_dir, exp_name))
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Créer deux axes y pour les losses et accuracies
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()
        
        # Tracer les losses sur l'axe y gauche
        l1 = ax1.plot(epochs, results['train_losses'], 'b-', label='Train Loss', linewidth=2)
        l2 = ax1.plot(epochs, results['valid_losses'], 'b--', label='Valid Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Tracer les accuracies sur l'axe y droit
        l3 = ax2.plot(epochs, results['train_accs'], 'r-', label='Train Acc', linewidth=2)
        l4 = ax2.plot(epochs, results['valid_accs'], 'r--', label='Valid Acc', linewidth=2)
        ax2.set_ylabel('Accuracy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Ajouter le titre
        plt.title(f'MLPMixer Training Metrics (e512, b4, d0.0, {epochs_num} epochs)')
        
        # Combiner les légendes des deux axes
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center right')
        
        # Ajuster la mise en page
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Sauvegarder le graphique
        save_dir = "plots/question-6"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'mlpmixer_{epochs_num}epochs_metrics.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"Generated {epochs_num} epochs metrics plot in {save_dir}")
        
    except FileNotFoundError:
        print(f"Warning: No results found for {exp_name}")

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    generate_embed_dim_plots()
    # Générer les graphiques pour 50 et 100 époques
    create_metrics_plot_epochs(50)
    create_metrics_plot_epochs(100) 