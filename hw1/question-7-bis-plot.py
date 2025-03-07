import os
import json
from utils import generate_plots
import matplotlib.pyplot as plt
# Configuration
base_dir = "logs/question-7-bis"
log_dirs = [
    os.path.join(base_dir, "mlpmixer-e128"),
    os.path.join(base_dir, "mlpmixer-e256"),
    os.path.join(base_dir, "mlpmixer-e512"),
    # os.path.join(base_dir, "mlpmixer-e1024")
]

legend_names = [
    "embed_dim=128",
    "embed_dim=256",
    "embed_dim=512",
    # "embed_dim=1024"
]

def load_results(logdir):
    """Load results from json file"""
    with open(os.path.join(logdir, 'results.json'), 'r') as f:
        return json.load(f)

def generate_patch_size_plots():
    """Generate comparison plots for different patch sizes of MLPMixer"""
    # Create plots directory if it doesn't exist
    save_dir = "plots/question-7-bis"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots using utils.generate_plots
    generate_plots(log_dirs, legend_names, save_dir)
    print(f"All plots have been saved in {save_dir}")


def create_metrics_plot_epochs(embed_dim=512):
    plt.figure(figsize=(12, 8))
    
    # Charger les résultats
    exp_name = f"mlpmixer-e{embed_dim}"
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
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Ajouter le titre
        plt.title(f'MLPMixer Training Metrics (embed_dim={embed_dim}, blocks=4, dropout=0.0, patch_size=4)')
        
        # Combiner les légendes des deux axes
        lns = l1 + l2 + l3 + l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center right')
        
        # Ajuster la mise en page
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Sauvegarder le graphique
        save_dir = "plots/question-7-bis"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'mlpmixer_{embed_dim}_metrics.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"Generated metrics plot in {save_dir}")
        
    except FileNotFoundError:
        print(f"Warning: No results found for {exp_name}")

if __name__ == "__main__":
    generate_patch_size_plots() 
    create_metrics_plot_epochs(128)
    create_metrics_plot_epochs(256)
    create_metrics_plot_epochs(512)
    # create_metrics_plot_epochs(1024)