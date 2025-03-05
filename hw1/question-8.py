import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Usage: python question-8.py logs/question-4/mlpmixer-2/results.json "MLPMixer Patch Size 2"

def load_results(json_path):
    """Load results from json file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_gradient_flow(results, exp_name, save_dir):
    """Create gradient flow visualization
    Args:
        results: Dictionary containing the results
        exp_name: Name of the experiment for the title
        save_dir: Directory where to save the plot
    """
    gradient_flows = results['gradient_flows']
    epochs = range(1, len(gradient_flows) + 1)
    
    # Créer la figure
    plt.figure(figsize=(12, 6))
    
    # Tracer les gradients des couches cachées
    hidden_layers_count = len(gradient_flows[0]['hidden_layers'])
    for layer_idx in range(hidden_layers_count):
        layer_grads = [epoch['hidden_layers'][layer_idx] for epoch in gradient_flows]
        plt.semilogy(epochs, layer_grads, label=f'Layer {layer_idx+1}', alpha=0.8)
    
    # Tracer le gradient de la couche de sortie
    output_grads = [epoch['output_layer'] for epoch in gradient_flows]
    plt.semilogy(epochs, output_grads, label='Output Layer', linewidth=2, color='black')
    
    # Personnaliser le graphique
    plt.title(f'Gradient Flow - {exp_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Sauvegarder le graphique
    save_path = os.path.join(save_dir, 'gradient_flow.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Gradient flow plot saved as {save_path}")

def main():
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Plot gradient flow from results JSON')
    parser.add_argument('json_path', type=str, help='Path to the results JSON file')
    parser.add_argument('exp_name', type=str, help='Name of the experiment')
    args = parser.parse_args()
    
    # Charger les résultats
    results = load_results(args.json_path)
    
    # Obtenir le dossier du fichier JSON
    save_dir = os.path.dirname(args.json_path)
    
    # Créer le graphique
    plot_gradient_flow(results, args.exp_name, save_dir)

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    main() 