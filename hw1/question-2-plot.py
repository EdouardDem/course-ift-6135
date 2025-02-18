import os
from utils import generate_plots

# Configuration
base_dir = "logs/question-2"
log_dirs = [
    os.path.join(base_dir, "mlp-relu"),
    os.path.join(base_dir, "mlp-tanh"),
    os.path.join(base_dir, "mlp-sigmoid")
]
legend_names = ["ReLU", "Tanh", "Sigmoid"]

def generate_activation_plots():
    """Generate comparison plots for different activation functions"""
    # Create plots directory if it doesn't exist
    save_dir = "plots/question-2"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots using utils.generate_plots
    generate_plots(log_dirs, legend_names, save_dir)
    print(f"All plots have been saved in {save_dir}")

if __name__ == "__main__":
    generate_activation_plots() 