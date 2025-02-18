import os
from utils import generate_plots

# Configuration
base_dir = "logs/question-3"
log_dirs = [
    os.path.join(base_dir, "resnet18-0-001"),
    os.path.join(base_dir, "resnet18-0-0001"),
    os.path.join(base_dir, "resnet18-0-00001")
]

# Learning rates pour la légende (dans le même ordre que log_dirs)
legend_names = [
    "lr=0.001",
    "lr=0.0001",
    "lr=0.00001"
]

def generate_lr_plots():
    """Generate comparison plots for different learning rates of ResNet18"""
    # Create plots directory if it doesn't exist
    save_dir = "plots/question-3"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots using utils.generate_plots
    generate_plots(log_dirs, legend_names, save_dir)
    print(f"All plots have been saved in {save_dir}")

if __name__ == "__main__":
    generate_lr_plots() 