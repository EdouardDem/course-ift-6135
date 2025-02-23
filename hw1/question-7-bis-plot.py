import os
from utils import generate_plots

# Configuration
base_dir = "logs/question-7-bis"
log_dirs = [
    os.path.join(base_dir, "mlpmixer-e128"),
    os.path.join(base_dir, "mlpmixer-e256"),
    os.path.join(base_dir, "mlpmixer-e512"),
    os.path.join(base_dir, "mlpmixer-e1024")
]

legend_names = [
    "embed_dim=128",
    "embed_dim=256",
    "embed_dim=512",
    "embed_dim=1024"
]

def generate_patch_size_plots():
    """Generate comparison plots for different patch sizes of MLPMixer"""
    # Create plots directory if it doesn't exist
    save_dir = "plots/question-7-bis"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots using utils.generate_plots
    generate_plots(log_dirs, legend_names, save_dir)
    print(f"All plots have been saved in {save_dir}")

if __name__ == "__main__":
    generate_patch_size_plots() 