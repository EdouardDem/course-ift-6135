import os
from utils import generate_plots

# Configuration
base_dir = "logs/question-4"
log_dirs = [
    os.path.join(base_dir, "mlpmixer-4"),
    os.path.join(base_dir, "mlpmixer-8"),
    os.path.join(base_dir, "mlpmixer-16")
]

legend_names = [
    "patch_size=4",
    "patch_size=8",
    "patch_size=16"
]

def generate_patch_size_plots():
    """Generate comparison plots for different patch sizes of MLPMixer"""
    # Create plots directory if it doesn't exist
    save_dir = "plots/question-4"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all plots using utils.generate_plots
    generate_plots(log_dirs, legend_names, save_dir)
    print(f"All plots have been saved in {save_dir}")

if __name__ == "__main__":
    generate_patch_size_plots() 