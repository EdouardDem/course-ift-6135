import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os

from cfg_utils.args import *
from cfg_utils.dataset import *
from cfg_utils.unet import *
from q3_trainer_cfg import Trainer, show_save
from q3_cfg_diffusion import CFGDiffusion

def main():
    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')
        
    # Print device info
    print(f"Using {args.device} backend")

    # Initialize the UNet model with conditioning
    eps_model = UNet_conditional(c_in=1, c_out=1, num_classes=10)
    # eps_model = load_weights(eps_model, args.MODEL_PATH)
    print("No weights to load" if eps_model is None else "Loaded weights from checkpoint")

    # Initialize the diffusion model with classifier-free guidance
    diffusion_model = CFGDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device,
    )

    # Initialize the trainer
    trainer = Trainer(args, eps_model, diffusion_model)

    # Create the dataloader with MNIST dataset
    dataloader = torch.utils.data.DataLoader(
        MNISTDataset(),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # Start training
    print("Starting training...")
    trainer.train(dataloader)

    # Generate samples with different guidance scales
    print("\nGenerating samples with different guidance scales...")
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    for cfg_scale in guidance_scales:
        print(f"Generating samples with guidance scale {cfg_scale}...")
        # Generate fixed digits (0-9)
        fixed_labels = torch.arange(0, 10, device=args.device)
        if fixed_labels.shape[0] < args.n_samples:
            # Repeat labels if needed
            fixed_labels = fixed_labels.repeat(args.n_samples // 10 + 1)[:args.n_samples]
        
        samples = trainer.sample(labels=fixed_labels, cfg_scale=cfg_scale, set_seed=True)
        show_save(
            samples, 
            fixed_labels,
            show=False, 
            save=True, 
            file_name=f"images/q3/CFG_scale_{cfg_scale}.png"
        )
    
    print("Training completed and samples saved to images/ directory")

if __name__ == "__main__":
    main() 