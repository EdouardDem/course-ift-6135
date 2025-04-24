import torch
import torch.utils.data
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

from ddpm_utils.args import *
from ddpm_utils.dataset import *
from ddpm_utils.unet import *
from q2_trainer_ddpm import *
from q2_ddpm import DenoiseDiffusion

def main():
    # Print device info
    print(f"Using {args.device} backend")

    # Initialize the UNet model
    eps_model = UNet(c_in=1, c_out=1)
    eps_model = load_weights(eps_model, args.MODEL_PATH)
    print("No weights to load" if eps_model is None else "Loaded weights from checkpoint")

    # Initialize the diffusion model
    diffusion_model = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=args.n_steps,
        device=args.device,
    )

    # Initialize the trainer
    trainer = Trainer(args, eps_model, diffusion_model)

    # Create the dataloader
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

    # Generate and plot intermediate samples after training
    print("\nGenerating intermediate samples...")
    steps_to_show = [0, 100, 500, 800, 900, 950, 980, 999]
    images = trainer.generate_intermediate_samples(n_samples=4, steps_to_show=steps_to_show)

    # Plot the intermediate samples
    def plot_intermediate_samples(images, steps_to_show, n_samples):
        plt.figure(figsize=(25, 15*n_samples))
        fig, axs = plt.subplots(n_samples, len(steps_to_show))
        for sample_idx in range(n_samples):
            for step_idx, img in enumerate(images):
                axs[sample_idx, step_idx].imshow(img[sample_idx, 0], cmap='gray')
                step = steps_to_show[step_idx] if step_idx < len(steps_to_show) else args.n_steps
                axs[sample_idx, step_idx].set_title(f' Image {sample_idx} \nt={args.n_steps - step-1}', size=8)
                axs[sample_idx, step_idx].axis('off')
        plt.tight_layout()
        plt.savefig('images/q2/intermediate_samples.png')
        plt.close()

    plot_intermediate_samples(images, steps_to_show, n_samples=4)
    print("Training completed and intermediate samples saved to images/intermediate_samples.png")

if __name__ == "__main__":
    main() 