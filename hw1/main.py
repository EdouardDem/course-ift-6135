"""
Basic Usage:
python main.py --model <model_name> --model_config <path_to_json> --logdir <result_dir> ...
Please see config.py for other command line usage.
"""
import copy
import warnings

import torch
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from utils import seed_experiment, to_device, cross_entropy_loss, compute_accuracy
from config import get_config_parser
import json
from mlp import MLP
from resnet18 import ResNet18
from mlpmixer import MLPMixer
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import os

fix_datasplit = True
verbose_datasplit = False

def train(epoch, model, dataloader, optimizer, args):
    model.train()
    total_iters = 0
    epoch_accuracy=0
    epoch_loss=0
    gradient_flow = None
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        batch = to_device(batch, args.device)
        optimizer.zero_grad()
        imgs, labels = batch
        logits = model(imgs)
        loss = cross_entropy_loss(logits, labels)
        acc = compute_accuracy(logits, labels)

        loss.backward()
        optimizer.step()
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        total_iters += 1

        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
    
    # TODO: Should be done for every batch
    gradient_flow = model.get_gradient_flow()
        
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")
    return epoch_loss, epoch_accuracy, time.time() - start_time, gradient_flow


def evaluate(epoch, model, dataloader, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            imgs, labels = batch
            logits = model(imgs)
            loss = cross_entropy_loss(logits, labels)
            acc = compute_accuracy(logits, labels)
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1
            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )
    return epoch_loss, epoch_accuracy, time.time() - start_time

        
if __name__ == "__main__":
    parser = get_config_parser()
    args = parser.parse_args()

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory "
            "shortly. You can try setting batch_size=1 to reduce memory usage."
        )

    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                        ])
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    if fix_datasplit:
        dataset = CIFAR10(root='./data', train=True, download=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [45000, 5000])
        # Clone val_set, otherwise the reference to dataset is the same for train_set and val_set
        val_set = copy.deepcopy(val_set)
        train_set.dataset.transform = train_transform
        val_set.dataset.transform = test_transform
    else:
        train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
        val_dataset = CIFAR10(root='./data', train=True, transform=test_transform, download=True)
        train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
        _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    if verbose_datasplit:
        # Display metrics of the training set
        print(f"Training set size: {len(train_set)}")
        print(f"Validation set size: {len(val_set)}")
        # Print transform of the training set
        print(f"Training set transform: {train_set.dataset.transform}")
        print(f"Validation set transform: {val_set.dataset.transform}")
        # Print overlap between training and validation set
        print(f"Overlap between training and validation set: {len(set(train_set.indices) & set(val_set.indices))}")
    
    # Loading the test set
    test_set = CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    
    # Load model
    print(f'Build model {args.model.upper()}...')
    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')
    
    # Override mlp_ratio values with command line arguments if model is mlpmixer
    if args.model == 'mlpmixer':
        model_config['mlp_ratio_tokens'] = args.mlp_ratio_tokens
        model_config['mlp_ratio_channels'] = args.mlp_ratio_channels
        
    print(f'########## {args.model.upper()} CONFIG ################')
    for key, val in model_config.items():
        print(f'{key}:\t{val}')
    print('############################################')
    model_cls = {'mlp': MLP, 'resnet18': ResNet18, 'mlpmixer': MLPMixer}[args.model]
    model = model_cls(**model_config)
    model.to(args.device)
    
    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    
    print(
        f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
        f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
    )

    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []
    gradient_flows = []  # Nouvelle liste pour stocker les gradients
    
    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    for epoch in range(args.epochs):
        tqdm.write(f"====== Epoch {epoch} ======>")
        loss, acc, wall_time, grad_flow = train(epoch, model, train_dataloader, optimizer, args)
        train_losses.append(loss)
        train_accs.append(acc)
        train_times.append(wall_time)
        if grad_flow is not None:
            gradient_flows.append(grad_flow)

        loss, acc, wall_time = evaluate(epoch, model, valid_dataloader, args)
        valid_losses.append(loss)
        valid_accs.append(acc)
        valid_times.append(wall_time)

    test_loss, test_acc, test_time = evaluate(
        epoch, model, test_dataloader, args, mode="test"
    )
    print(f"===== Best validation Accuracy: {max(valid_accs):.3f} =====>")

    # Save log if logdir provided
    if args.logdir is not None:
        print(f'Writing training logs to {args.logdir}...')
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
            results_dict = {
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_accs": train_accs,
                "valid_accs": valid_accs,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "gradient_flows": gradient_flows
            }
            
            f.write(json.dumps(results_dict, indent=4))
    
        # Visualize
        if args.visualize and args.model in ['resnet18', 'mlpmixer']:
            model.visualize(args.logdir)
