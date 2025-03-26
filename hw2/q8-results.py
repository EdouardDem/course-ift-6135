from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from data import get_arithmetic_dataset, BOS_TOKEN
from torch.utils.data import DataLoader
import seaborn as sns
from gpt import GPT  # Import GPT model class

# Configuration
model_path = Path("logs/q8/model=gpt-optimizer=adamw-n_steps=10000-save_model_step=20000/seed=0/0/model.pth")
results_dir = Path("results/q8")
os.makedirs(results_dir, exist_ok=True)

def load_model(model_path):
    """
    Load the saved GPT model.
    
    Parameters:
    -----------
    model_path : Path
        Path to the saved model file
    
    Returns:
    --------
    model : torch.nn.Module
        The loaded GPT model
    container : dict
        The original loaded container
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load the model container
    container = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract model parameters from the state dict
    state_dict = container.get('model_state_dict', container)  # Try both keys
    
    # Create a new GPT model instance
    # We need to extract model hyperparameters from the state dict
    # Examine keys to determine model structure
    
    # Extract hyperparameters from state dict
    # Number of layers = how many blocks in decoder
    block_keys = [k for k in state_dict.keys() if 'decoder.blocks' in k]
    unique_blocks = set([k.split('.')[2] for k in block_keys if len(k.split('.')) > 2])
    num_layers = len(unique_blocks)
    
    # Number of heads - need to determine from state dict dimension
    # W_Q weight shape is (embedding_size, embedding_size)
    # and embedding_size should be divisible by num_heads
    embedding_weight = state_dict.get('embedding.tokens.weight', None)
    if embedding_weight is None:
        # If not found, try the first layer's Q projection weight
        w_q_weight_key = [k for k in state_dict.keys() if 'W_Q.weight' in k][0]
        embedding_size = state_dict[w_q_weight_key].shape[0]
    else:
        embedding_size = embedding_weight.shape[1]
    
    # Reasonable default values for GPT model
    num_heads = 4  # Common value, can be adjusted
    vocabulary_size = state_dict['classifier.weight'].shape[0]
    sequence_length = 6  # A reasonable default
    
    # Create GPT model instance
    model = GPT(
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_size=embedding_size,
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        multiplier=4,
        dropout=0.0,
        non_linearity="gelu",
        padding_index=None,
        bias_attention=True,
        bias_classifier=True,
        share_embeddings=False
    )

    model.to(torch.device('cpu'))
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model config: layers={num_layers}, embedding_size={embedding_size}, vocab_size={vocabulary_size}")
    
    return model, container

def get_samples_from_dataset(num_samples=2):
    """
    Get samples from the arithmetic dataset.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to retrieve
    
    Returns:
    --------
    tuple
        Samples, tokenizer, max_length, padding_index
    """
    # Parameters for dataset creation - match those used in training
    p = 11  # Modulo for operations
    q = p   # Modulo for results
    operator = "+"  # Operation
    r_train = 0.8   # Train-validation split
    
    # Get the dataset
    (train_dataset, _), tokenizer, max_length, padding_index = get_arithmetic_dataset(
        p=p, q=q, operator=operator, r_train=r_train,
        operation_orders=2, is_symmetric=False, shuffle=True, seed=42
    )
    
    # Create a dataloader to extract samples
    dataloader = DataLoader(train_dataset, batch_size=num_samples, shuffle=True)
    
    # Get a single batch
    inputs, targets, eq_positions, masks = next(iter(dataloader))
    
    print(f"Retrieved {num_samples} samples from the dataset")
    return inputs, targets, eq_positions, masks, tokenizer, max_length, padding_index

def run_model_with_attention(model, inputs, masks, tokenizer):
    """
    Run the model and extract attention weights.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The GPT model
    inputs : torch.Tensor
        Input tensor of shape (B, S)
    masks : torch.Tensor
        Attention mask of shape (B, S)
    tokenizer : Tokenizer
        Tokenizer for decoding tokens
    
    Returns:
    --------
    tuple
        Model outputs including attentions
    """
    with torch.no_grad():
        # Run the model and get outputs including attention weights
        logits, (hidden_states, attentions) = model(inputs)
    
    # Print the input sequences
    for i in range(inputs.shape[0]):
        tokens = inputs[i]
        valid_tokens = tokens[masks[i].bool()]
        decoded = tokenizer.decode(valid_tokens)
        print(f"Sample {i+1}: {decoded}")
    
    print(f"Model run successfully. Attention shape: {attentions.shape}")
    
    return logits, hidden_states, attentions

def visualize_attention_weights(attentions, inputs, masks, tokenizer, save_dir):
    """
    Visualize attention weights for each layer and head.
    
    Parameters:
    -----------
    attentions : torch.Tensor
        Attention weights of shape (B, num_layers, num_heads, S, S)
    inputs : torch.Tensor
        Input tensor of shape (B, S)
    masks : torch.Tensor
        Attention mask of shape (B, S)
    tokenizer : Tokenizer
        Tokenizer for decoding tokens
    save_dir : Path
        Directory to save the visualizations
    """
    # Get dimensions
    batch_size, num_layers, num_heads, seq_len, _ = attentions.shape
    
    # For each sample in the batch
    for sample_idx in range(batch_size):
        # Extract tokens from the input
        tokens = inputs[sample_idx]
        valid_mask = masks[sample_idx].bool()
        valid_tokens = tokens[valid_mask]
        valid_seq_len = valid_tokens.size(0)
        
        # Decode tokens to get their string representation
        token_strings = [tokenizer.itos[token.item()] for token in valid_tokens]
        
        # Create a grid of plots: num_layers x num_heads
        fig, axes = plt.subplots(num_layers, num_heads, 
                                figsize=(num_heads * 2.5, num_layers * 2.5),
                                squeeze=False)
        
        # Set the title for the entire figure
        fig.suptitle(f"Attention Weights - Sample {sample_idx + 1}: {tokenizer.decode(valid_tokens)}", 
                    fontsize=16)
        
        # Plot each layer and head
        for layer in range(num_layers):
            for head in range(num_heads):
                ax = axes[layer, head]
                
                # Extract attention weights for this layer, head, and sample
                # Only use the valid sequence length
                attn = attentions[sample_idx, layer, head, :valid_seq_len, :valid_seq_len].cpu().numpy()
                
                # Create a heatmap
                im = ax.imshow(attn, cmap='viridis', vmin=0, vmax=1)
                
                # Set title, labels and ticks
                ax.set_title(f"Layer {layer+1}, Head {head+1}", fontsize=10)
                
                # Only label axes on the left and bottom edges
                if head == 0:
                    ax.set_ylabel("Query")
                if layer == num_layers - 1:
                    ax.set_xlabel("Key")
                
                # Set tick positions and labels
                ax.set_xticks(np.arange(valid_seq_len))
                ax.set_yticks(np.arange(valid_seq_len))
                ax.set_xticklabels(token_strings, rotation=90, fontsize=8)
                ax.set_yticklabels(token_strings, fontsize=8)
                
                # Add grid to make it easier to see which cells correspond to which tokens
                ax.grid(visible=False)
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        cbar.set_label('Attention Weight')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
        
        # Save the figure
        fig_path = save_dir / f"sample_{sample_idx + 1}_attention.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention visualization for sample {sample_idx + 1} saved to {fig_path}")

if __name__ == "__main__":
    try:
        # Load the model
        model, container = load_model(model_path)
        
        # Get samples from the dataset
        inputs, targets, eq_positions, masks, tokenizer, max_length, padding_index = get_samples_from_dataset(num_samples=2)
        
        # Run the model and extract attention weights
        logits, hidden_states, attentions = run_model_with_attention(model, inputs, masks, tokenizer)
        
        # Visualize attention weights
        visualize_attention_weights(attentions, inputs, masks, tokenizer, results_dir)
        
        print("\nAttention weight visualization completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 