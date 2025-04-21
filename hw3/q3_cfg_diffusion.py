# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        
        # Ensure t is in the right shape
        t = t.view(-1, 1, 1, 1)
        
        # Convert lambda_min and lambda_max to tensors if needed
        lambda_min = torch.tensor(self.lambda_min, device=t.device, dtype=t.dtype)
        lambda_max = torch.tensor(self.lambda_max, device=t.device, dtype=t.dtype)
        
        # Calculate parameters a and b
        b = torch.arctan(torch.exp(-lambda_max/2))
        a = torch.arctan(torch.exp(-lambda_min/2)) - b
        
        # Convert t to u in [0,1] based on timestep
        u = t.float() / self.n_steps
        
        # Calculate lambda_t using the formula from the paper
        lambda_t = -2 * torch.log(torch.tan(a * u + b))

        return lambda_t
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        
        # Calculate alpha(lambda_t) using the formula from the paper: \alpha_\lambda^2 = 1/(1+e^{-\lambda})
        alpha_lambda_2 = 1 / (1 + torch.exp(-lambda_t))
        
        return alpha_lambda_2.sqrt()
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
    
        # \sigma_\lambda^2 = 1-\alpha_\lambda^2 
        alpha_lambda_2 = self.alpha_lambda(lambda_t)**2
        sigma_lambda_2 = 1 - alpha_lambda_2

        return sigma_lambda_2.sqrt()
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)

        alpha_lambda_t = self.alpha_lambda(lambda_t)
        sigma_lambda_t = self.sigma_lambda(lambda_t)
        z_lambda_t = alpha_lambda_t * x + sigma_lambda_t * noise

        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        sigma_lambda_t = self.sigma_lambda(lambda_t)
        var_q = (sigma_lambda_t ** 2) * (1 - torch.exp(lambda_t - lambda_t_prim))
    
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        sigma_lambda_t_prim = self.sigma_lambda(lambda_t_prim)
        var_q_x = (sigma_lambda_t_prim ** 2) * (1 - torch.exp(lambda_t - lambda_t_prim))
    
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        alpha_lambda_t = self.alpha_lambda(lambda_t)
        alpha_lambda_t_prim = self.alpha_lambda(lambda_t_prim)

        alpha_ratio = alpha_lambda_t_prim / alpha_lambda_t
        exp_diff = torch.exp(lambda_t - lambda_t_prim)

        mu = exp_diff * alpha_ratio * z_lambda_t + (1 - exp_diff) * alpha_lambda_t_prim * x
    
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        sigma_q_2 = self.sigma_q(lambda_t, lambda_t_prim) ** 2
        sigma_q_x_2 = self.sigma_q_x(lambda_t, lambda_t_prim) ** 2

        var = (sigma_q_x_2 ** (1 - v)) * (sigma_q_2 ** v)

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)

        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)

        sample = mu + torch.randn_like(mu) * var.sqrt()
    
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        
        #TODO: q_sample z
        lambda_t = self.get_lambda(t)
        z_lambda_t = self.q_sample(x0, lambda_t, noise)

        #TODO: compute loss
        lambda_t_prim = self.get_lambda(t - 1)
        x_t = self.p_sample(z_lambda_t, lambda_t, lambda_t_prim, x0)

        loss = F.mse_loss(x_t, x0)

    
        return loss



    