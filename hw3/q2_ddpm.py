import torch 
from torch import nn 
from typing import Optional, Tuple


class DenoiseDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    ### UTILS
    def gather(self, c: torch.Tensor, t: torch.Tensor):
        c_ = c.gather(-1, t)
        return c_.reshape(-1, 1, 1, 1)

    ### FORWARD SAMPLING
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: return mean and variance of q(x_t|x_0)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        mean = alpha_bar_t.sqrt() * x0
        var = 1 - alpha_bar_t
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)
        # TODO: return x_t sampled from q(•|x_0) according to (1)
        mean, var = self.q_xt_x0(x0, t)
        sample = mean + (var.sqrt() * eps)
        return sample

    ### REVERSE SAMPLING
    def p_xt_prev_xt(self, xt: torch.Tensor, t: torch.Tensor):
        # TODO: return mean and variance of p_theta(x_{t-1} | x_t) according to (2)
        eps_theta = self.eps_model(xt, t)
        alpha_t = self.gather(self.alpha, t)
        beta_t = self.gather(self.beta, t)
        alpha_bar_t = self.gather(self.alpha_bar, t)
        
        # Compute mean according to equation (2)
        mu_theta = (1 / alpha_t.sqrt()) * (xt - beta_t / (1 - alpha_bar_t).sqrt() * eps_theta)
        
        # Compute variance
        var = beta_t
        
        return mu_theta, var

    # TODO: sample x_{t-1} from p_theta(•|x_t) according to (3)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        
        # Get mean and variance
        mu_theta, var = self.p_xt_prev_xt(xt, t)
        
        # Sample from N(mu_theta, var)
        eps = torch.randn_like(xt)
        sample = mu_theta + (var.sqrt() * eps)
        
        return sample

    ### LOSS
    # TODO: compute loss according to (4)
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        # TODO
        
        # Predict noise
        alpha_bar_t = self.gather(self.alpha_bar, t)
        term1 = alpha_bar_t.sqrt() * x0
        term2 = (1 - alpha_bar_t).sqrt() * noise
        eps_theta = self.eps_model(term1 + term2, t)
        
        # Compute MSE loss
        loss = torch.mean((noise - eps_theta) ** 2)
        
        return loss
