"""
Solutions for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch

torch.manual_seed(42)

def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    #TODO: compute log_likelihood_bernoulli
    # Clamp values to avoid issues
    eps = 1e-9
    mu = torch.clamp(mu, eps, 1 - eps)
    # From https://web.stanford.edu/class/archive/cs/cs109/cs109.1206/lectureNotes/LN20_parameters_mle.pdf
    log_likelihood = target * torch.log(mu) + (1 - target) * torch.log(1 - mu)
    ll_bernoulli = log_likelihood.sum(dim=1)
    
    return ll_bernoulli


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    #TODO: compute log normal
    normal_dist = torch.distributions.Normal(mu, torch.exp(logvar / 2))
    ll_normal = normal_dist.log_prob(z)
    ll_normal = ll_normal.sum(dim=1)
    
    return ll_normal


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    #TODO: compute log_mean_exp
    a = torch.max(y, dim=1)[0]
    sum = torch.sum(torch.exp(y - a.unsqueeze(1)), dim=1)
    lme = torch.log(sum / sample_size) + a

    return lme 


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)
    
    # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # log variances to variances
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    term1 = logvar_p - logvar_q  # log(var_p/var_q)
    term2 = (var_q + (mu_q - mu_p).pow(2)) / var_p  # (var_q + (mu_q - mu_p)^2)/var_p
    term3 = -1  # constant term

    kl_per_dimension = 0.5 * (term1 + term2 + term3)
    kl_gg = kl_per_dimension.sum(dim=1)
    
    return kl_gg


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    
    # Create the standard deviation for sampling
    std_q = torch.exp(0.5 * logvar_q)
    epsilon = torch.randn_like(std_q)
    z = mu_q + std_q * epsilon
    
    # Compute log q(z)
    log_q_z = -0.5 * (logvar_q + ((z - mu_q) / std_q).pow(2) + math.log(2 * math.pi))
    log_q_z = log_q_z.sum(dim=2)
    
    # Compute log p(z)
    std_p = torch.exp(0.5 * logvar_p)
    log_p_z = -0.5 * (logvar_p + ((z - mu_p) / std_p).pow(2) + math.log(2 * math.pi))
    log_p_z = log_p_z.sum(dim=2)

    kl_per_sample = log_q_z - log_p_z
    kl_mc = kl_per_sample.mean(dim=1)
    
    return kl_mc

