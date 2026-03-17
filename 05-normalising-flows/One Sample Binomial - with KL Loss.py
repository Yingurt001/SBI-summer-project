"""
Approximates a Beta posterior for a single Binomial observation using an affine sigmoid flow.
  - Observed y ∼ Binomial(n,θ), prior θ ∼ Beta(α₀,β₀)
  - Flow: θ = sigmoid(s·z + b), z ∼ N(0,1), with (log s,b) from a tiny MLP on y
  - Builds Monte Carlo ELBO: E_q[log p(θ,y) − log q(θ)]
  - Optimizes flow parameters via Adam
  - Compares learned q(θ) to analytic Beta(α₀+y,β₀+n−y) posterior
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
import numpy as np

# 1) Observed data: N independent Binomial observations
y_obs    = torch.tensor([7., 4., 5., 8.,9.,10,1,2,7,8,9,])  # replace with your N data points
n_trials = 10                              # number of trials per observation
N        = y_obs.shape[0]

# 2) Beta(α0, β0) prior hyperparameters
alpha0, beta0 = 2.0, 5.0

# 3) Analytic posterior for comparison: α_post = α0 + sum(y_i), β_post = β0 + N*n_trials - sum(y_i)
sum_y      = y_obs.sum()
alpha_post = alpha0 + sum_y
beta_post  = beta0  + N * n_trials - sum_y

# Create grid on θ∈(0,1) for plotting
x = np.linspace(0.01, 0.99, 500)
true_pdf = beta_dist.pdf(x, alpha_post.item(), beta_post.item())

# 4) Inference network: maps y_obs (N-vector) → (log_scale, shift)
class AffineFlowLogit(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        # Linear layer input_dim = N observations, output_dim = 2 params
        self.linear = nn.Linear(obs_dim, 2)
    def forward(self, y):
        # y: shape (N,), unsqueeze to (1, N) for the linear layer
        params    = self.linear(y.unsqueeze(0))  # shape (1,2)
        log_scale = params[0, 0]                # unconstrained scale
        shift     = params[0, 1]                # location shift
        scale     = torch.exp(log_scale)        # ensure scale > 0
        return scale, shift

# Instantiate model and optimizer
net       = AffineFlowLogit(obs_dim=N)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Small epsilon for numerical stability in log and Beta support
eps = 1e-6

# 5) Training loop: maximize the ELBO via Monte Carlo
n_epochs  = 2000
n_samples = 2000  # MC samples per iteration

for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Sample base noise z ~ N(0,1)
    z     = torch.randn(n_samples)
    scale, shift = net(y_obs)     # get flow params conditioned on all observations
    
    # Affine transform in logit space, then sigmoid to map to (0,1)
    logit_theta = scale * z + shift
    theta       = torch.sigmoid(logit_theta).clamp(eps, 1-eps)
    
    # Compute log q(θ|y_obs) using change-of-variables:
    #   log q = log N(z;0,1) - log(scale) - log(sigmoid'(·))
    log_qz      = torch.distributions.Normal(0,1).log_prob(z)
    log_jacobian= torch.log(theta * (1 - theta))                # derivative of sigmoid
    log_q       = log_qz - torch.log(scale) - log_jacobian
    
    # Compute log-prior (Beta) and log-likelihood (sum over N Binomials)
    log_prior = torch.distributions.Beta(alpha0, beta0).log_prob(theta)
    # replicate theta and y_obs to shape (n_samples, N)
    theta_rep = theta.unsqueeze(1).expand(-1, N)
    y_rep     = y_obs.unsqueeze(0).expand_as(theta_rep)
    log_lik   = torch.distributions.Binomial(n_trials, theta_rep).log_prob(y_rep).sum(dim=1)
    
    # ELBO and loss
    elbo = (log_prior + log_lik - log_q).mean()
    loss = -elbo
    
    loss.backward()
    optimizer.step()

# 6) Extract learned parameters and define approximate q(θ|y_obs)
with torch.no_grad():
    learned_scale, learned_shift = net(y_obs)

# Build q_pdf on the grid x
x_t       = torch.tensor(x)
logit_x   = torch.log(x_t / (1 - x_t))            # logit inverse of sigmoid
z_x       = (logit_x - learned_shift) / learned_scale
log_qxz   = torch.distributions.Normal(0,1).log_prob(z_x)
log_qx    = log_qxz - torch.log(learned_scale) - torch.log(x_t * (1 - x_t))
q_pdf     = log_qx.exp().numpy()

# 7) Plot learned q vs true Beta posterior
plt.figure(figsize=(8, 4))
plt.plot(x, q_pdf,    '-',  label='Learned q(θ|y)')
plt.plot(x, true_pdf, '--', label='True Beta posterior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Affine Flow Approximation with Multiple Observations')
plt.legend()
plt.tight_layout()
plt.show()
