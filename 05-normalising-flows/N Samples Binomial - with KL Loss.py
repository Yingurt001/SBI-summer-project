"""
Extends the single‐sample Binomial flow to N observations.
  - Observations y₁…y_N ∼ Binomial(n,θ)
  - Flow parameters (log s,b) predicted by Linear(N→2) net on the vector y
  - Computes Monte Carlo ELBO and minimizes −ELBO
  - Fits q(θ) to approximate Beta(α₀+∑y,β₀+N·n−∑y)
  - Plots approximate vs. analytic Beta posterior
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, gaussian_kde
import numpy as np

# 1) Observed data and Binomial model parameters
y_obs     = torch.tensor([5.0])   # observed number of successes
n_trials  = 10                    # total trials
alpha0, beta0 = 2.0, 5.0          # Beta(α0, β0) prior hyperparameters

# 2) Analytic Beta posterior for comparison: α_post = α0 + y, β_post = β0 + n−y
alpha_post = alpha0 + y_obs
beta_post  = beta0  + (n_trials - y_obs)

# grid for plotting θ∈(0,1)
x = torch.linspace(0.01, 0.99, 500)
true_pdf = beta_dist.pdf(x.numpy(), alpha_post.item(), beta_post.item())

# 3) Inference network: y → (log_scale, shift) for affine flow on logit(θ)
class AffineFlowLogit(nn.Module):
    def __init__(self):
        super().__init__()
        # map 1D y to 2 params: log_scale and shift for logit space
        self.linear = nn.Linear(1, 2)
    def forward(self, y):
        params    = self.linear(y.unsqueeze(1))  # shape (1,2)
        log_scale = params[0, 0]
        shift     = params[0, 1]
        scale     = torch.exp(log_scale)         # enforce scale>0
        return scale, shift

# model + optimizer
net = AffineFlowLogit()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# 4) Training: maximize ELBO via Monte Carlo on z ~ N(0,1)
n_epochs, n_samples = 2000, 2000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # sample base noise
    z = torch.randn(n_samples)
    # get flow parameters conditioned on y_obs
    scale, shift = net(y_obs)
    # affine in logit space, then sigmoid to get θ ∈ (0,1)
    logit_theta = scale * z + shift
    theta       = theta = torch.sigmoid(logit_theta).clamp(1e-6, 1 - 1e-6)

    
    # change-of-variable: log q(θ) = log N(z;0,1) - log(scale) - log(sigmoid'(...))
    log_q_z      = torch.distributions.Normal(0,1).log_prob(z)
    log_jacobian = torch.log(theta * (1 - theta))  # derivative of sigmoid
    log_q_theta  = log_q_z - torch.log(scale) - log_jacobian
    
    # log prior: Beta(α0,β0)
    log_prior = torch.distributions.Beta(alpha0, beta0).log_prob(theta)
    # log likelihood: Binomial(n_trials,θ)
    log_lik   = torch.distributions.Binomial(n_trials, theta).log_prob(y_obs)
    
    # ELBO and negative ELBO
    elbo = (log_prior + log_lik - log_q_theta).mean()
    loss = -elbo
    loss.backward()
    optimizer.step()

# 5) Extract learned parameters and form q(θ|y)
with torch.no_grad():
    learned_scale, learned_shift = net(y_obs)

# evaluate approximate q pdf via change-of-vars on grid x
logit_x    = torch.log(x/(1-x))
z_x        = (logit_x - learned_shift) / learned_scale
log_qx_z   = torch.distributions.Normal(0,1).log_prob(z_x)
log_qx     = log_qx_z - torch.log(learned_scale) - torch.log(x * (1-x))
q_pdf      = torch.exp(log_qx)

# 6) Plot learned q vs true Beta posterior
plt.figure(figsize=(8,4))
plt.plot(x.numpy(), q_pdf.numpy(), '-',  label='Learned q(θ|y)')
plt.plot(x.numpy(), true_pdf,           '--', label='True Beta posterior')
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Affine Flow Approximation of Binomial Posterior')
plt.legend()
plt.tight_layout()
plt.show()
