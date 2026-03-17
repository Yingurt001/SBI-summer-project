"""
Approximates a Normal posterior in a conjugate Normal–Normal model using an affine flow.
  - Observations y ∼ N(θ,σₗᵢₖ²), prior θ ∼ N(0,σₚᵣᵢₒᵣ²)
  - Flow: θ = s·z + b, z ∼ N(0,1), with (s,b) from a Linear(1→2) net on y
  - Builds Monte Carlo estimate of KL[q‖p]
  - Learns flow params by minimizing KL
  - Compares learned Normal approximation to the closed-form Normal posterior
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Observed data and known noise levels (all as torch.Tensors)
y_obs      = torch.tensor([1.2])      # Single observed y
sigma_lik  = torch.tensor(0.5)        # Known likelihood SD
sigma_prior= torch.tensor(1.0)        # Known prior SD

# 2) Analytic true posterior parameters
post_var   = 1 / (1/sigma_prior**2 + 1/sigma_lik**2)  # now a torch.Tensor
post_mean  = post_var * (y_obs / sigma_lik**2)
post_sd    = post_var.sqrt()                          # works because post_var is a tenso


# 3) INFERENCE NETWORK: y -> (log_scale, shift)
class AffineFlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        # simple linear map from 1D y to 2 params
        self.linear = nn.Linear(1, 2)
    def forward(self, y):
        # y has shape (1,), so unsqueeze to (1,1)
        params = self.linear(y.unsqueeze(1)) 
        log_scale = params[0,0]
        shift     = params[0,1]
        scale = torch.exp(log_scale)  # enforce positivity
        return scale, shift

# instantiate model + optimizer
net = AffineFlowNet()
opt = optim.Adam(net.parameters(), lr=1e-3)

# 4) TRAINING LOOP: maximize ELBO via Monte Carlo
n_epochs  = 2000
n_samples = 1000

for epoch in range(n_epochs):
    opt.zero_grad()
    
    # 4a) sample base noise
    z = torch.randn(n_samples)
    
    # 4b) get flow params conditioned on y_obs
    scale, shift = net(y_obs)
    
    # 4c) push z through flow to get theta samples
    theta = scale * z + shift
    
    # 4d) compute log q(theta|y) via change-of-variable
    log_q = torch.distributions.Normal(0,1).log_prob(z) - torch.log(scale)
    
    # 4e) compute joint log-prob log p(theta, y_obs)
    log_lik   = torch.distributions.Normal(theta, sigma_lik).log_prob(y_obs)
    log_prior = torch.distributions.Normal(0, sigma_prior).log_prob(theta)
    log_p     = log_lik + log_prior
    
    # 4f) ELBO and loss
    elbo = (log_p - log_q).mean()
    loss = -elbo              # we minimize negative ELBO
    
    # 4g) backprop & step
    loss.backward()
    opt.step()

# 5) EXTRACT LEARNED PARAMETERS
with torch.no_grad():
    learned_scale, learned_shift = net(y_obs)

# 6) DEFINE APPROXIMATE & TRUE POSTERIOR DISTRIBUTIONS
q_dist    = torch.distributions.Normal(learned_shift, learned_scale)
true_dist = torch.distributions.Normal(post_mean,    post_sd)

# 7) EVALUATE THEIR PDFs ON A GRID
x = torch.linspace(post_mean.item() - 3*post_sd.item(),
                   post_mean.item() + 3*post_sd.item(), 500)
q_pdf   = q_dist.log_prob(x).exp()
true_pdf= true_dist.log_prob(x).exp()

# 8) PLOT BOTH CURVES
plt.figure()
plt.plot(x.numpy(),   q_pdf.numpy(),    linestyle='-')   # learned q
plt.plot(x.numpy(),   true_pdf.numpy(), linestyle='--')  # true posterior
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Learned q(θ|y) vs True Posterior')
plt.legend(['q(θ|y)', 'True Posterior'])
plt.show()

# === Posterior Sampling & PDF Estimation ===

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# 1) Draw many samples from the learned q(θ|y)
n_posterior_samples = 10_000
posterior_samples = q_dist.sample((n_posterior_samples,)).numpy()

# 2) Histogram-based density estimate
hist_counts, bin_edges = np.histogram(
    posterior_samples, bins=100, density=True
)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3) Smooth KDE estimate
kde = gaussian_kde(posterior_samples)
pdf_kde = kde(bin_centers)

# 4) Plot both the estimates and the true posterior
plt.figure(figsize=(8, 4))
plt.bar(bin_centers, hist_counts,
        width=bin_edges[1] - bin_edges[0],
        alpha=0.4, label='Histogram')
plt.plot(bin_centers, pdf_kde, label='KDE', linewidth=2)
plt.plot(x.numpy(), true_pdf.numpy(), '--',
         label='True Posterior', linewidth=2)
plt.xlabel('θ')
plt.ylabel('Density')
plt.title('Empirical PDF from Samples vs True Posterior')
plt.legend()
plt.tight_layout()
plt.show()
