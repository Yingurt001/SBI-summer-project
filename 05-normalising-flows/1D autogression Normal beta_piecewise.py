"""
Defines a 1D unconditional normalizing flow with a learnable piecewise‐linear monotonic map.
  - Partitions input range into K linear segments with learnable heights
  - Samples z ∼ N(0,1) and maps x via the piecewise function
  - Computes log-density via change-of-variables
  - Optimizes segment heights to fit a Beta(2,5) target
  - Plots resulting density against the true Beta PDF
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, Beta
from scipy.stats import norm

# Set seed
torch.manual_seed(0)

# Base and target distributions
base_dist = Normal(0, 1)
target_dist = Beta(2.0, 5.0)
x_data = target_dist.sample((512,))

# Define the Piecewise Linear Transform
class PiecewiseLinearTransform(nn.Module):
    def __init__(self, num_bins=10, tail_bound=3.0):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.heights_logits = nn.Parameter(torch.randn(num_bins))

    def forward(self, z):
        z = z.clamp(-self.tail_bound + 1e-4, self.tail_bound - 1e-4)
        bin_width = 2 * self.tail_bound / self.num_bins
        bin_idx = ((z + self.tail_bound) / bin_width).long().clamp(0, self.num_bins - 1)

        heights = F.softmax(self.heights_logits, dim=0)
        cum_heights = torch.cumsum(heights, dim=0)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)

        left_z = -self.tail_bound + bin_idx * bin_width
        bin_frac = (z - left_z) / bin_width

        h_left = cum_heights[bin_idx]
        h_right = cum_heights[bin_idx + 1]
        x = h_left + bin_frac * (h_right - h_left)
        return x.unsqueeze(-1)

    def inverse(self, x):
        x = x.clamp(1e-4, 1 - 1e-4)
        heights = F.softmax(self.heights_logits, dim=0)
        cum_heights = torch.cumsum(heights, dim=0)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)
    
        edges = cum_heights[:-1]  # shape [num_bins]
        x_expanded = x.view(-1, 1)  # shape [batch_size, 1]
        bin_idx = torch.sum(x_expanded >= edges, dim=1) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
    
        h_left = cum_heights[bin_idx]
        h_right = cum_heights[bin_idx + 1]
        bin_frac = (x.squeeze(-1) - h_left) / (h_right - h_left + 1e-6)
    
        bin_width = 2 * self.tail_bound / self.num_bins
        left_z = -self.tail_bound + bin_idx * bin_width
        z = left_z + bin_frac * bin_width
        return z.unsqueeze(-1)


    def log_abs_det_jacobian(self, x):
        x = x.clamp(1e-4, 1 - 1e-4)
        heights = F.softmax(self.heights_logits, dim=0)
        cum_heights = torch.cumsum(heights, dim=0)
        cum_heights = F.pad(cum_heights, (1, 0), value=0.0)
    
        edges = cum_heights[:-1]
        x_expanded = x.view(-1, 1)
        bin_idx = torch.sum(x_expanded >= edges, dim=1) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)
    
        h_left = cum_heights[bin_idx]
        h_right = cum_heights[bin_idx + 1]
        slope = (h_right - h_left) / (2 * self.tail_bound / self.num_bins)
        return torch.log(slope + 1e-8).unsqueeze(-1)


# Negative log-likelihood
def compute_nll(x, transform, base_dist):
    z = transform.inverse(x)
    log_p_z = base_dist.log_prob(z)
    log_det = transform.log_abs_det_jacobian(x)
    return -(log_p_z - log_det).sum()

# Initialize transform and optimizer
transform = PiecewiseLinearTransform()
optimizer = torch.optim.Adam(transform.parameters(), lr=1e-2)

# Training loop
for epoch in range(1000):
    x_batch = x_data[torch.randperm(len(x_data))[:256]]
    loss = compute_nll(x_batch, transform, base_dist)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, NLL: {loss.item():.4f}")

# Forward function for plotting
def f(z):
    z_tensor = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        return transform.forward(z_tensor).numpy()

# Numerical derivative of f
def df_dz(z):
    eps = 1e-4
    return (f(z + eps) - f(z - eps)) / (2 * eps)

# Visualization
z = np.arange(-3, 3, 0.01)
pr_z = norm.pdf(z)
x = f(z)
epsilon = 1e-8
pr_x = pr_z / (np.abs(df_dz(z)) + epsilon)


fig, ax = plt.subplots()
ax.plot(x, pr_x, label="Learned p(x) via f(z)", linewidth=2)

x_curve = torch.linspace(0.01, 0.99, 500)
y_curve = target_dist.log_prob(x_curve).exp().numpy()
ax.plot(x_curve.numpy(), y_curve, label="Target Beta(2,5)", linewidth=2)

ax.set_xlim([0, 1])
ax.set_ylim([0, 3])
ax.set_xlabel('x')
ax.set_ylabel('Pr(x)')
ax.set_title("Learned density vs Target")
ax.grid(True)
ax.legend()
plt.show()
