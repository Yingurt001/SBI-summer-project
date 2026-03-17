"""
Defines a 1D unconditional normalizing flow using a learnable affine + sigmoid transform.
  - Samples z ∼ N(0,1)
  - Transforms x = sigmoid(a·z + b)
  - Computes log-density via change-of-variables
  - Minimizes negative log-likelihood to match a Beta(2,5) target
  - Plots learned vs. true Beta PDF
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal
import numpy as np
# Set random seed for reproducibility
torch.manual_seed(0)

# Define the base distribution: standard normal N(0, 1)
base_dist = Normal(loc=0.0, scale=1.0)

# Sample 500 points from the base distribution
z = base_dist.sample((1000,))

# Plot the histogram of the base samples
plt.hist(z.numpy(), bins=30, density=True, alpha=0.6, label='Base: N(0,1)')
plt.title("Base distribution z ~ N(0,1)")
plt.legend()
plt.show()


from torch.distributions import Beta
# Your observed data x ~ Beta(2, 5)
target_dist = Beta(2.0, 5.0)
x_data = target_dist.sample((512,))  # Use x as training data

import torch.nn as nn
import torch

class InvertibleSigmoidAffineTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        return torch.sigmoid(self.a * z + self.b)

    def inverse(self, x):
        # Clamp to avoid log(0) or log(∞)
        x = x.clamp(1e-4, 1 - 1e-4)
        t = torch.log(x / (1 - x))  # inverse sigmoid
        return (t - self.b) / self.a

    def log_abs_det_jacobian(self, x):
        # Use inverse pass
        z = self.inverse(x)
        t = self.a * z + self.b
        s = torch.sigmoid(t)
        return torch.log(torch.abs(self.a) * s * (1 - s) + 1e-8)
    
    
    
def compute_nll(x, transform, base_dist):
    # Invert x to get z
    z = transform.inverse(x)

    # Base log-prob in z space
    log_p_z = base_dist.log_prob(z)

    # Log determinant of Jacobian
    log_det = transform.log_abs_det_jacobian(x)

    # Change-of-variable formula: log p(x) = log p(z) - log|df/dz|
    return -(log_p_z - log_det).sum()



# Transform
transform = InvertibleSigmoidAffineTransform()
optimizer = torch.optim.Adam(transform.parameters(), lr=1e-2)

# Train using negative log-likelihood
for epoch in range(1000):
    x_batch = x_data[torch.randperm(len(x_data))[:256]]
    loss = compute_nll(x_batch, transform, base_dist)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, NLL: {loss.item():.4f}")

def f(z):
    z_tensor = torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        return transform.forward(z_tensor).numpy()

# Compute gradient of that function using finite differences
def df_dz(z):
    eps = 1e-4
    return (f(z + eps) - f(z - eps)) / (2 * eps)


from scipy.stats import norm
z = np.arange(-3,3,0.01)
pr_z = norm.pdf(z, loc=0, scale=1)
x = f(z)
pr_x = pr_z*1/np.abs(df_dz(z))


fig, ax = plt.subplots()

# Plot learned density via change of variables
ax.plot(x, pr_x, label="Learned p(x) via f(z)", linewidth=2)

# Plot target Beta(2,5) PDF
x_curve = torch.linspace(0.01, 0.99, 500)
y_curve = target_dist.log_prob(x_curve).exp().numpy()
ax.plot(x_curve.numpy(), y_curve, label="Target Beta(2,5)", linewidth=2)

# Formatting
ax.set_xlim([0, 1])
ax.set_ylim([0, 3])
ax.set_xlabel('x')
ax.set_ylabel('Pr(x)')
ax.set_title("Learned density vs Target")
ax.grid(True)
ax.legend()

plt.show()
