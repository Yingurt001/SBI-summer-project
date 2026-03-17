'''
What we do:

Prior: θ∼Beta(1,1).
Simulate M=10⁶ draws of θ and y∼Binomial(n=100,θ).
Fit a tiny MDN/DNN (predicting μ and log σ²) by maximizing the empirical Gaussian log-likelihood of the θ samples for each y.
Plot the true Beta posterior vs the learned Gaussian approximation at a single y.

'''

import os
# Ensure OpenMP conflicts don't crash PyTorch imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.stats import norm, beta, binom  # statistical distributions
import numpy as np                         # numerical operations
import matplotlib.pyplot as plt            # plotting
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
import math  # math functions like log, pi

# --------------------
# Reproducibility settings
SEED = 123
random.seed(SEED)         # seed Python RNG
np.random.seed(SEED)      # seed NumPy RNG
torch.manual_seed(SEED)   # seed PyTorch RNG
# Optional: make cuDNN deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------
# Neural network definition
class ProbabilisticNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Network to predict the mean μ of the approximate posterior
        self.mean_net = nn.Sequential(
            nn.Linear(1, 64),  # input: y (as fraction), hidden:64
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1)     # output: scalar μ
        )
        # Network to predict the log-variance log σ² of the approximate posterior
        self.log_var_net = nn.Sequential(
            nn.Linear(1, 16),  # input: y, hidden:16
            nn.ReLU(),
            nn.Linear(16, 1)    # output: scalar log σ²
        )

    def forward(self, x):
        # Forward pass returns predicted μ and log σ²
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # return log_var (stabilizes variance learning)

# --------------------
# Prior parameters for Beta distribution
alpha = 1         # prior alpha
beta_para = 1     # prior beta

# --------------------
# Simulation settings
M = 10**6         # number of simulated samples
n_trials = 100    # number of trials in Binomial

# --------------------
# Simulate (θ, y) pairs
# θ ~ Beta(alpha, beta)
theta_samples = beta.rvs(alpha, beta_para, size=M)
# y | θ ~ Binomial(n_trials, θ)
y_samples = binom.rvs(n=n_trials, p=theta_samples)

# Combine θ and y into a single matrix for grouping
out_mat = np.column_stack([theta_samples, y_samples])

# --------------------
# Group θ values by observed y
theta_dict = defaultdict(list)
for i in range(n_trials + 1):
    mask = (out_mat[:, 1] == i)       # select rows where y == i
    theta_dict[i] = out_mat[mask, 0]  # store corresponding θ array

# --------------------
# Compute empirical posterior stats for each y
post_means = {}
post_vars = {}
for y_val, thetas in theta_dict.items():
    if len(thetas) >= 2:
        post_means[y_val] = np.mean(thetas)
        post_vars[y_val]  = np.var(thetas)
    elif len(thetas) == 1:
        post_means[y_val] = thetas[0]
        post_vars[y_val]  = 0.0
    else:
        # no samples: set to zero to skip in training
        post_means[y_val] = 0.0
        post_vars[y_val]  = 0.0

# Replace empty groups with a dummy array to avoid errors
for y_val in theta_dict:
    if len(theta_dict[y_val]) == 0:
        theta_dict[y_val] = np.array([0.0])

# --------------------
# Instantiate model and optimizer
model = ProbabilisticNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --------------------
# Training loop: maximize grouped log-likelihood
for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0  # accumulator for this epoch

    # Loop over each possible observed y
    for y_val in range(n_trials + 1):
        # Skip groups with no variance (no real data)
        if post_vars[y_val] == 0.0:
            continue

        # Prepare input tensor: y as proportion of successes
        x_input = torch.tensor([[y_val / n_trials]], dtype=torch.float32)

        # Get predicted μ and log σ² from the network
        mu_pred, log_var_pred = model(x_input)
        # Convert log-variance to variance
        var_pred = torch.exp(log_var_pred)

        # True θ samples for this y group
        theta_vals = torch.tensor(theta_dict[y_val], dtype=torch.float32).view(-1, 1)

        # Compute log-probability under Normal(μ, σ²)
        # log p(θ) = -0.5 * [log(2π) + log σ² + (θ - μ)² / σ²]
        log_probs = -0.5 * (
            math.log(2 * math.pi) + log_var_pred +
            (theta_vals - mu_pred).pow(2) / var_pred
        )

        # Accumulate sum of log-likelihoods
        total_log_likelihood += log_probs.sum()

    # Define loss as negative total log-likelihood
    loss = -total_log_likelihood

    # Backpropagation step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# --------------------
# Evaluation: compare NN approximation vs true Beta posterior
# Choose a test observed count
y_obs = 20
x_test = torch.tensor([[y_obs / n_trials]], dtype=torch.float32)

with torch.no_grad():
    # Predict μ and log σ² for test y
    mu_pred, log_var_pred = model(x_test)
    sigma_pred = math.sqrt(torch.exp(log_var_pred).item())
    mu_pred = mu_pred.item()

# True Beta posterior: parameters update
true_posterior = beta(a=alpha + y_obs, b=beta_para + n_trials - y_obs)

# Grid over θ for plotting
theta_range = np.linspace(0.001, 0.999, 300)
# True posterior PDF (Beta)
true_pdf   = true_posterior.pdf(theta_range)
# Neural net approximation PDF (Gaussian)
approx_pdf = norm.pdf(theta_range, loc=mu_pred, scale=sigma_pred)

# Plot comparison of densities
plt.figure(figsize=(8,5))
plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", linewidth=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)", linestyle="--", linewidth=2)
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
