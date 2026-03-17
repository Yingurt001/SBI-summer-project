'''
What we do:

Prior: θ∼Beta(2,5).
Same MDN‐style network (mean_net + log_var_net).
Simulate M=10⁶ (θ,y) pairs, groups θ by each y∈[0..100], computes empirical mean/var.
Train the network by maximizing the Gaussian log-likelihood of all θ’s for each y.
Plot true vs approximated posterior for a chosen y_obs.

'''

from scipy.stats import norm, poisson, beta, binom  # statistical distributions
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

# ------------------------------------------------------------------------------
# Reproducibility: fix random seeds across libraries
SEED = 123
random.seed(SEED)         # Python built-in RNG
np.random.seed(SEED)      # NumPy RNG
torch.manual_seed(SEED)   # PyTorch RNG

# Ensure deterministic behavior in cuDNN (optional)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# Define a neural network that outputs a mean and log-variance given input x
class ProbabilisticNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Sub-network for predicting the mean (μ)
        self.mean_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        # Sub-network for predicting the log-variance (log σ²)
        self.log_var_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Forward pass: compute μ and log σ²
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # return predicted mean and log-variance

# ------------------------------------------------------------------------------
# Prior parameters for θ ∼ Beta(alpha, beta_para)
alpha = 2
beta_para = 5

# Number of Monte Carlo samples for simulation
M = 10**6

# ------------------------------------------------------------------------------
# Binomial data simulation setup
n_trials = 100

# 1. Sample θ from the Beta(2,5) prior
theta_samples = beta.rvs(alpha, beta_para, size=M)

# 2. For each θ, sample y ∼ Binomial(n_trials, θ)
y_samples = binom.rvs(n=n_trials, p=theta_samples)

# Reshape into column vectors (not strictly needed later, but kept for clarity)
theta_np = theta_samples.reshape(-1, 1)
y_np = y_samples.reshape(-1, 1)

# ------------------------------------------------------------------------------
# Combine θ and y into a single array for indexing
out_mat = np.column_stack([theta_samples, y_samples])

# ------------------------------------------------------------------------------
# Build a dictionary mapping each observed count i to all θ that produced it
theta_dict = defaultdict(list)
for i in range(n_trials + 1):  # i ranges from 0 to n_trials
    sel_index = (out_mat[:, 1] == i)      # mask for y == i
    theta_vals = out_mat[sel_index, 0]    # extract corresponding θ values
    theta_dict[i] = theta_vals            # store in dictionary

# ------------------------------------------------------------------------------
# Compute empirical posterior means and variances for each y = i
post_means = {}
post_vars = {}
for i, thetas in theta_dict.items():
    if len(thetas) >= 2:
        post_means[i] = np.mean(thetas)
        post_vars[i] = np.var(thetas)
    elif len(thetas) == 1:
        post_means[i] = thetas[0]
        post_vars[i] = 0.0
    else:
        # No samples for this count: default to zero
        post_means[i] = 0.0
        post_vars[i] = 0.0

# Ensure every key has at least one value array
for key, thetas in theta_dict.items():
    if len(thetas) == 0:
        theta_dict[key] = np.array([0.0])

# ------------------------------------------------------------------------------
# Instantiate the model and optimizer
model = ProbabilisticNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------------------------------------------------------------------
# Training loop: maximize total log-likelihood across all y classes
for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0

    # Loop over each possible count i = 0..n_trials
    for i in range(n_trials + 1):
        # Skip if empirical variance is zero (no or single sample)
        if post_vars[i] == 0:
            continue

        # Normalize input x = i / n_trials
        x_input = torch.tensor([[i / n_trials]], dtype=torch.float32)

        # Predict mean and log-variance for θ | y = i
        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)  # convert log-variance to variance

        # True θ values that produced y = i
        theta_vals = torch.tensor(theta_dict[i], dtype=torch.float32).view(-1, 1)

        # Compute log-probabilities under predicted Gaussian
        log_probs = -0.5 * (
            torch.log(torch.tensor(2 * torch.pi))  # constant term log(2π)
            + log_var_pred                         # log σ²
            + (theta_vals - mu_pred) ** 2 / var_pred  # squared error / σ²
        )

        # Accumulate log-likelihood
        total_log_likelihood += log_probs.sum()

    # Negative log-likelihood as loss
    loss = -total_log_likelihood

    # Backpropagation step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# ------------------------------------------------------------------------------
# Evaluation for a particular observed count y_obs
y_obs = 60
x_test = torch.tensor([[y_obs / n_trials]], dtype=torch.float32)

with torch.no_grad():
    # Predict approximate posterior parameters for θ | y_obs
    mu_pred, log_var_pred = model(x_test)
    var_pred = torch.exp(log_var_pred)
    mu_val = mu_pred.item()
    sigma_val = np.sqrt(var_pred.item())

# ------------------------------------------------------------------------------
# True Beta posterior for comparison: Beta(α + y_obs, β + n_trials - y_obs)
true_posterior = beta(a=alpha + y_obs, b=beta_para + n_trials - y_obs)

# Prepare a grid over θ ∈ (0,1)
theta_range = np.linspace(0.001, 0.999, 300)

# Compute true Beta posterior PDF and Gaussian approximation PDF
true_pdf = true_posterior.pdf(theta_range)
approx_pdf = norm.pdf(theta_range, loc=mu_val, scale=sigma_val)

# Plot the two densities for visual comparison
plt.figure(figsize=(8, 5))
plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)", lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
