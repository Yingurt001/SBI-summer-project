from scipy.stats import norm, poisson, beta, binom, gamma, poisson  # import distributions
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

"""
Gamma toy example: approximate a Gamma posterior p(θ | y_sum) with a Gaussian DNN.

- Prior: θ ∼ Gamma(shape=alpha=2, rate=beta_para=3).
- Simulate M samples of θ and corresponding y_sum, where y_sum is the sum of 
  `n_trials` Poisson observations with mean θ.
- Group θ samples by each unique y_sum to form empirical posteriors.
- Train a neural network to predict the posterior mean and variance of θ given y_sum.
- Finally, compare the Gaussian approximation (from the network) to the true Gamma posterior.
"""

# ------------------------------------------------------------------------------
# Reproducibility: fix random seeds across libraries
SEED = 123
random.seed(SEED)         # seed for Python random
np.random.seed(SEED)      # seed for NumPy random
torch.manual_seed(SEED)   # seed for PyTorch random

# Ensure deterministic behavior in cuDNN (optional)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# Define a neural network that outputs mean and log-variance for θ | y_sum
class ProbabilisticNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Sub-network to predict the mean (μ)
        self.mean_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        # Sub-network to predict the log-variance (log σ²)
        self.log_var_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Forward pass: compute μ and log σ² from input x
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # return predicted mean and log-variance

# ------------------------------------------------------------------------------
# Prior parameters for θ ∼ Gamma(shape=alpha, rate=beta_para)
alpha = 2
beta_para = 3

# Number of Monte Carlo samples
M = 10**6

# ------------------------------------------------------------------------------
# Data simulation: for each θ_i, sample y_sum_i = sum of n_trials Poisson(θ_i)
n_trials = 100

# 1. Sample θ_i from the Gamma prior
theta_samples = gamma.rvs(a=alpha, loc=0, scale=1/beta_para, size=M)

# 2. For each θ_i, simulate `n_trials` Poisson draws and sum them
y_samples = np.sum(
    poisson.rvs(mu=theta_samples[:, None], size=(len(theta_samples), n_trials)),
    axis=1
)

# Reshape samples into column vectors (for consistency)
theta_np = theta_samples.reshape(-1, 1)
y_np = y_samples.reshape(-1, 1)

# Combine θ and y_sum into a single array for grouping
out_mat = np.column_stack([theta_samples, y_samples])

# ------------------------------------------------------------------------------
# Group θ samples by each unique y_sum to build empirical posteriors
theta_dict = defaultdict(list)
unique_y_vals = np.unique(out_mat[:, 1])  # all observed y_sum values

for y in unique_y_vals:
    # Select θ values corresponding to this y_sum
    sel_index = (out_mat[:, 1] == y)
    theta_vals = out_mat[sel_index, 0]
    theta_dict[y] = theta_vals

# ------------------------------------------------------------------------------
# Compute empirical posterior means and variances for each y_sum
post_means = {}
post_vars = {}

for y, thetas in theta_dict.items():
    if len(thetas) >= 2:
        post_means[y] = np.mean(thetas)
        post_vars[y] = np.var(thetas)
    elif len(thetas) == 1:
        post_means[y] = thetas[0]
        post_vars[y] = 0.0
    else:
        # No samples; default to zero
        post_means[y] = 0.0
        post_vars[y] = 0.0

# Ensure every y_sum has at least one sample array
for y in theta_dict:
    if len(theta_dict[y]) == 0:
        theta_dict[y] = np.array([0.0])

# ------------------------------------------------------------------------------
# Instantiate the model and optimizer
model = ProbabilisticNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------------------------------------------------------------------
# Training loop: maximize total log-likelihood over y_sum groups
y_max = np.max(out_mat[:, 1])  # for normalizing y_sum

for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0

    # Loop over each unique y_sum value
    for y in unique_y_vals:
        # Skip if no variability in θ samples
        if post_vars[y] == 0:
            continue

        # Normalize y_sum to [0,1] for network input
        x_input = torch.tensor([[y / y_max]], dtype=torch.float32)

        # Predict μ and log σ² for θ | y_sum
        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)  # convert log-variance to variance

        # Actual θ values for this y_sum
        theta_vals = torch.tensor(theta_dict[y], dtype=torch.float32).view(-1, 1)

        # Compute Gaussian log-probabilities under predicted parameters
        log_probs = -0.5 * (
            np.log(2 * np.pi)             # constant term log(2π)
            + log_var_pred                # log σ²
            + (theta_vals - mu_pred)**2 / var_pred  # squared error / σ²
        )

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
# Evaluation: compare learned Gaussian approximation to true Gamma posterior
y_obs = 20
x_test = torch.tensor([[y_obs / y_max]], dtype=torch.float32)

with torch.no_grad():
    # Predict approximate posterior parameters for θ | y_obs
    mu_pred, log_var_pred = model(x_test)
    var_pred = torch.exp(log_var_pred)
    mu_val = mu_pred.item()
    sigma_val = np.sqrt(var_pred.item())

# True Gamma posterior for comparison
true_posterior = gamma(a=alpha + y_obs, scale=1 / (beta_para + n_trials))

# Prepare θ grid for plotting densities
theta_range = np.linspace(0.001, 0.999, 300)
true_pdf = true_posterior.pdf(theta_range)
approx_pdf = norm.pdf(theta_range, loc=mu_val, scale=sigma_val)

# Plot comparison of the true vs. approximate posterior PDFs
plt.figure(figsize=(8, 5))
plt.plot(theta_range, true_pdf, label="True Posterior (Gamma)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)",
         lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

