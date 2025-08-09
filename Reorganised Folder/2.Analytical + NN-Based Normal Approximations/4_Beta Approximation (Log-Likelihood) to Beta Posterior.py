'''
What we do:
Prior: θ∼Beta(1,1), M=10⁶, Binomial(n=100).
New modeling choice: network now outputs two positive scalars (α, β) via Softplus.
Trains by maximizing the exact Beta log-likelihood of the grouped θ’s, i.e. uses the true posterior family.
Finally plots the true posterior vs the network’s Beta(α̂, β̂ ).

'''


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import beta, binom
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ============================== #
#      1. Global Configuration
# ============================== #
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================== #
#      2. Define Model
# ============================== #
class BetaPosteriorNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Softplus()  # Ensure alpha > 0
        )
        self.beta_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensure beta > 0
        )

    def forward(self, x):
        alpha = self.alpha_net(x)
        beta = self.beta_net(x)
        return alpha, beta

# ============================== #
#      3. Data Generation
# ============================== #
def simulate_beta_binomial_data(M, n_trials, alpha, beta_param):
    """
    Generate M samples from:
        θ ~ Beta(alpha, beta)
        y ~ Binomial(n_trials, θ)
    Returns:
        theta_samples: (M, 1)
        y_samples: (M, 1)
    """
    theta_samples = beta.rvs(a=alpha, b=beta_param, size=M)
    y_samples = binom.rvs(n=n_trials, p=theta_samples)
    return theta_samples.reshape(-1, 1), y_samples.reshape(-1, 1)

# ============================== #
#      4. Group theta by y
# ============================== #
def group_theta_by_y(theta_samples, y_samples):
    """
    Group theta samples by unique y values
    Returns: dict { y_val : [theta_list] }
    """
    data = np.column_stack([theta_samples.squeeze(), y_samples.squeeze()])
    theta_dict = defaultdict(list)
    for y in np.unique(y_samples):
        theta_dict[y] = data[data[:,1] == y, 0]
    return theta_dict

# ============================== #
#      5. Train Neural Network
# ============================== #
def train_model(model, theta_dict, y_max, epochs=1000, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_log_likelihood = 0.0
        model.train()
        for y_val, theta_vals in theta_dict.items():
            if len(theta_vals) < 20:
                continue
            x = torch.tensor([[y_val / y_max]], dtype=torch.float32)
            theta = torch.tensor(theta_vals, dtype=torch.float32).view(-1, 1)
            theta = torch.clamp(theta, 1e-6, 1 - 1e-6)
            alpha_pred, beta_pred = model(x)

            log_probs = (
                (alpha_pred - 1) * torch.log(theta) +
                (beta_pred - 1) * torch.log(1 - theta) -
                torch.lgamma(alpha_pred) - torch.lgamma(beta_pred) +
                torch.lgamma(alpha_pred + beta_pred)
            )
            total_log_likelihood += log_probs.sum()
        
        loss = -total_log_likelihood
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ============================== #
#      6. Visualize Posterior
# ============================== #
def plot_posterior_comparison(model, y_obs, n_trials, prior_alpha, prior_beta, y_max):
    x = torch.tensor([[y_obs / y_max]], dtype=torch.float32)
    with torch.no_grad():
        alpha_pred, beta_pred = model(x)
    alpha_val = alpha_pred.item()
    beta_val = beta_pred.item()

    # True posterior
    true_posterior = beta(a=prior_alpha + y_obs, b=prior_beta + n_trials - y_obs)

    theta_range = np.linspace(0.001, 0.999, 300)
    true_pdf = true_posterior.pdf(theta_range)
    nn_pdf = beta.pdf(theta_range, a=alpha_val, b=beta_val)

    plt.figure(figsize=(8,5))
    plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
    plt.plot(theta_range, nn_pdf, '--', label="NN Approx Posterior", lw=2)
    plt.title(f"Posterior Comparison | y = {y_obs}")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================== #
#      7. Main Program
# ============================== #
if __name__ == "__main__":
    # Settings
    prior_alpha = 1
    prior_beta = 1
    M = 10**6
    n_trials = 100

    print("Generating data...")
    theta_samples, y_samples = simulate_beta_binomial_data(M, n_trials, prior_alpha, prior_beta)
    theta_dict = group_theta_by_y(theta_samples, y_samples)
    y_max = y_samples.max()

    model = BetaPosteriorNN()
    print("Training model...")
    train_model(model, theta_dict, y_max, epochs=1000, lr=1e-2)

    # Visualize one case
    y_obs = 99
    plot_posterior_comparison(model, y_obs, n_trials, prior_alpha, prior_beta, y_max)
