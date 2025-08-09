"""
Defines a neural network to predict Beta posterior mean & variance for Binomial data with varying n.
  - Simulates θ ∼ Beta(α,β) and y ∼ Binomial(n,θ) over n = 5,10,…,50
  - Groups samples by (y,n), computes empirical posterior mean & var
  - Trains a NN taking [y/n, log(n)] → [μ, logσ²] via Monte Carlo MLE
  - Evaluates on held-out (y,n), plots Normal approximation vs true Beta(α+y,β+n−y) PDF
"""
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

# Set seed
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set prior for beta distribution
alpha = 2
beta_para = 5
M = 100000  # number of samples per n

df_final = pd.DataFrame()

# Loop over different n values
for n_trials in range(5, 51, 5):
    theta_vec = np.random.beta(alpha, beta_para, size=M)
    y_vec = np.random.binomial(n=n_trials, p=theta_vec)

    df = pd.DataFrame({
        'theta': theta_vec,
        'y': y_vec,
        'n': n_trials
    })
    df_final = pd.concat([df_final, df], ignore_index=True)

# Group by (y, n)
unique_pairs = df_final[['y', 'n']].drop_duplicates().reset_index(drop=True)
thetas = []

for _, row in unique_pairs.iterrows():
    y_val, n_val = row['y'], row['n']
    theta_samples = df_final[
        (df_final['y'] == y_val) &
        (df_final['n'] == n_val)
    ]['theta'].values
    thetas.append(theta_samples)

post_means = [np.mean(t) for t in thetas]
post_vars = [np.var(t) for t in thetas]

# Train a simple NN: input = [y/n, log(n)], output = mean and log variance
class BetaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var

model = BetaNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(300):
    total_loss = 0.0
    for i, ((y_val, n_val), theta_arr) in enumerate(zip(unique_pairs.values, thetas)):
        if len(theta_arr) == 0:
            continue
        x_input = torch.tensor([[y_val / n_val, np.log(n_val)]], dtype=torch.float32)
        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)
        theta_vals = torch.tensor(theta_arr, dtype=torch.float32).view(-1, 1)
        log_probs = -0.5 * (np.log(2 * np.pi) + log_var_pred + (theta_vals - mu_pred)**2 / var_pred)
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")



from scipy.stats import ks_2samp, wasserstein_distance

# Define test (y, n) combinations
test_cases = [
    (1, 5),
    (2, 10),
    (5, 10),
    (5, 20),
    (10, 30),
    (15, 30),
    (20, 40)
]

# Create figure
plt.figure(figsize=(10, 4 * len(test_cases)))
theta_vals = np.linspace(0, 1, 500)

for i, (y_test, n_test) in enumerate(test_cases):
    # NN prediction
    x_test = torch.tensor([[y_test / n_test, np.log(n_test)]], dtype=torch.float32)
    with torch.no_grad():
        mu_pred, log_var_pred = model(x_test)
        mu = mu_pred.item()
        sigma = np.sqrt(torch.exp(log_var_pred).item())

    # True Beta posterior
    alpha_post = alpha + y_test
    beta_post = beta_para + n_test - y_test
    true_pdf = beta.pdf(theta_vals, alpha_post, beta_post)
    approx_pdf = norm.pdf(theta_vals, loc=mu, scale=sigma)

    # Sample for metrics
    true_samples = beta.rvs(alpha_post, beta_post, size=2000)
    approx_samples = np.random.normal(mu, sigma, size=2000)
    ks, p_val = ks_2samp(true_samples, approx_samples)
    wd = wasserstein_distance(true_samples, approx_samples)

    # Plot
    ax = plt.subplot(len(test_cases), 1, i + 1)
    ax.plot(theta_vals, true_pdf, label='True Beta Posterior', lw=2)
    ax.plot(theta_vals, approx_pdf, label='NN Approx Normal', lw=2, linestyle='--')
    ax.set_title(f"y={y_test}, n={n_test} | KS={ks:.3f}, W={wd:.3f}")
    ax.set_xlabel("θ")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
