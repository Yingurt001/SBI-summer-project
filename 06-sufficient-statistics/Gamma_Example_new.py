"""
Fixed-n (n=50) Gamma–Poisson simulation with NN-based Normal approximation via likelihood maximization.
  - θ ∼ Gamma(α,β), y_sum for n = 50
  - Groups θ by (y_sum,n), computes empirical mean & var
  - Trains GammaNN [y_sum/y_max, n/n_max] → [μ, logσ²] by maximizing Gaussian log-likelihood
  - Tests on (y_sum,n) pairs: compares NN Normal vs true Gamma posterior PDF/K-S & Wasserstein
"""

import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, ks_2samp, wasserstein_distance

# Set seed
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set prior parameters
alpha = 2
beta_prior = 5
M = 10000  # number of samples per n

df_final = pd.DataFrame()

# Loop over different n values
for n in range(50,51):
    theta_vec = np.random.gamma(shape=alpha, scale=1/beta_prior, size=M)
    y_sum_vec = [np.sum(np.random.poisson(lam=theta, size=n)) for theta in theta_vec]
    n_vec = np.full(M, n)
    df = pd.DataFrame({'theta': theta_vec, 'y_sum': y_sum_vec, 'n': n_vec})
    df_final = pd.concat([df_final, df], ignore_index=True)

## Now we've obtained the df_final that we have theta_vec , y_sum , n_vec at the same time
'''
Indeed we can check the relationshio between theta,y_sum and n_vec, 
'''
plt.figure(figsize=(6, 4))
plt.scatter(df_final['theta'] * df_final['n'], df_final['y_sum'], alpha=0.3)
plt.plot([0, max(df_final['y_sum'])], [0, max(df_final['y_sum'])], 'r--', label='Ideal line: y_sum = n·theta')
plt.xlabel("n · theta")
plt.ylabel("y_sum")
plt.title("Check if y_sum ≈ n · theta")
plt.legend()
plt.grid(True)
plt.show()


# We want unique (y_sum,n) because now we want all the thetas for the same pair,
# Get unique (y_sum, n) pairs
unique_pairs = df_final[['y_sum', 'n']].drop_duplicates().reset_index(drop=True)# This code extract all the unique (y_sum.n) pairs

thetas = [
    df_final[(df_final['y_sum'] == row['y_sum']) & (df_final['n'] == row['n'])]['theta'].values
    for _, row in unique_pairs.iterrows()
]
# okay since we'e saved the thetas for each unique pairs, form example , our first row  of unique_pairs is (2,50), we find 
# all the (2,50) in df_final, then we collect the correspoinding theta, hence , one pair, one theta, 
# Two lists are of the same shape , i.e. 5000 pairs. 
post_means = [np.mean(t) for t in thetas]
post_vars = [np.var(t) for t in thetas]

# We check the variables' shape and they are both 5000, which is correct

# Now our input is y_sum and n, our out put is mean and log.variance, this is very easy in terms of neural network

# Define NN model
class GammaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.ReLU(),
            nn.Linear(4, 1)
        )
        self.log_var_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4), nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var

model = GammaNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
y_max = df_final['y_sum'].max()
n_max = df_final['n'].max()

# Training
for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0
    # for y_val in df_final['y_sum'].unique():
    #     theta_dict[y_val] = df_final[df_final['y_sum'] == y_val]['theta'].values
        
    for i, ((y_val, n_val), theta_arr) in enumerate(zip(unique_pairs.values, thetas)):
        if len(theta_arr) == 0 or np.var(theta_arr) == 0:
            continue

        x_input = torch.tensor([[y_val / y_max, n_val / n_max]], dtype=torch.float32)

        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)
        theta_vals = torch.tensor(theta_arr, dtype=torch.float32).view(-1, 1)

        
        
        log_probs = -0.5 * (np.log(2 * np.pi) + log_var_pred + (theta_vals - mu_pred) ** 2 / var_pred)
        total_log_likelihood += log_probs.sum()
    loss = -log_probs.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_log_likelihood:.4f}")


# Evaluation on multiple test cases
test_cases = [(5, 5)]
theta_vals = np.linspace(0.01, 4.0, 500)

for y_val, n_val in test_cases:
    x_test = torch.tensor([[y_val / y_max, n_val / n_max]], dtype=torch.float32)
    with torch.no_grad():
        mu_pred, log_var_pred = model(x_test)
        mu = mu_pred.item()
        sigma = np.sqrt(torch.exp(log_var_pred).item())

    alpha_post = alpha + y_val
    beta_post = beta_prior + n_val
    true_pdf = gamma.pdf(theta_vals, a=alpha_post, scale=1 / beta_post)
    approx_pdf = norm.pdf(theta_vals, loc=mu, scale=sigma)

    true_samples = gamma.rvs(a=alpha_post, scale=1 / beta_post, size=2000)
    approx_samples = np.random.normal(loc=mu, scale=sigma, size=2000)
    ks, _ = ks_2samp(true_samples, approx_samples)
    wd = wasserstein_distance(true_samples, approx_samples)

    plt.figure()
    plt.plot(theta_vals, true_pdf, label='True Gamma Posterior')
    plt.plot(theta_vals, approx_pdf, label='NN Approx Normal', linestyle='--')
    plt.title(f"Posterior Comparison (y_sum={y_val}, n={n_val})\nKS={ks:.3f}, W={wd:.3f}")
    plt.xlabel("θ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()




# # Assume model is already trained
# model.eval()


# y_test_list = [50]
# n_test_list = [50]  # Still used to compute true posterior

# # Normalize input and create test batch
# x_test_batch = torch.tensor([
#     [y / y_max] for y in y_test_list
# ], dtype=torch.float32)

# # Run through the model
# with torch.no_grad():
#     mu_pred, log_var_pred = model(x_test_batch)
#     mu_pred = mu_pred.view(-1).numpy()
#     var_pred = torch.exp(log_var_pred).view(-1).numpy()

# # ------------------------------
# # Plot posterior comparison
# # ------------------------------
# for i, (y_val, n_val) in enumerate(zip(y_test_list, n_test_list)):
#     # Compute true posterior parameters
#     alpha_post = alpha + y_val
#     beta_post = beta_prior + n_val

#     theta_vals = np.linspace(0.01, 5, 500)  # Wider range

#     # True posterior: Gamma
#     true_pdf = gamma.pdf(theta_vals, a=alpha_post, scale=1 / beta_post)

#     # Approximate posterior: Normal from NN
#     approx_pdf = norm.pdf(theta_vals, loc=mu_pred[i], scale=np.sqrt(var_pred[i]))

#     # Plot
#     plt.figure()
#     plt.plot(theta_vals, true_pdf, label='True Gamma Posterior', lw=2)
#     plt.plot(theta_vals, approx_pdf, label='NN Approx Normal', lw=2, linestyle='--')
#     plt.title(f"Posterior θ | y_sum={y_val}, n={n_val}")
#     plt.xlabel("θ")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(True)
#     plt.show()