"""
Simulates Gamma prior + Poisson likelihood and learns a NN to estimate posterior moments.
  - θ ∼ Gamma(α,β), y_sum = ∑_{i=1}^n Poisson(θ), for n = 5…49
  - Groups θ by (y_sum,n), computes empirical posterior mean & var
  - Defines SplitGammaNN mapping [y_sum,n] → [mean,var], trained with weighted MSE
  - Plots posterior mean & variance vs y_sum and shows fit quality
"""

import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import norm, poisson, beta, binom

# Set seed
SEED = 123
random.seed(SEED)         # Python random
np.random.seed(SEED)      # NumPy random
torch.manual_seed(SEED)   # PyTorch random

# Set prior
alpha = 1
beta_prior = 1
theta_true = 2
M = 10**3  # number of samples per n

final_mat = []

# Loop over different n values
for n in range(5, 50):
    theta_vec = np.full(M, np.nan)
    y_sum_vec = np.full(M, np.nan)

    for i in range(M):  
        theta = np.random.gamma(shape=alpha, scale=1/beta_prior)
        theta_vec[i] = theta
        y_sum_vec[i] = np.sum(np.random.poisson(lam=theta, size=n))

    # Now create n_vec and combine once after filling all M rows
    n_vec = np.full(M, n)
    out_matrix = np.column_stack((theta_vec, y_sum_vec, n_vec))
    final_mat.append(out_matrix)  # append (M x 3) matrix

# Stack all (M x 3) matrices into one big matrix
final_matrix = np.vstack(final_mat)  # Shape: ((50-5)*M, 3)
print(final_matrix.shape)  # Should be (45000, 3)


# Extract columns 2 and 3: ysum and n
ysum_n = final_matrix[:, [1, 2]]

# Keep unique rows only
ysum_n_unique = np.unique(ysum_n, axis=0)

# Sort by ysum (column 0)
ysum_n_sorted = ysum_n_unique[np.argsort(ysum_n_unique[:, 0])]

print(ysum_n_sorted[:10])

# Step 1: extract columns
theta_all = final_matrix[:, 0]
ysum_all = final_matrix[:, 1].astype(int)  # ysum might be float, cast to int
n_all = final_matrix[:, 2].astype(int)

# Step 2: combine ysum and n into tuples for grouping
ysum_n_pairs = np.column_stack((ysum_all, n_all))
unique_pairs = np.unique(ysum_n_pairs, axis=0)

# Step 3: initialize a dictionary to store grouped theta values
theta_dict = {}

# Step 4: group theta values by (ysum, n)
for pair in unique_pairs:
    y, n_val = pair
    # Find the indices matching this (ysum, n)
    mask = (ysum_all == y) & (n_all == n_val)
    
    # Extract corresponding θ values
    theta_group = theta_all[mask]
    
    # Store in dictionary
    theta_dict[(y, n_val)] = theta_group

# Group θ samples by ysum only (ignoring n)
theta_by_ysum = defaultdict(list)

for (ysum, n), theta_vals in theta_dict.items():
    theta_by_ysum[ysum].extend(theta_vals)

ysum_values = sorted(theta_by_ysum.keys())

posterior_mean = []
posterior_var = []
posterior_sd = []

for y in ysum_values:
    theta_vals = np.array(theta_by_ysum[y])
    posterior_mean.append(np.mean(theta_vals))
    posterior_var.append(np.var(theta_vals, ddof=1))  # sample variance
    posterior_sd.append(np.std(theta_vals, ddof=1))


plt.figure(figsize=(8, 5))
plt.scatter(ysum_values, posterior_mean, marker='o')
plt.xlabel('Observed total count (ysum)')
plt.ylabel('Posterior mean of θ')
plt.title('Posterior Mean of θ vs. ysum')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(ysum_values, posterior_var, marker='o')
plt.xlabel('Observed total count (ysum)')
plt.ylabel('Posterior variance of θ')
plt.title('Posterior variance of θ vs. ysum')
plt.grid(True)
plt.tight_layout()
plt.show()



## Now we try neural network to estimate this

# Given y_sum and its corresponding N , can we find its mean and variance?



# Define the deep neural network
import torch.nn as nn

class SplitGammaNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.var_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        mean_out = self.mean_head(shared)
        var_out = self.var_head(shared)
        return torch.cat([mean_out, var_out], dim=1)  # [batch_size, 2]
def weighted_mse(pred, target, mean_weight=1.0, var_weight=2.0):
    mean_loss = (pred[:, 0] - target[:, 0]) ** 2
    var_loss = (pred[:, 1] - target[:, 1]) ** 2
    return (mean_weight * mean_loss.mean()) + (var_weight * var_loss.mean())


# For this kind of question , do we use the loss function of Gamma?

# At this stage we use Mean Squared Error (MSE) loss:
    
X_clean = []
y_clean = []

for (y, n), thetas in theta_dict.items():
    thetas = np.array(thetas)
    if len(thetas) > 1:
        mu = np.mean(thetas)
        var = np.var(thetas, ddof=1)
        if not np.isnan(var):
            X_clean.append([y, n])
            y_clean.append([mu, var])



X_train = np.array(X_clean)
y_train = np.array(y_clean)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

model = SplitGammaNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 500
batch_size = 64

indices = np.arange(len(X_train_tensor))

for epoch in range(n_epochs):
    np.random.shuffle(indices)
    
    for start_idx in range(0, len(indices), batch_size):
        end_idx = start_idx + batch_size
        batch_idx = indices[start_idx:end_idx]
        
        x_batch = X_train_tensor[batch_idx]
        y_batch = y_train_tensor[batch_idx]

        preds = model(x_batch)
        loss = weighted_mse(preds, y_batch)  # or nn.MSELoss()(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")


model.eval()
with torch.no_grad():
    preds = model(X_train_tensor).numpy()

# Extract mean and variance predictions
pred_mean = preds[:, 0]
pred_var = preds[:, 1]

true_mean = y_train[:, 0]
true_var = y_train[:, 1]
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Posterior Mean
plt.subplot(1, 2, 1)
plt.scatter(true_mean, pred_mean, alpha=0.6)
plt.plot([true_mean.min(), true_mean.max()],
         [true_mean.min(), true_mean.max()], 'r--')
plt.xlabel('True Posterior Mean')
plt.ylabel('Predicted Posterior Mean')
plt.title('Mean Comparison')
plt.grid(True)

# Posterior Variance
plt.subplot(1, 2, 2)
plt.scatter(true_var, pred_var, alpha=0.6)
plt.plot([true_var.min(), true_var.max()],
         [true_var.min(), true_var.max()], 'r--')
plt.xlabel('True Posterior Variance')
plt.ylabel('Predicted Posterior Variance')
plt.title('Variance Comparison')
plt.grid(True)

plt.tight_layout()
plt.show()


