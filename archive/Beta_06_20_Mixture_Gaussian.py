from scipy.stats import norm, poisson, beta, binom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

# Set seed value
SEED = 123
random.seed(SEED)         # Python random
np.random.seed(SEED)      # NumPy random
torch.manual_seed(SEED)   # PyTorch random

# For reproducibility (optional but recommended)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the deep neural network
class MixtureDensityNN(nn.Module):
    def __init__(self, K=8):
        super().__init__()
        self.K = K
        self.backbone = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(),
            nn.Linear(24 ,25),
            nn.ReLU(),
            nn.Linear(25, 16),
            nn.ReLU(),
        )
        #  the logits need to sum up to 1 , so we must apply a softmax
        self.logits_head = nn.Linear(16, K)      # raw logits → softmax
        self.mu_head = nn.Linear(16, K)          # component means
        self.log_var_head = nn.Linear(16, K)     # component log-variances

    def forward(self, x):
        h = self.backbone(x)                        # h represents hidden layer, it has 32 units, contaning the information
        logits = self.logits_head(h)             # shape: (B, K)
        weights = torch.softmax(logits, dim=-1)  # shape: (B, K)

        mu = self.mu_head(h)                     # shape: (B, K)
        log_var = self.log_var_head(h)           # shape: (B, K)

        return weights, mu, log_var



    
    

# the prior is beta distribution
alpha = 2
beta_para = 5

# number of samples
M = 10**6
#-------------------------------------------------------------------------------------------


# binomial distribution

n_trials = 100

# First sample theta form Beta(2,5)
theta_samples = beta.rvs(alpha, beta_para, size=M)
# Then for each theta, generate the binomial , also 

y_samples = binom.rvs(n=n_trials, p = theta_samples)

# Now you have (θ, x) pairs
# print("First 5 theta values:", theta_samples[:5])
# print("First 5 x values:", y_samples[:5])

theta_np = theta_samples.reshape(-1,1)
y_np = y_samples.reshape(-1,1)


# Combine  theta_np and y_np together
out_mat = np.column_stack([theta_samples, y_samples])
# Create a list of theta

#----------------------------------------------------------------------------------------


theta_dict = defaultdict(list)
sel_index = (out_mat[:, 1] == 1) 
# print(sel_index)
for i in range(n_trials + 1):  # y ∈ [0, n_trials]
    sel_index = (out_mat[:, 1] == i)         # select rows where y == i
    theta_vals = out_mat[sel_index, 0]       # get corresponding θ
    theta_dict[i] = theta_vals               # store as list or array
# print(theta_dict)



# Now calculate the mean and variance of theta list
post_means = {}
post_vars = {}

for i in theta_dict:
    if len(theta_dict[i]) >=2:
        thetas = np.array(theta_dict[i])
        post_means[i] = np.mean(thetas)
        post_vars[i] = np.var(thetas)
    elif len(theta_dict[i]) == 1:
        thetas = np.array(theta_dict[i])
        post_means[i] = np.mean(thetas)
        post_vars[i] = 0
    else:
        post_means[i] = 0
        post_vars[i] = 0
        
# print(post_means)
# print(post_vars)

for key in theta_dict:
    if len(theta_dict[key]) == 0:
        theta_dict[key] = np.array([0.0])



# Define what our optimizer is
model = MixtureDensityNN(K=8)

optimizer = optim.Adam(model.parameters(), lr=0.01)

#=================
def mixture_log_likelihood(theta_vals, weights, mu, log_var):
    """
    theta_vals: (N, 1)
    weights, mu, log_var: (1, K)
    Return: scalar negative log-likelihood
    """
    var = torch.exp(log_var)  # shape: (1, K)
    std = torch.sqrt(var)

    # Expand theta: (N, 1) → (N, K)
    theta = theta_vals.expand(-1, mu.shape[1])  # (N, K)

    # Compute Gaussian log probs: shape (N, K)
    log_probs = -0.5 * (log_var + np.log(2 * np.pi) + (theta - mu)**2 / var)

    # Weighted log sum exp over K components
    log_weighted_probs = torch.log(weights) + log_probs
    log_sum = torch.logsumexp(log_weighted_probs, dim=1)  # shape: (N,)

    return -log_sum.mean()
#=================
for epoch in range(1000):
    model.train()  
    total_loss = 0.0  # Accumulate total loss across all y values

    # Loop over all possible y values (from 0 to n_trials)
    for i in range(n_trials + 1):
        if post_vars[i] == 0:
            continue  # Skip y values with no corresponding theta samples

        # Normalize y value (scale to [0, 1]) and convert to tensor
        x_input = torch.tensor([[i / n_trials]], dtype=torch.float32)

        # Forward pass: predict mixture parameters for current y
        # weights: (1, K) → mixture weights (pi_k), sum to 1
        # mu: (1, K) → means of each Gaussian component
        # log_var: (1, K) → log-variances for numerical stability
        weights, mu, log_var = model(x_input)

        # Get the θ samples for current y, reshape to (N, 1)
        theta_vals = torch.tensor(theta_dict[i], dtype=torch.float32).view(-1, 1)

        # Compute the negative log-likelihood under the mixture model
        # This function handles the log-sum-exp across K components
        loss_i = mixture_log_likelihood(theta_vals, weights, mu, log_var)

        # Accumulate loss over all y values
        total_loss += loss_i

    # Backward pass: compute gradients and update model parameters
    loss = total_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")


y_obs = 90
x_test = torch.tensor([[y_obs / n_trials]], dtype=torch.float32)

with torch.no_grad():
    weights, mu, log_var = model(x_test)  # Mixture of Gaussians output

# Create evaluation grid for θ
theta_range = np.linspace(0.001, 0.999, 300)

# True posterior (Beta)
true_posterior = beta(a=alpha + y_obs, b=beta_para + n_trials - y_obs)
true_pdf = true_posterior.pdf(theta_range)

# Approximate posterior (Mixture of Gaussians)
approx_pdf = np.zeros_like(theta_range)
K = weights.shape[1]  # Number of mixture components

for k in range(K):
    pi_k = weights[0, k].item()
    mu_k = mu[0, k].item()
    std_k = np.sqrt(np.exp(log_var[0, k].item()))
    approx_pdf += pi_k * norm.pdf(theta_range, loc=mu_k, scale=std_k)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Mixture of Gaussians)", lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
