# %% Imports and config
from scipy.stats import norm, poisson, beta, binom,gamma
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random
from modles import get_model,test_model,train_model,mixture_log_likelihood,plot_posterior_comparisons,plot_gamma_posterior_comparisons


# %% Data generation
# Set seed value
SEED = 123
random.seed(SEED)         # Python random
np.random.seed(SEED)      # NumPy random
torch.manual_seed(SEED)   # PyTorch random

# For reproducibility (optional but recommended)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# the prior is Gamma distribution
alpha = 2
beta_para = 3

# number of samples
M = 10**6
#-------------------------------------------------------------------------------------------


# Gamma distribution

n_trials = 100

# First sample theta form Beta(2,5)
theta_samples = gamma.rvs(a = alpha, loc=0, scale = 1/beta_para, size=M)


# Broadcasting over theta_samples
y_samples = np.sum(poisson.rvs(mu=theta_samples[:, None], size=(len(theta_samples), n_trials)), axis=1)




# Now we have (θ, y_sum pairs
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
# Find the unique y_vals
unique_y_vals = np.unique(out_mat[:, 1])
# print(sel_index)
for y in unique_y_vals:
    sel_index = (out_mat[:, 1] == y)
    theta_vals = out_mat[sel_index, 0]
    theta_dict[y] = theta_vals



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

# %% Train MoG model
# Define what our optimizer is
model_1 = get_model(model_type='mog', K=8)

optimizer = optim.Adam(model_1.parameters(), lr=0.01)


for epoch in range(1000):
    model_1.train()  
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
        weights, mu, log_var = model_1(x_input)

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
# %% Train Normal model

model_2 = get_model(model_type='normal', K=3)

optimizer = optim.Adam(model_2.parameters(), lr=0.01)


for epoch in range(1000):
    model_2.train()
    total_log_likelihood = 0.0

    for i in range(n_trials + 1):
        if post_vars[i] == 0:
            continue

        x_input = torch.tensor([[i / n_trials]], dtype=torch.float32)

        mu_pred, log_var_pred = model_2(x_input)
        var_pred = torch.exp(log_var_pred)

        theta_vals = torch.tensor(theta_dict[i], dtype=torch.float32).view(-1, 1)

        log_probs = -0.5 * (torch.log((torch.tensor(2 * torch.pi)) )
                            + log_var_pred 
                            + (theta_vals - mu_pred) 
                            ** 2 / var_pred)

        total_log_likelihood += log_probs.sum()

    loss = -total_log_likelihood

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
# %% Test Results
y_obs = 70

# MoG output
weights, mu_mog, log_var_mog = test_model(model_1, y_obs, model_type='mog', n_trials=n_trials)

# Normal output
mu_normal, log_var_normal = test_model(model_2, y_obs, model_type='normal', n_trials=n_trials)




# %% Visualization and comparison
plot_gamma_posterior_comparisons(
    model_mog=model_1,
    model_normal=model_2,
    alpha=alpha,
    beta_para=beta_para,
    n_trials=n_trials,
    y_list=[0, 20,30, 40,50, 60,70, 80,90, 100]
)

