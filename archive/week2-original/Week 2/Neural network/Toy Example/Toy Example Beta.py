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
class ProbabilisticNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4 ,1)
        )
        self.log_var_net = nn.Sequential(  
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # Not var anymore, but log_var, more statble

    
    

# the prior is beta distribution
alpha = 1
beta_para = 1

# number of samples
M = 10**6
#-------------------------------------------------------------------------------------------


# binomial distribution

n_trials = 100

# First sample theta form Beta(alpha,beta)
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
model = ProbabilisticNN()

optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0

    for i in range(n_trials + 1):
        if post_vars[i] == 0:
            continue

        x_input = torch.tensor([[i / n_trials]], dtype=torch.float32)

        mu_pred, log_var_pred = model(x_input)
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






y_obs = 20
x_test = torch.tensor([[y_obs / n_trials]], dtype=torch.float32)

with torch.no_grad():
    mu_pred, log_var_pred = model(x_test)
    var_pred = torch.exp(log_var_pred)
    sigma_val = np.sqrt(var_pred.item())


mu_val = mu_pred.item()
sigma_val = np.sqrt(var_pred.item())


true_posterior = beta(a=alpha + y_obs, b=beta_para + n_trials - y_obs)


theta_range = np.linspace(0.001, 0.999, 300)
true_pdf = true_posterior.pdf(theta_range)
approx_pdf = norm.pdf(theta_range, loc=mu_val, scale=sigma_val)

plt.figure(figsize=(8,5))
plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)", lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
