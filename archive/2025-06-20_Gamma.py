from scipy.stats import norm, poisson, beta, binom,gamma,poisson
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random


"""
- We try Gamma distribution using DNN
- We set the parameters alpha <- 1 and beta <- 1 with number of pairs M <- 10^3
- sample thetas from rgamma(1, shape = alpha, rate = beta) and find the y_sum
- put them into matrix form
- Find the thetas for unique y_sum
- Now for each different y_sum , we find the pdf of the likelihood of theta
- Approximate Gamma distribution using Normal distribution of our mu_pred and var_pred

"""


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
        self.log_var_net = nn.Sequential(  # New part
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # Not var anymore, but log_var, more statble

    

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



# Define what our optimizer is
model = ProbabilisticNN()

optimizer = optim.Adam(model.parameters(), lr=0.01)



#===================================================================================================#
# Now our problem becomes how to find the loss function
# Our input would be y_sum , and our output would be still many thetas , the only difference is the true gamma distribution
# 
 


#===================================================================================================#
y_max = np.max(out_mat[:, 1]) # For normalization


for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0

    for y in unique_y_vals:
        if post_vars[y] == 0:
            continue
    
        x_input = torch.tensor([[y / y_max]], dtype=torch.float32)

        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)

        theta_vals = torch.tensor(theta_dict[y], dtype=torch.float32).view(-1, 1)


        log_probs = -0.5 * (np.log(2 * np.pi) + log_var_pred + (theta_vals - mu_pred)**2 / var_pred)


        total_log_likelihood += log_probs.sum()

    loss = -total_log_likelihood

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")



y_obs = 20
x_test = torch.tensor([[y_obs / y_max]], dtype=torch.float32)

with torch.no_grad():
    mu_pred, log_var_pred = model(x_test)
    var_pred = torch.exp(log_var_pred)
    sigma_val = np.sqrt(var_pred.item())


mu_val = mu_pred.item()
sigma_val = np.sqrt(var_pred.item())


true_posterior = gamma(a=alpha + y_obs, scale=1 / (beta_para + n_trials))



theta_range = np.linspace(0.001, 0.999, 300)
true_pdf = true_posterior.pdf(theta_range)
approx_pdf = norm.pdf(theta_range, loc=mu_val, scale=sigma_val)

plt.figure(figsize=(8,5))
plt.plot(theta_range, true_pdf, label="True Posterior (Gamma)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)", lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
