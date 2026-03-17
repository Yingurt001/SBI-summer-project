'''

What we do:
    
prior Beta(2,5) → Binomial data.
First compute empirical posterior means/variances for every y.
Treat the empirical μ and log var as regression targets.
Train a small net (2-output) with MSE loss on (μ, log σ) directly.
Visualize both the μ‐vs‐y curve and the final posterior comparison.

'''

from scipy.stats import norm, poisson, beta, binom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


# Define the deep neural network
class ProbabilisticNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(),
            nn.Linear(24 ,25),
            nn.ReLU(),
            nn.Linear(25, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        out = self.net(x)
        mu = out[:, 0:1]
        log_sigma = out[:, 1:2]  # Predict log(σ) to ensure positivity
        return mu, log_sigma
    
# the prior is beta distribution
alpha = 2
beta_para = 5

# number of samples
M = 10**6

# binomial distribution
success = 80
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
    thetas = np.array(theta_dict[i])
    post_means[i] = np.mean(thetas)
    post_vars[i] = np.var(thetas)
    
# print(post_means)
# print(post_vars)




# Plot Expectatiion versus y
y_vals = list(range(n_trials + 1))  # y = 0 to 100
exp_vals = [post_means[y] for y in y_vals]

plt.plot(y_vals, exp_vals)
plt.xlabel("Observed y")
plt.ylabel("Posterior Variance of θ given y")
plt.title("Posterior Variance of θ vs Observed y")
plt.grid(True)
plt.show()



y_vals = list(range(n_trials + 1))  # y = 0 to 100
var_vals = [post_vars[y] for y in y_vals]

# Plot Var(\t)
plt.plot(y_vals, var_vals)
plt.xlabel("Observed y")
plt.ylabel("Posterior Variance of θ given y")
plt.title("Posterior Variance of θ vs Observed y")
plt.grid(True)
plt.show()



# conver t our post_mean and post variance into list 

x_list = []      # y values (input)
y_list = []      # [mean, log_variance] (output)
# var = post_vars[0]
# sigma = 0.5 * np.log(var)
# print(sigma)
for y_val in sorted(post_means.keys()):
    mu = post_means[y_val]
    var = post_vars[y_val]
    if var>0:
        # var = max(var, 1e-6)
        x_list.append([y_val])
        y_list.append([mu, 0.5 * np.log(var)])


x_np = np.array(x_list, dtype=np.float32)      # shape (N, 1)
y_np = np.array(y_list, dtype=np.float32)      # shape (N, 2)


x = torch.tensor(x_np, dtype=torch.float32)  # shape (N, 1)
y = torch.tensor(y_np, dtype=torch.float32)  # shape (N, 2)

# We use MSE loss
mse = nn.MSELoss()


        
model = ProbabilisticNN()
# Define what our optimizer is
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Extract training targets
y_target = y[:, 0:1]            # true mean
log_var_target = y[:, 1:2]      # true log variance
print(log_var_target)


for epoch in range(1000):  # do 1000 passes over the data
    model.train()  # set model to training mode

    mu_pred, log_sigma_pred = model(x)  # forward pass: compute predictions

    loss_mu = mse(mu_pred, y_target)
    loss_logvar = mse(log_sigma_pred, log_var_target)
    loss = loss_mu + loss_logvar  # compute loss (MSE between prediction and true y)

    optimizer.zero_grad()  # clear previous gradients
    loss.backward()        # backpropagation: compute gradients
    optimizer.step()       # update weights using optimizer

    # Optional: print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
        
        
#  Plot our training results
model.eval()
x_obs = torch.tensor([[50.0]], dtype=torch.float32)
mu_pred, log_sigma_pred = model(x_obs)

mu_val = mu_pred.item()
sigma_val = torch.exp(log_sigma_pred).item()

# print(f"Posterior for y=80 → N(mean={mu_val:.4f}, std={sigma_val:.4f})")



# Draw the true distribution and our appximated distribution and the same time
alpha, beta_para = 2, 5
n = 100
y_obs = 50
true_posterior = beta(a=alpha + y_obs, b=beta_para + n - y_obs)

# Our appximated distribution
x_obs = torch.tensor([[y_obs]], dtype=torch.float32)
mu_pred, log_sigma_pred = model(x_obs)
mu_val = mu_pred.item()
sigma_val = torch.exp(log_sigma_pred).item()


# Plot both
theta_range = np.linspace(0.001, 0.999, 300)

# True posterior density
true_pdf = true_posterior.pdf(theta_range)

# Neural approximation
approx_pdf = norm.pdf(theta_range, loc=mu_val, scale=sigma_val)

# Plot both
plt.figure(figsize=(8,5))
plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
plt.plot(theta_range, approx_pdf, label="NN Approx Posterior (Gaussian)", lw=2, linestyle="--")
plt.title(f"Posterior Comparison for y = {y_obs}")
plt.xlabel("θ")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()