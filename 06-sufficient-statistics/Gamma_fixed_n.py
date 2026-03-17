"""
Approximates Gamma–Poisson posterior at fixed n via single-Gaussian flow.
  - Simulates θ ∼ Gamma(α,β), y_sum for n = 10
  - Computes empirical posterior mean & var for each y_sum
  - Defines GammaNN [y_sum/y_max] → [μ, logσ²], trains by maximizing Gaussian log-likelihood
  - Evaluates on test y_sum: plots Normal approximation vs true Gamma(α+y,β+n) PDF
"""

import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import norm, poisson, beta, binom,gamma
from mpl_toolkits.mplot3d import Axes3D  # 3D support

# Set seed
SEED = 123
random.seed(SEED)         # Python random
np.random.seed(SEED)      # NumPy random
torch.manual_seed(SEED)   # PyTorch random

# Set prior
alpha = 2
beta_prior = 5

M = 10000  # number of samples per n

df_final = pd.DataFrame()
n=10
# Loop over different n values
for _ in range(1):
    theta_vec = np.full(M, np.nan)
    y_sum_vec = np.full(M, np.nan)

    for i in range(M):  
        theta = np.random.gamma(shape=alpha, scale=1/beta_prior)
        theta_vec[i] = theta
        y_sum_vec[i] = np.sum(np.random.poisson(lam=theta, size=n))

    # Now create n_vec and combine once after filling all M rows

    df = pd.DataFrame({
        'theta': theta_vec,
        'y_sum': y_sum_vec})
    # Combine all the data
    df_final = pd.concat([df_final,df],ignore_index=True)
    
# Sort by y_sum value
    
df_final = df_final.sort_values(by = 'y_sum',ascending = True).reset_index(drop = True)
    
# Find the unique pairs of y_sum and n
unique_pairs = df_final[['y_sum']].drop_duplicates().reset_index(drop=True)
    
thetas = []

for _, row in unique_pairs.iterrows():
    y_val = row['y_sum']
    matching_theta = df_final[df_final['y_sum'] == y_val]['theta'].values  # only extract theta column

    thetas.append(matching_theta)
    

post_means = []
post_vars = []  

for theta_va in thetas:
    post_means.append(np.mean(theta_va))
    post_vars.append(np.var(theta_va))
    
# Check value of post_var whether zero or nan
unique_var =   np.unique(post_vars)
nan_zero = np.where((np.isnan(post_vars)) | (post_vars == 0))[0]


# Now give the scatter plot
plt.scatter(unique_pairs['y_sum'],post_means )
plt.xlabel('Unique y_sum')
plt.ylabel('Post mean')
plt.title('Scatter Plot Example')
# Show the plot
plt.show()






# Now give the scatter plot
plt.scatter(unique_pairs['y_sum'],post_vars )
plt.xlabel('Unique y_sum')
plt.ylabel('Post var')
plt.title('Scatter Plot Example')
# Show the plot
plt.show()


### Now construct the neural networks with  input y_sum and n, output the corresponding mean and variance ,
# Our loss function will be simply Gaussian N(mu, sigma^2)


    

class GammaNN(nn.Module):
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
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4 ,1)
        )

    def forward(self, x):
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        return mu, log_var  # Not var anymore, but log_var, more statble

'''
for key in thetas:
    if len(thetas[key]) == 0:
        thetas[key] = np.array([0.0])
'''       
        
        
## Start model training

model = GammaNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

y_max = df_final['y_sum'].max()

theta_dict = {}

for y_val in df_final['y_sum'].unique():  # 
    theta_dict[y_val] = df_final[df_final['y_sum'] == y_val]['theta'].values
    
    
for epoch in range(1000):
    model.train()
    total_log_likelihood = 0.0

    for y_val in unique_pairs['y_sum']:


        # Normalize input
        x_input = torch.tensor([[y_val / y_max]], dtype=torch.float32)

        # Forward pass
        mu_pred, log_var_pred = model(x_input)
        var_pred = torch.exp(log_var_pred)
        
        theta_vals = torch.tensor(theta_dict[y_val], dtype=torch.float32).view(-1, 1)

        # Gaussian log-likelihood
        log_probs = -0.5 * (np.log(2 * np.pi) + log_var_pred + (theta_vals - mu_pred)**2 / var_pred)
        total_log_likelihood += log_probs.sum()

    # Loss = negative log-likelihood
    loss = -total_log_likelihood

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")


# Assume model is already trained
model.eval()


y_test_list = [50]
n_test_list = [10]  # Still used to compute true posterior

# Normalize input and create test batch
x_test_batch = torch.tensor([
    [y / y_max] for y in y_test_list
], dtype=torch.float32)

# Run through the model
with torch.no_grad():
    mu_pred, log_var_pred = model(x_test_batch)
    mu_pred = mu_pred.view(-1).numpy()
    var_pred = torch.exp(log_var_pred).view(-1).numpy()

# ------------------------------
# Plot posterior comparison
# ------------------------------
for i, (y_val, n_val) in enumerate(zip(y_test_list, n_test_list)):
    # Compute true posterior parameters
    alpha_post = alpha + y_val
    beta_post = beta_prior + n_val

    theta_vals = np.linspace(0.01, 5, 500)  # Wider range

    # True posterior: Gamma
    true_pdf = gamma.pdf(theta_vals, a=alpha_post, scale=1 / beta_post)

    # Approximate posterior: Normal from NN
    approx_pdf = norm.pdf(theta_vals, loc=mu_pred[i], scale=np.sqrt(var_pred[i]))

    # Plot
    plt.figure()
    plt.plot(theta_vals, true_pdf, label='True Gamma Posterior', lw=2)
    plt.plot(theta_vals, approx_pdf, label='NN Approx Normal', lw=2, linestyle='--')
    plt.title(f"Posterior θ | y_sum={y_val}, n={n_val}")
    plt.xlabel("θ")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()