import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import scipy.stats as stats

# 1. True parameters and data generation
alpha_true = 2.0
beta_true = 1.0
n = 100

# Generate n IID gamma observations (shape=alpha, rate=beta)
y_np = np.random.gamma(shape=alpha_true, scale=1.0/beta_true, size=n)
y = torch.tensor(y_np, dtype=torch.float32)

# Compute summary statistics: sum log y and sum y
sum_log_y = torch.sum(torch.log(y)).item()
sum_y     = torch.sum(y).item()
y_bar_log = sum_log_y / n
y_bar     = sum_y / n
summary = torch.tensor([y_bar_log, y_bar], dtype=torch.float32)

# 2. Prior hyperparameters (Exponential with rate=1)
lambda_alpha = 1.0
lambda_beta = 1.0

# 3. Variational flow model for (alpha, beta)
class AffineFlow2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 4)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, summary, z):
        if summary.dim() == 1:
            summary = summary.unsqueeze(0)
        log_s1_raw, b1_raw, log_s2_raw, b2_raw = self.linear(summary)[0]
        log_s1 = torch.clamp(log_s1_raw, -5, 5)
        log_s2 = torch.clamp(log_s2_raw, -5, 5)
        s1 = torch.exp(log_s1)
        s2 = torch.exp(log_s2)
        b1 = torch.clamp(b1_raw, -10, 10)
        b2 = torch.clamp(b2_raw, -10, 10)
        u1 = torch.clamp(s1 * z[:,0] + b1, -20, 20)
        u2 = torch.clamp(s2 * z[:,1] + b2, -20, 20)
        alpha = torch.exp(u1)
        beta  = torch.exp(u2)
        log_base = Normal(0,1).log_prob(z).sum(dim=1)
        log_jac  = (u1 + u2) - (log_s1 + log_s2)
        log_q    = log_base + log_jac
        return alpha, beta, log_q

flow = AffineFlow2D()
optimizer = optim.Adam(flow.parameters(), lr=1e-3)
base_dist = Normal(torch.zeros(2), torch.ones(2))

# 4. ELBO optimization
num_samples = 500
epochs = 2000
for epoch in range(1, epochs := epochs if 'epochs' in globals() else 2000 + 1):
    z = base_dist.sample((num_samples,))
    alpha_q, beta_q, log_q = flow(summary, z)
    log_prior = -lambda_alpha * alpha_q - lambda_beta * beta_q
    log_lik   = (n * alpha_q * torch.log(beta_q.clamp(min=1e-8))
                 - n * torch.lgamma(alpha_q)
                 + (alpha_q - 1) * sum_log_y
                 - beta_q * sum_y)
    elbo = torch.mean(log_prior + log_lik - log_q)
    loss = -elbo
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.3f} | ELBO = {elbo.item():.3f}")

# 5. Estimate parameters via variational means
with torch.no_grad():
    z = base_dist.sample((2000,))
    alpha_samp, beta_samp, _ = flow(summary, z)
alpha_est = alpha_samp.mean().item()
beta_est  = beta_samp.mean().item()
print(f"Estimated parameters: alpha={alpha_est:.3f}, beta={beta_est:.3f}")

# 6. Plot true vs. estimated Gamma PDFs
x = np.linspace(0, np.max(y_np) * 1.2, 200)
true_pdf = stats.gamma.pdf(x, a=alpha_true, scale=1.0/beta_true)
est_pdf  = stats.gamma.pdf(x, a=alpha_est,  scale=1.0/beta_est)

plt.figure(figsize=(6,4))
plt.hist(y_np, bins=30, density=True, alpha=0.3, label='Data histogram')
plt.plot(x, true_pdf, label='True Gamma PDF', linewidth=2)
plt.plot(x, est_pdf,  label='Estimated Gamma PDF', linestyle='--', linewidth=2)
plt.xlabel('y')
plt.ylabel('Density')
plt.legend()
plt.title('True vs Estimated Gamma Distribution')
plt.show()
