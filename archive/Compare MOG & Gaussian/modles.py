# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, beta,gamma

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
        mu = self.mean_net(x)  # no constraint, output unrestricted
        log_var = self.log_var_net(x)
        return mu, log_var

class MixtureDensityNN(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.K = K
        self.backbone = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.logits_head = nn.Linear(32, K)
        self.mu_head = nn.Linear(32, K)  # no activation constraint
        self.log_var_head = nn.Linear(32, K)

    def forward(self, x):
        h = self.backbone(x)
        weights = torch.softmax(self.logits_head(h), dim=-1)
        mu = self.mu_head(h)  # unrestricted
        log_var = self.log_var_head(h)
        return weights, mu, log_var

def get_model(model_type='mog', K=3):
    if model_type == 'mog':
        return MixtureDensityNN(K=K)
    elif model_type == 'normal':
        return ProbabilisticNN()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# ---------- Training and Testing Utilities ----------

def mixture_log_likelihood(theta_vals, weights, mu, log_var):
    var = torch.exp(log_var)
    theta = theta_vals.expand(-1, mu.shape[1])  # (N, K)
    log_probs = -0.5 * (log_var + torch.log(torch.tensor(2 * torch.pi)) + (theta - mu) ** 2 / var)
    weighted_log_probs = torch.log(weights) + log_probs
    log_likelihoods = torch.logsumexp(weighted_log_probs, dim=1)
    return -log_likelihoods.mean()

def train_model(model, optimizer, theta_dict, post_vars, model_type='mog', n_trials=100, epochs=1000):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for y in range(n_trials + 1):
            if post_vars[y] == 0:
                continue
            x_input = torch.tensor([[y / n_trials]], dtype=torch.float32)
            theta_vals = torch.tensor(theta_dict[y], dtype=torch.float32).view(-1, 1)

            if model_type == 'mog':
                weights, mu, log_var = model(x_input)
                loss_i = mixture_log_likelihood(theta_vals, weights, mu, log_var)
            elif model_type == 'normal':
                mu, log_var = model(x_input)
                var = torch.exp(log_var)
                log_probs = -0.5 * (torch.log(torch.tensor(2 * torch.pi)) + log_var + (theta_vals - mu)**2 / var)
                loss_i = -log_probs.mean()
            else:
                raise ValueError("Invalid model_type")

            total_loss += loss_i

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.5f}")

def test_model(model, y_obs, model_type='mog', n_trials=100):
    model.eval()
    x_test = torch.tensor([[y_obs / n_trials]], dtype=torch.float32)
    with torch.no_grad():
        if model_type == 'mog':
            weights, mu, log_var = model(x_test)
            return weights[0], mu[0], log_var[0]  # Return 1D tensors
        elif model_type == 'normal':
            mu, log_var = model(x_test)
            return mu[0, 0].item(), log_var[0, 0].item()
        else:
            raise ValueError("Invalid model_type")

# ---------- Plot Posterior Comparison for Multiple y ----------
def plot_posterior_comparisons(model_mog, model_normal, alpha, beta_para, n_trials, y_list):
    theta_range = np.linspace(0.001, 0.999, 300)
    for y_obs in y_list:
        # True posterior
        true_posterior = beta(a=alpha + y_obs, b=beta_para + n_trials - y_obs)
        true_pdf = true_posterior.pdf(theta_range)

        # MoG output
        weights, mu_mog, log_var_mog = test_model(model_mog, y_obs, model_type='mog', n_trials=n_trials)
        approx_pdf_mog = np.zeros_like(theta_range)
        K = weights.shape[0]
        for k in range(K):
            pi_k = weights[k].item()
            mu_k = mu_mog[k].item()
            std_k = np.sqrt(np.exp(log_var_mog[k].item()))
            approx_pdf_mog += pi_k * norm.pdf(theta_range, loc=mu_k, scale=std_k)

        # Normal output
        mu_normal, log_var_normal = test_model(model_normal, y_obs, model_type='normal', n_trials=n_trials)
        std_val = np.sqrt(np.exp(log_var_normal))
        approx_pdf_normal = norm.pdf(theta_range, loc=mu_normal, scale=std_val)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(theta_range, true_pdf, label="True Posterior (Beta)", lw=2)
        plt.plot(theta_range, approx_pdf_mog, label="NN Posterior (Mixture of Gaussians)", lw=2, linestyle="--")
        plt.plot(theta_range, approx_pdf_normal, label="NN Posterior (Single Gaussian)", lw=2, linestyle=":")
        plt.title(f"Posterior Comparison for y = {y_obs}")
        plt.xlabel("θ")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
        
        
def plot_gamma_posterior_comparisons(model_mog, model_normal, alpha, beta_para, n_trials, y_list):
    theta_range = np.linspace(0.001, 0.999, 300)
    for y_obs in y_list:
        # Gamma posterior
        true_posterior = gamma(a=alpha + y_obs, scale=1 / (beta_para + n_trials))
        true_pdf = true_posterior.pdf(theta_range)

        # MoG model output
        weights, mu_mog, log_var_mog = test_model(model_mog, y_obs, model_type='mog', n_trials=n_trials)
        approx_pdf_mog = np.zeros_like(theta_range)
        K = weights.shape[0]
        for k in range(K):
            pi_k = weights[k].item()
            mu_k = mu_mog[k].item()
            std_k = np.sqrt(np.exp(log_var_mog[k].item()))
            approx_pdf_mog += pi_k * norm.pdf(theta_range, loc=mu_k, scale=std_k)

        # Normal model output
        mu_normal, log_var_normal = test_model(model_normal, y_obs, model_type='normal', n_trials=n_trials)
        std_val = np.sqrt(np.exp(log_var_normal))
        approx_pdf_normal = norm.pdf(theta_range, loc=mu_normal, scale=std_val)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(theta_range, true_pdf, label="True Posterior (Gamma)", lw=2)
        plt.plot(theta_range, approx_pdf_mog, label="NN Posterior (Mixture of Gaussians)", lw=2, linestyle="--")
        plt.plot(theta_range, approx_pdf_normal, label="NN Posterior (Single Gaussian)", lw=2, linestyle=':')
        plt.title(f"Posterior Comparison for y = {y_obs}")
        plt.xlabel("θ")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()
