rm(list=ls())

set.seed(1)
library(ggplot2)

# Observed data
y_obs <- 1.2
lik_sd <- 0.5
prior_sd <- 1

# True posterior
posterior_var <- 1 / (1 / prior_sd^2 + 1 / lik_sd^2)
posterior_sd <- sqrt(posterior_var)
posterior_mean <- posterior_var * (y_obs / lik_sd^2)

# Base samples from standard normal
n_samples <- 1000
z <- rnorm(n_samples)  # z ~ N(0,1)

# Affine flow: theta = scale * z + shift
affine_flow <- function(z, scale, shift) {
  scale * z + shift
}

# Log of unnormalised posterior: log p(y_obs | theta) + log p(theta)
log_posterior <- function(theta, y_obs, lik_sd) {
  dnorm(y_obs, mean = theta, sd = lik_sd, log = TRUE) +
    dnorm(theta, mean = 0, sd = 1, log = TRUE)
}

# ELBO = E_q[log p(theta, y_obs) - log q(theta)]
# q(theta) is the flow-transformed distribution

neg_elbo <- function(params) {
  log_scale <- params[1]
  shift <- params[2]
  scale <- exp(log_scale)
  
  theta <- affine_flow(z, scale, shift)
  log_q <- dnorm(z, log = TRUE) - log(scale)  # change-of-variable formula
  log_post <- log_posterior(theta, y_obs, lik_sd)
  
  -mean(log_post - log_q)
}

# Optimise parameters
init <- c(log(1), 0)  # log(scale), shift
fit <- optim(init, neg_elbo, method = "BFGS")
opt_scale <- exp(fit$par[1])
opt_shift <- fit$par[2]

cat("Learned scale:", round(opt_scale, 3), "\n")
cat("Learned shift:", round(opt_shift, 3), "\n")
cat("True posterior sd:", round(posterior_sd, 3), "\n")
cat("True posterior mean:", round(posterior_mean, 3), "\n")

# Apply learned flow
theta_samples <- affine_flow(z, opt_scale, opt_shift)

# Plot results
df <- data.frame(theta = theta_samples)
ggplot(df, aes(x = theta)) +
  geom_histogram(aes(y = ..density..), bins = 40, fill = "skyblue", alpha = 0.6) +
  stat_function(fun = function(x) dnorm(x, mean = posterior_mean, sd = posterior_sd),
                color = "red", size = 1.2) +
  labs(title = "Learned Normalising Flow Posterior vs True Posterior",
       x = expression(theta), y = "Density") +
  theme_minimal()
