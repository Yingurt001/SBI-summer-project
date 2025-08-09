rm(list=ls())

library(mclust)

# Code for simulating a household epidemic.
#
# 1 - infective; n-1 - susceptible
# Infectious periods Gamma(k,k) - k=0 implies constant infectious period =1.
# Infection rate lamdba_L


# function simulate final size data from an SIR model
# 1 initial infection, N-1 susceptible: total population size = N
# Gamma(k,k) infectious period -- if k = 0 is constant = 1 infectious period
SIR_sim=function(N,lambda,k)
{
  S=N-1
  y=1
  while(y>0)
  {
    y=y-1
    if(k==0) I=1
    if(k>0) I=rgamma(1,k,k)
    Z=rpois(1,(lambda*I))
    if(Z>0)
    {
      for(j in 1:Z)
      {
        u=runif(1)
        if(u<(S/N))
        {
          S=S-1
          y=y+1
        }
      }
    }
  }
  N-S-1
}

# this is a function to compute the final size
# m initial susceptibles
# a initial infectives
# beta infection rate
# c the length of the fixed infectious period
#compute.conditional.prob <- function(m, j, a, beta, c = 1){
#   if (j==0) {
#     res <- (exp(-beta*m*c))^a
#   }
#   else {
#     
#     part.one <- exp(-beta*(m-j)*c)^(j+a)
#     part.two <- choose(m, j)
#     
#     sum <- 0;
#     for (k in 0:(j-1)) {      
#       sum <- sum + choose(m-k, j-k)*compute.conditional.prob(m,k,a,beta,c)/(exp(-beta*(m-j)*c)^(k+a))
#     }
#     res <- part.one*(part.two - sum)
#   }
#   return(res)
# }
# 
# lik <- function(theta, y, N) {
#   compute.conditional.prob(N-1, y, a = 1, beta = theta, c = 1)
# }
#   
# loglik <- function(theta, y, N) {
#   compute.conditional.prob(N-1, y, a = 1, beta = theta, c = 1)
# }


N <- 100
true.theta <- 2.5

K <- 10^5
final.size.vec <- rep(NA, K) 
for (k in 1:K) {
  final.size.vec[k] <- SIR_sim(N, true.theta, 1)
}

hist(final.size.vec, prob=TRUE)

fs.distn <- as.vector(table(final.size.vec)/K)
barplot(fs.distn, names.arg = 0:(N-1))
fs.distn

final.size.distn <- sir_final_size_full_safe(N, true.theta/N, gamma = 1)
final.size.distn[-1]

plot(fs.distnfinal.size.distn[-1]); abline(0,1, col=2)

lik(true.theta, 90, N)

plot(seq(0, 7, len = 200), sapply(seq(0, 7, len = 200), function(x) lik(x, 60, N)))
abline(v=true.theta)

sir_final_size_full_safe(N, true.theta/N, gamma = 1)[1:10]

M <- 10^4
N <- 100

out <- matrix(NA, M, 2)
for (m in 1:M) {
  theta <- rexp(1, 0.01)
  y <- SIR_sim(N, theta, 0)
  out[m, ] <- c(theta, y)
}

write.table(out, file="theta_y_samples_Exp_0.001.txt", row.names = FALSE, col.names = FALSE,  quote = FALSE)

plot(out[,1], out[,2], cex = 0.8, xlab = "theta", ylab = "y")
abline(h=80, col=2, lwd=2)


y.obs <- 80
lambda.given.y.obs <- out[out[,2]==y.obs,1]
hist(lambda.given.y.obs, prob=TRUE, main = "pi(lambda | y_obs)", xlab = "lambda")



# read the samples
post.samples <- read.table("theta_y_samples_Exp_1.txt", header=FALSE)
dim(post.samples)

post.samples <- post.samples[sample(1:nrow(post.samples), 10000),]

theta <- post.samples[,1]
y <- post.samples[,2]

par(mfrow=c(1,1))
plot(theta, y, pch = 16, cex = 0.6, xlab=expression(beta), ylab=expression(y), cex.axis=1.3, cex.lab=1.5)

par(mfrow=c(1,1), mar = c(4, 4, 1, 1), mgp=c(2.5,1,0), bg=NA)   
par(mfrow=c(1,1), mar = c(4, 4, 1, 1), mgp=c(2.5,1,0))   
plot(theta, y, pch = 16, cex = 0.5, xlab=expression(beta), ylab=expression(y), cex.axis=2, cex.lab=2)
y.obs <- 80
abline(h=y.obs, col=2, lty=2, lwd=2)
text(6, 83, "observed final size",col=2, cex=2)
dev.copy(png,'beta-y-pairs.png')
dev.off()

y.obs <- 80
abline(h=y.obs, col=2, lwd=1)

theta.samples <- theta[y==y.obs]
par(mfrow=c(1,1), mar = c(4, 4, 1, 1), mgp=c(2.5,0.5,0),bg=NA)   
par(mfrow=c(1,1), mar = c(4, 5, 1, 1), mgp=c(3.0,0.5,0))
hist(theta.samples, cex.axis=2, cex.lab=2, cex.main = 2, prob = TRUE, main = "Neural Posterior", 
     xlab = expression(beta), ylim=c(0, 1.7), ylab=expression(paste(pi,"(", beta, "|", y, " )")))
# fit a Normal
theta.values <- seq(min(theta.samples), max(theta.samples), len = 200)
pdf.values <- sapply(theta.values, function(x) dnorm(x, mean(theta.samples), sd(theta.samples)))
lines(theta.values, pdf.values, col = 2, lwd=2)
legend(c(1.2), lwd=2, col=2, expression(paste(q[phi],"(", theta, "|", y, " )")), cex=1.7)
dev.copy(png,'neural-posterior-exp1.png')
dev.off()


theo <- densityMclust(theta.samples)
summary(theo)
plot(theo, what = "density")
theo$parameters

lines(density(theta.samples), col=2)


theta.mean <- NULL
theta.sd <- NULL
out <- post.samples

for (j in 1:100) {
  
  theta.mean[j] <- mean(out[out[,2]==j,1])
  theta.sd[j] <- sd(out[out[,2]==j,1])
  
}
plot(1:100, theta.mean)
plot(1:100, theta.sd)
