import numpy as np
import matplotlib.pyplot as plt

# --- 1. 定义基础分布 Pr(z) ---
def gauss_pdf(z, mu, sigma):
    return np.exp(-0.5 * (z - mu)**2 / sigma**2) / (np.sqrt(2 * np.pi) * sigma)

z = np.arange(-3, 3, 0.01)
pr_z = gauss_pdf(z, 0, 1)

fig, ax = plt.subplots()
ax.plot(z, pr_z)
ax.set_xlim([-3, 3])
ax.set_xlabel('z (Latent variable)')
ax.set_ylabel('Pr(z)')
ax.set_title('Base Density')
plt.show()

# --- 2. 定义非线性可逆函数 x = f(z) ---
def f(z):
    x1 = 6 / (1 + np.exp(-(z - 0.25) * 1.5)) - 3
    x2 = z
    p = z**2 / 9
    x = (1 - p) * x1 + p * x2
    return x

# --- 3. 数值求导 df/dz ---
def df_dz(z):
    eps = 1e-4
    return (f(z + eps) - f(z - eps)) / (2 * eps)

x = f(z)

fig, ax = plt.subplots()
ax.plot(z, x)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_xlabel('z (Latent variable)')
ax.set_ylabel('x (Observed variable)')
ax.set_title('Transformation Function x = f(z)')
plt.show()

# --- 4. 使用公式 Pr(x) = Pr(z) / |df/dz| 计算 x 空间的密度 ---
df = df_dz(z)
pr_x = pr_z / np.abs(df)

fig, ax = plt.subplots()
ax.plot(x, pr_x)
ax.set_xlim([-3, 3])
ax.set_ylim([0, 0.5])
ax.set_xlabel('x (Observed variable)')
ax.set_ylabel('Pr(x)')
ax.set_title('Transformed Density Pr(x)')
plt.show()

# --- 5. 从 z 分布中采样并变换为 x ---
np.random.seed(1)
n_sample = 20
z_samples = np.random.normal(0, 1, size=(n_sample, 1))
x_samples = f(z_samples)

fig, ax = plt.subplots()
ax.plot(x, pr_x)
for x_sample in x_samples:
    ax.plot([x_sample, x_sample], [0, 0.1], 'r-')
ax.set_xlim([-3, 3])
ax.set_ylim([0, 0.5])
ax.set_xlabel('x (Observed variable)')
ax.set_ylabel('Pr(x)')
ax.set_title('Samples from Pr(x)')
plt.show()
