import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(),
            nn.Linear(24 ,6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
    def forward(self, x):
        return self.net(x)

x_np = np.linspace(-10,10,1000).reshape(-1,1)
y_np = (np.exp(x_np)+x_np**2).reshape(-1,1)

# plt.plot(x_np,y_np)
# plt.show()

model = DeepNN()
# Define what our optimizer is
optimizer = optim.Adam(model.parameters(), lr=0.01)
# define our loss function
loss_fn = nn.MSELoss()

x = torch.tensor(x_np, dtype=torch.float32) # Of shape (100,1)
y = torch.tensor(y_np, dtype=torch.float32) # Of shape (100,1)


for epoch in range(1000):  # do 1000 passes over the data
    model.train()  # set model to training mode

    y_pred = model(x)  # forward pass: compute predictions

    loss = loss_fn(y_pred, y)  # compute loss (MSE between prediction and true y)

    optimizer.zero_grad()  # clear previous gradients
    loss.backward()        # backpropagation: compute gradients
    optimizer.step()       # update weights using optimizer

    # Optional: print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        
#  Plot our training results
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-20, 20, 1000).reshape(-1, 1)
    y_test = model(x_test)
        
        
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="Noisy Data")
plt.plot(x_test, y_test.numpy(), color='red', label="Neural Net Prediction")
plt.legend()
plt.title("Neural Network Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

