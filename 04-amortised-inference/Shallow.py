import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Fit data of y = sin(x)

# generate the data 
x_np = np.linspace(-10,10,1000).reshape(-1,1)

y_np = x_np**2
# plt.plot(x_np,y_np)
# plt.show()


# Define neural network shallow
class ShallowNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 20)   # 1 input → 10 hidden
        self.out = nn.Linear(20, 1)      # 10 hidden → 1 output
        self.act = nn.Tanh()             # activation function

    def forward(self, x):
        x = self.act(self.hidden(x))     # hidden layer + activation
        return self.out(x)               # output layer

# set up training
# name our model
model = ShallowNN()
# Define what our optimizer is
optimizer = optim.Adam(model.parameters(), lr=0.01)
# define our loss function
loss_fn = nn.MSELoss()

# Now x_np and y_np are all Numpy array
# We need to convert them into pytorch tensor
# Convert data to PyTorch tensors
x = torch.tensor(x_np, dtype=torch.float32) # Of shape (100,1)
y = torch.tensor(y_np, dtype=torch.float32) # Of shape (100,1)

# Check type
print(type(x_np))  # <class 'numpy.ndarray'>
print(type(x))     # <class 'torch.Tensor'>
print(x_np.shape)
print(x.shape)
print(y.shape)
print(x.dtype)     # torch.float32

# Train Loop

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
    x_test = torch.linspace(-10, 10, 1000).reshape(-1, 1)
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