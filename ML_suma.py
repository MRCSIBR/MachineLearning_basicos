# Machine Learning to Sum Two Numbers
# This script demonstrates how to use a neural network to learn the summation of two numbers.
# We'll use PyTorch to build and train the model, generate synthetic data, and visualize the results.

# Import Libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Generate Synthetic Data
# Create pairs of numbers (a, b) and their sums (a + b) with slight noise.
# Generate 1000 pairs of numbers between -10 and 10.
n_samples = 1000
X = np.random.uniform(-10, 10, (n_samples, 2))  # Two numbers per sample
y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples)  # Sum with noise

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Visualize a subset of the data
plt.figure(figsize=(6, 4))
plt.scatter(X[:100, 0], X[:100, 1], c=y[:100], cmap='viridis')
plt.colorbar(label='Sum (a + b)')
plt.xlabel('Number a')
plt.ylabel('Number b')
plt.title('Sample Data: Two Numbers and Their Sum')
plt.savefig('data_visualization.png')
plt.close()

# Step 2: Define the Neural Network
# A simple feedforward neural network with one hidden layer:
# - Input layer: 2 neurons (for the two numbers)
# - Hidden layer: 10 neurons with ReLU activation
# - Output layer: 1 neuron (for the sum)
class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SumNet()

# Step 3: Set Up Loss and Optimizer
# - Loss: Mean Squared Error (MSE) for regression
# - Optimizer: Stochastic Gradient Descent (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the Model
# Train for 1000 epochs, printing loss every 100 epochs
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Clear previous gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Evaluate and Visualize Results
# Compute predictions and plot true vs. predicted sums
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()
y_true = y_tensor.numpy()

# Plot true vs predicted sums
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
plt.plot([-20, 20], [-20, 20], 'r--', label='Perfect prediction (y=x)')
plt.xlabel('True Sum')
plt.ylabel('Predicted Sum')
plt.title('True vs Predicted Sums')
plt.legend()
plt.grid(True)
plt.savefig('sum_predictions.png')
plt.close()

# Test a few examples
test_inputs = torch.FloatTensor([[1, 2], [5, -3], [-4, 7]])
with torch.no_grad():
    test_preds = model(test_inputs).numpy()
for i, (a, b) in enumerate(test_inputs):
    print(f'Input: {a:.1f} + {b:.1f}, Predicted Sum: {test_preds[i][0]:.2f}, True Sum: {a+b:.2f}')

# Key Takeaways:
# - Neural networks can learn simple functions like addition.
# - Data preparation: Convert inputs/outputs to tensors.
# - Model: A small network with one hidden layer is sufficient.
# - Loss/Optimization: MSE and SGD are standard for regression.
# - Visualization: Plotting true vs. predicted values assesses performance.
