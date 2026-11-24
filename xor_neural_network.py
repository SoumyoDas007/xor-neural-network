import os
import numpy as np
import matplotlib.pyplot as plt


# XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(42)

# 2 inputs -> 2 hidden neurons -> 1 output neuron
w1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.zeros((1, 2))
w2 = np.random.uniform(-1, 1, (2, 1))
b2 = np.zeros((1, 1))

losses = []

# Train the network
for epoch in range(10000):
    hidden = sigmoid(X.dot(w1) + b1)
    output = sigmoid(hidden.dot(w2) + b2)

    # Calculate errors
    output_error = y - output
    output_delta = output_error * output * (1 - output)

    hidden_error = output_delta.dot(w2.T)
    hidden_delta = hidden_error * hidden * (1 - hidden)

    # Update weights and biases
    w2 += hidden.T.dot(output_delta) * 0.5
    b2 += np.sum(output_delta, axis=0, keepdims=True) * 0.5
    w1 += X.T.dot(hidden_delta) * 0.5
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * 0.5

    loss = -np.mean(y * np.log(output + 1e-12) +
                    (1 - y) * np.log(1 - output + 1e-12))
    losses.append(loss)


# Test the network
hidden = sigmoid(X.dot(w1) + b1)
output = sigmoid(hidden.dot(w2) + b2)
predictions = (output >= 0.5).astype(int)

print("XOR Neural Network Test Results:")
for inputs, result, prediction in zip(X, output, predictions):
    print(f"Input: {inputs} -> Probability: {result[0]:.4f} -> Output: {prediction[0]}")

accuracy = np.mean(predictions == y) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Final loss: {losses[-1]:.6f}")


# Save the model
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(folder, exist_ok=True)
np.savez(os.path.join(folder, "xor_model.npz"),
         w1=w1, b1=b1, w2=w2, b2=b2)
print(f"Model saved to: {os.path.join(folder, 'xor_model.npz')}")


# Plot loss and decision boundary
x1 = np.linspace(-0.1, 1.1, 300)
x2 = np.linspace(-0.1, 1.1, 300)
grid_x1, grid_x2 = np.meshgrid(x1, x2)
grid = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
grid_hidden = sigmoid(grid.dot(w1) + b1)
grid_output = sigmoid(grid_hidden.dot(w2) + b2).reshape(grid_x1.shape)

figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(losses)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

image = axes[1].contourf(grid_x1, grid_x2, grid_output,
                         levels=20, cmap="RdBu_r", alpha=0.75)
axes[1].contour(grid_x1, grid_x2, grid_output,
                levels=[0.5], colors="black", linewidths=2)
axes[1].scatter(X[:, 0], X[:, 1], c=y.ravel(),
                cmap="RdBu_r", edgecolors="black", s=120)
axes[1].set_title("XOR Decision Boundary")
figure.colorbar(image, ax=axes[1], label="Probability")

plot_path = os.path.join(folder, "fig_5_1_output_plots.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {plot_path}")
