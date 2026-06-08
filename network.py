import numpy as np
from layer import Layer


class Network:
    def __init__(self, layer_sizes):
        """
        Build a fully-connected network from a list of sizes.

        Example: Network([2, 4, 1])
          Input layer:  2 neurons (your features)
          Hidden layer: 4 neurons (learns patterns)
          Output layer: 1 neuron  (your prediction)

        We create (len(layer_sizes) - 1) Layer objects.
        Each layer connects adjacent sizes.
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(n_inputs=layer_sizes[i], n_neurons=layer_sizes[i + 1])
            )

    # ── Forward pass ──────────────────────────────────────────

    def forward(self, x):
        """Feed x through every layer in sequence."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ── Loss functions ────────────────────────────────────────

    def mse_loss(self, prediction, target):
        """Mean Squared Error: average of (pred - true)^2"""
        return np.mean((prediction - target) ** 2)

    def mse_loss_derivative(self, prediction, target):
        """
        Gradient of MSE w.r.t. prediction.
        d/dpred [ (pred - target)^2 ] = 2 * (pred - target)
        This is where backprop starts.
        """
        return 2 * (prediction - target)

    # ── Backward pass ─────────────────────────────────────────

    def backward(self, prediction, target, learning_rate):
        """
        Backpropagate the loss through all layers.

        1. Start with loss gradient at the output.
        2. Walk BACKWARDS through layers (reversed).
        3. Each layer updates its weights and passes gradient further back.
        """
        grad = self.mse_loss_derivative(prediction, target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    # ── One training step ─────────────────────────────────────

    def train_step(self, x, y, learning_rate=0.1):
        """Forward pass + compute loss + backward pass. Returns loss."""
        pred = self.forward(x)
        loss = self.mse_loss(pred, y)
        self.backward(pred, y, learning_rate)
        return loss


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    net = Network([2, 4, 1])
    x = np.array([0.5, 0.8])
    y = np.array([1.0])

    print("=== Network [2 → 4 → 1] ===")
    print(f"Before training: pred = {net.forward(x)[0]:.4f}  target = {y[0]}")

    for _ in range(200):
        net.train_step(x, y, learning_rate=0.1)

    print(f"After 200 steps: pred = {net.forward(x)[0]:.4f}  target = {y[0]}")
    print("  (should be noticeably closer to 1.0)")