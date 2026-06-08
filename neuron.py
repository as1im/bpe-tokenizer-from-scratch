import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        # Random small weights — one per input
        # 0.1 scale keeps weights small so training starts stable
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0

        # Cached during forward pass, needed for backprop
        self.last_input = None
        self.last_output = None

    # ── Activation ────────────────────────────────────────────

    def sigmoid(self, z):
        """Squash any number into the range (0, 1)."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, output):
        """
        Derivative of sigmoid, expressed in terms of the output.
        Formula: s'(x) = s(x) * (1 - s(x))
        We can compute this from the output alone — no need for the original input.
        """
        return output * (1 - output)

    # ── Forward pass ──────────────────────────────────────────

    def forward(self, inputs):
        """
        Compute the neuron's output:
          1. Weighted sum: z = w1*x1 + w2*x2 + ... + bias
          2. Activation:   output = sigmoid(z)
        """
        self.last_input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.last_output = self.sigmoid(z)
        return self.last_output

    # ── Backward pass ─────────────────────────────────────────

    def backward(self, d_output, learning_rate):
        """
        Backpropagate through this neuron.

        d_output: gradient of loss w.r.t. this neuron's output
                  (received from the layer ahead)

        Chain rule, step by step:
          d_z       = d_output * sigmoid'(output)   # through activation
          d_weights = d_z * last_input              # through weighted sum
          d_bias    = d_z                            # bias has gradient = 1
          d_input   = d_z * weights                 # sent to previous layer
        """
        d_z = d_output * self.sigmoid_derivative(self.last_output)

        d_weights = d_z * self.last_input
        d_bias    = d_z
        d_input   = d_z * self.weights

        # Gradient descent: move opposite to gradient
        self.weights -= learning_rate * d_weights
        self.bias    -= learning_rate * d_bias

        return d_input  # gradient flows to the layer behind us


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = Neuron(n_inputs=3)
    x = np.array([0.5, -0.2, 0.8])

    print("=== Single Neuron ===")
    print(f"inputs : {x}")
    print(f"weights: {n.weights.round(4)}")
    print(f"bias   : {n.bias}")
    out = n.forward(x)
    print(f"output : {out:.4f}  (between 0 and 1 — sigmoid working)")