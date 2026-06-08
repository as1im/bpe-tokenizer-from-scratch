import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, n_inputs, n_neurons):
        """
        Create a layer of n_neurons neurons, each expecting n_inputs.
        All neurons see the same input — they each learn different patterns.
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs):
        """
        Pass inputs through every neuron.
        Output is a vector — one value per neuron.
        """
        return np.array([n.forward(inputs) for n in self.neurons])

    def backward(self, d_outputs, learning_rate):
        """
        d_outputs: one gradient per neuron in this layer.

        Each neuron computes its own backward pass and returns
        the gradient it wants to send to the previous layer.

        Because all neurons received the SAME input, their
        individual d_inputs are summed — combined responsibility.
        """
        d_inputs = np.array([
            neuron.backward(d_out, learning_rate)
            for neuron, d_out in zip(self.neurons, d_outputs)
        ])
        return d_inputs.sum(axis=0)  # sum contributions from all neurons


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    layer = Layer(n_inputs=2, n_neurons=3)
    x = np.array([0.5, 0.8])

    print("=== Layer (2 inputs → 3 neurons) ===")
    out = layer.forward(x)
    print(f"input : {x}")
    print(f"output: {out.round(4)}")
    print(f"shape : {out.shape}  (one value per neuron)")