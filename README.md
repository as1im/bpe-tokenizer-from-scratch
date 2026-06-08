# Neural network from scratch — NumPy only

A fully-connected neural network built using **only Python and NumPy** — no PyTorch, no TensorFlow, no magic.

The goal is understanding, not performance. Every line is written to be readable and explained.

---

## What it does

Trains a neural network to solve the XOR problem:

```
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0
```

XOR cannot be solved by a straight line (it is not linearly separable). You *need* a hidden layer. This is the classic proof that multi-layer networks can learn non-linear patterns.

---

## File structure

```
├── neuron.py      — a single neuron: weights, sigmoid, forward, backward
├── layer.py       — N neurons reading the same input
├── network.py     — chain layers together, MSE loss, full backprop loop
├── train_xor.py   — trains the network, prints predictions, saves loss curve
└── loss_curve.png — generated when you run train_xor.py
```

Each file runs independently as a sanity check:

```bash
python neuron.py    # single neuron output
python layer.py     # layer of 3 neurons
python network.py   # network trains on one example
python train_xor.py # full XOR training
```

---

## How to run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/neural-net-from-scratch.git
cd neural-net-from-scratch

# 2. Install the only dependency
pip install numpy matplotlib

# 3. Run
python train_xor.py
```

Expected output:

```
Training on XOR...

  Epoch    Avg Loss
----------------------
      0    0.2731
   1000    0.2489
   2000    0.1823
   ...
   9000    0.0041

── Final predictions ─────────────────────────
  [0 0]  →  pred: 0.047  target: 0  ✓
  [0 1]  →  pred: 0.951  target: 1  ✓
  [1 0]  →  pred: 0.953  target: 1  ✓
  [1 1]  →  pred: 0.052  target: 0  ✓

All correct! Backprop is working.
```

---

## Architecture

```
Input (2)  →  Hidden layer (4 neurons)  →  Output (1)
```

- **Activation**: Sigmoid — squashes output to (0, 1)
- **Loss**: Mean Squared Error
- **Optimizer**: Vanilla gradient descent
- **Learning rate**: 0.1
- **Epochs**: 10,000

---

## What I learned building this

**Backpropagation is just the chain rule, applied repeatedly.**
Before writing this, backprop felt like a black box. After manually coding `d_z = d_output * sigmoid_derivative(output)` and watching loss actually decrease, it became obvious. Every weight update is just: "how much did this weight contribute to the error?"

**Why cached values matter.**
During the backward pass, we need the inputs and outputs from the forward pass. Libraries like PyTorch record this automatically in a computation graph. Here we cache `last_input` and `last_output` manually — which makes the auto-grad magic in PyTorch completely transparent.

**XOR requires a hidden layer — this is not trivial.**
A single neuron draws one straight line. XOR requires two lines. The hidden layer learns to construct two intermediate features that the output neuron can then linearly separate. Watching this work confirmed that "depth = more expressive power" is not just a slogan.

**Weights are initialized small on purpose.**
Starting with large weights means large activations, which push sigmoid into its flat regions (near 0 or 1), where gradients are almost zero. Training stalls. Small init (×0.1) keeps activations in the steep part of sigmoid where gradients are healthy.

---

## Concepts covered

| Concept | Where |
|---|---|
| Weighted sum (dot product) | `neuron.py → forward()` |
| Sigmoid activation + derivative | `neuron.py → sigmoid()` |
| Chain rule / backprop | `neuron.py → backward()` |
| Gradient descent weight update | `neuron.py → backward()` |
| Layer as collection of neurons | `layer.py` |
| Gradient summation across neurons | `layer.py → backward()` |
| MSE loss + derivative | `network.py` |
| Full forward + backward loop | `network.py → train_step()` |
| Non-linear problem (XOR) | `train_xor.py` |

---

## Next steps

- [ ] Add a second hidden layer and compare convergence
- [ ] Replace sigmoid with ReLU and observe the difference
- [ ] Implement momentum-based gradient descent
- [ ] Visualize the decision boundary learned for XOR

---

*Part of my AI/ML bootcamp learning projects. Built to understand — not to compete with PyTorch.*