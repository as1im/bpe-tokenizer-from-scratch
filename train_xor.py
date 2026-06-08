import numpy as np
from network import Network

# ── XOR dataset ───────────────────────────────────────────────
# XOR cannot be solved with a single line (not linearly separable).
# A hidden layer is REQUIRED. This is why neural networks exist.
#
#   [0, 0] → 0    [0, 1] → 1
#   [1, 0] → 1    [1, 1] → 0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0],    [1],    [1],    [0]],    dtype=float)

# ── Build and train ───────────────────────────────────────────
np.random.seed(0)
net = Network([2, 4, 1])  # 2 inputs → 4 hidden → 1 output

EPOCHS = 10_000
LEARNING_RATE = 0.1
losses = []

print("Training on XOR...\n")
print(f"{'Epoch':>7}  {'Avg Loss':>10}")
print("-" * 22)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for x, y in zip(X, Y):
        total_loss += net.train_step(x, y, learning_rate=LEARNING_RATE)
    avg_loss = total_loss / len(X)
    losses.append(avg_loss)

    if epoch % 1000 == 0:
        print(f"{epoch:>7}  {avg_loss:>10.4f}")

# ── Final predictions ─────────────────────────────────────────
print("\n── Final predictions ─────────────────────────")
all_correct = True
for x, y in zip(X, Y):
    pred = net.forward(x)
    correct = abs(pred[0] - y[0]) < 0.1
    if not correct:
        all_correct = False
    mark = "✓" if correct else "✗"
    print(f"  {x.astype(int)}  →  pred: {pred[0]:.3f}  target: {int(y[0])}  {mark}")

print()
if all_correct:
    print("All correct! Backprop is working.")
else:
    print("Some wrong — try increasing EPOCHS or adjusting LEARNING_RATE.")

# ── Loss curve ────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.plot(losses, color="#534AB7", linewidth=1.5)
    plt.title("Loss over training — XOR problem", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Loss curve saved → loss_curve.png")
except ImportError:
    print("(pip install matplotlib to generate loss_curve.png)")