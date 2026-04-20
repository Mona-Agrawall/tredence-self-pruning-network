# Self-Pruning Neural Network

**Tredence AI Engineering Internship — Case Study Submission**  
**Mona Mahendra Kumar Agrawal | RA2311003011733**

---

## Overview

This project implements a self-pruning neural network that learns to remove its own unnecessary weights **during training**, not as a post-training step.

Each weight in the network is associated with a learnable scalar gate. During training, a sparsity regularization term pushes most gates toward zero, effectively pruning the corresponding weights on the fly. The result is a sparse network that retains accuracy while discarding redundant connections.

The model is trained and evaluated on **CIFAR-10** across four values of the sparsity hyperparameter λ (lambda) to demonstrate the accuracy vs. sparsity trade-off.

---

## Project Structure

```
.
├── prunable_network.py       # Main script: model, training loop, evaluation, plots
├── requirements.txt          # Dependencies
├── report.md                 # Analysis report with results and explanation
├── results/
│   ├── results.txt           # Accuracy and sparsity table for all λ values
│   └── gate_distribution.png # Gate value histogram for best model (λ = 1.0)
└── README.md
```

---

## How It Works

### 1. PrunableLinear Layer

A custom linear layer that replaces `nn.Linear`. In addition to the standard `weight` and `bias` parameters, it contains a `gate_scores` tensor of the same shape as the weights.

During the forward pass:
- `gate_scores` are passed through a **sigmoid** to produce gates ∈ (0, 1)
- Weights are element-wise multiplied with their corresponding gates: `pruned_weight = weight * gate`
- The linear operation proceeds with these pruned weights

Gradients flow through both `weight` and `gate_scores` via standard autograd.

### 2. Sparsity Loss

The total training loss is:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

where `SparsityLoss` is the **mean of all gate values** across all `PrunableLinear` layers. This acts as an L1-style penalty that continuously pushes gates toward zero, encouraging sparsity.

### 3. Training Strategy

- **Separate learning rates** for weights (`lr=1e-3`) and gate scores (`lr=5e-3`) via Adam
- Gates are initialized with mean `-1.5` so they start near zero (low initial activity)
- Sparsity is measured as the percentage of gates below threshold `0.1`

---

## Results

| Lambda | Test Accuracy | Sparsity (%) |
|--------|:-------------:|:------------:|
| 0.05   | 61.01%        | 46.68%       |
| 0.1    | 61.29%        | 51.43%       |
| 0.5    | 61.46%        | 69.58%       |
| 1.0    | 61.78%        | 78.60%       |

- Higher λ → more sparsity, with negligible accuracy drop
- At λ = 1.0, nearly **79% of weights are pruned** while maintaining ~61% test accuracy
- The network successfully identifies and eliminates redundant connections during training

---

## Gate Distribution (Best Model — λ = 1.0)

![Gate Distribution](results/gate_distribution.png)

The large spike near 0 confirms successful pruning. The dashed line marks the sparsity threshold (0.1).

---

## Setup and Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run training

```bash
python prunable_network.py
```

CIFAR-10 will be downloaded automatically on first run into `./data/`.

### Outputs

After training completes:
- `results/results.txt` — accuracy and sparsity for all λ values
- `results/gate_distribution.png` — gate histogram for the best model

> **Note:** Training was run on CPU. Runtime will vary by machine. A GPU is recommended for faster experimentation.

---

## Dependencies

```
torch
torchvision
matplotlib
numpy
```
