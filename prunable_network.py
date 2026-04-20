"""Self-Pruning Neural Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# ─────────────────────────────────────────────
# Prunable Linear Layer
# ─────────────────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.normal_(self.gate_scores, mean=-1.5, std=0.1)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        return F.linear(x, self.weight * gates, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach().cpu()


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            PrunableLinear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            PrunableLinear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            PrunableLinear(128, 10),
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

    def get_layers(self):
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self):
        gates = torch.cat([
            torch.sigmoid(l.gate_scores).view(-1)
            for l in self.get_layers()
        ])
        return gates.mean()

    def sparsity(self, threshold=0.1):
        gates = torch.cat([
            l.get_gates().view(-1)
            for l in self.get_layers()
        ])
        return (gates < threshold).float().mean().item() * 100


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def get_data():
    t_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])

    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])

    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
    test  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=t_test)

    return (
        DataLoader(train, batch_size=128, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=256, shuffle=False, num_workers=2)
    )


# ─────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────
def train_epoch(model, loader, opt, device, lam):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)

        ce = F.cross_entropy(out, y)
        s  = model.sparsity_loss()
        loss = ce + lam * s

        loss.backward()
        opt.step()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, 100 * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
def plot(model, lam):
    gates = torch.cat([
        l.get_gates().view(-1)
        for l in model.get_layers()
    ]).numpy()

    plt.figure(figsize=(8,4))
    plt.hist(gates, bins=100)
    plt.axvline(0.1, linestyle="--")
    plt.title(f"Gate Distribution (λ={lam})")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/gate_distribution.png")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("Device:", device)
    print("="*60)

    train_loader, test_loader = get_data()

    lambdas = [0.05, 0.1, 0.5, 1.0]
    epochs = 35

    results = []
    best_model, best_lam, best_acc = None, None, -1

    for lam in lambdas:
        print(f"\n{'='*20} λ = {lam} {'='*20}")

        model = Net().to(device)

        gate_params = [p for n,p in model.named_parameters() if "gate_scores" in n]
        other_params = [p for n,p in model.named_parameters() if "gate_scores" not in n]

        opt = torch.optim.Adam([
            {"params": other_params, "lr": 1e-3},
            {"params": gate_params, "lr": 5e-3}
        ])

        for ep in range(1, epochs+1):
            loss, acc = train_epoch(model, train_loader, opt, device, lam)

            if ep % 5 == 0 or ep == 1:
                sp = model.sparsity()
                print(f"Epoch {ep:02d} | Acc {acc:5.1f}% | Sparsity {sp:5.1f}%")

        test_acc = evaluate(model, test_loader, device)
        sp = model.sparsity()

        print(f"\nFinal → Accuracy {test_acc:.2f}% | Sparsity {sp:.2f}%")

        results.append((lam, test_acc, sp))

        if test_acc > best_acc:
            best_acc, best_model, best_lam = test_acc, model, lam

    # PRINT CLEAN TABLE
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Lambda':<10}{'Accuracy':<15}{'Sparsity (%)':<15}")
    print("-"*60)

    with open("results/results.txt", "w") as f:
        for lam, acc, sp in results:
            line = f"{lam:<10}{acc:<15.2f}{sp:<15.2f}"
            print(line)
            f.write(line + "\n")

    print("="*60)

    plot(best_model, best_lam)

    print("\nSaved files:")
    print("results/results.txt")
    print("results/gate_distribution.png")


if __name__ == "__main__":
    main()