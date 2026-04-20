# Self-Pruning Neural Network — Report

**Tredence AI Engineering Internship — Case Study Submission**  
**Mona Agrawal | RA2311003011733**

---

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

The sigmoid function maps any real-valued `gate_score` to a gate value in the range (0, 1). Without any regularization, the network has no incentive to push these gate values toward zero — they will settle wherever the classification loss is minimized.

Adding an **L1 penalty** on the gate values changes this. The L1 norm penalizes each gate proportional to its absolute value. Since gates are always positive after sigmoid, this penalty equals the sum (or mean) of all gate values. Minimizing this term directly pressures every gate toward zero.

The key property that makes L1 effective here — as opposed to L2 — is that L1 applies a **constant gradient** regardless of the gate's current magnitude. L2 produces a gradient proportional to the value itself, so as a weight shrinks toward zero, the gradient shrinks too and the weight never quite reaches zero. L1 does not have this problem: it applies equal pressure at every magnitude, which is what allows gates to collapse to (near) zero and stay there.

In practice:

- Gates that are **genuinely useful** to the network resist the penalty because eliminating them would hurt classification loss more than it saves on sparsity loss.
- Gates that correspond to **redundant or weak connections** offer little benefit to classification, so the L1 penalty wins and drives them to zero.

The result is a naturally sparse network where only the most informative connections survive.

**Note on threshold:** Sparsity is reported as the percentage of gates with value below `0.1`. Gates never reach exactly zero due to the asymptotic nature of sigmoid, so this threshold captures all gates that are functionally inactive.

---

## 2. Results

Training was conducted on CIFAR-10 for **35 epochs** across four values of λ using the Adam optimizer. Gate scores were trained at a higher learning rate (`5e-3`) than weights (`1e-3`) to allow the pruning mechanism to adapt faster.

| Lambda | Test Accuracy | Sparsity (%) |
|:------:|:-------------:|:------------:|
| 0.05   | 61.01%        | 46.68%       |
| 0.1    | 61.29%        | 51.43%       |
| 0.5    | 61.46%        | 69.58%       |
| 1.0    | 61.78%        | 78.60%       |

---

## 3. Analysis

### Sparsity increases consistently with λ

As expected, higher λ places a stronger penalty on active gates, causing more of them to collapse toward zero. Sparsity grows from ~47% at λ=0.05 to ~79% at λ=1.0 — nearly doubling the number of pruned weights.

### Accuracy remains stable across all λ values

The most notable observation is that test accuracy barely changes despite large increases in sparsity. Across all four settings, accuracy stays within a ~0.8% band (61.01% to 61.78%). This indicates that the network, even in its dense form, contains significant **redundancy** — a large fraction of its connections carry little discriminative information and can be removed without meaningful loss.

### Higher λ does not hurt — it slightly helps

Counterintuitively, the highest λ (1.0) produces both the highest sparsity and the marginally highest accuracy. This is likely a mild regularization effect: by eliminating weak, noisy connections, the network is forced to route information through its stronger pathways, which slightly reduces overfitting.

### The accuracy ceiling is set by architecture, not pruning

All four runs converge to approximately 61-62% test accuracy. This ceiling is a property of the underlying architecture — a simple 3-layer MLP applied to flattened 32×32 images — not of the pruning mechanism. A convolutional backbone would produce significantly higher accuracy, but that was not the focus of this task.

### λ trade-off in practice

In a deployment scenario, the optimal λ would be chosen based on a target sparsity or memory budget. Based on these results, λ=0.5 appears to be a practical sweet spot: it prunes nearly 70% of weights while losing less than 0.5% accuracy compared to the least-pruned model.

---

## 4. Gate Distribution

The plot below shows the distribution of all gate values for the best model (λ = 1.0). The dashed vertical line marks the sparsity threshold of 0.1.

![Gate Distribution](results/gate_distribution.png)

The distribution shows a large spike concentrated near zero, confirming that the majority of gates have been driven to near-zero values and are effectively pruned. A smaller tail of active gates (values > 0.1) represents the connections the network identified as genuinely necessary. This bimodal pattern — a dense cluster at zero and a sparse tail of surviving connections — is the expected signature of a successfully trained self-pruning network.
