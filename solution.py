"""
╔══════════════════════════════════════════════════════════════════╗
║         SELF-PRUNING NEURAL NETWORK — CIFAR-10                  ║
║         Tredence AI Engineering Intern — Case Study             ║
╚══════════════════════════════════════════════════════════════════╝

Core Algorithm
──────────────
1. Each weight matrix W has a companion gate_scores tensor (same shape).
2. gates = sigmoid(gate_scores)         ∈ (0, 1)
3. pruned_weight = W ⊙ gates            (element-wise multiply)
4. Total Loss = CrossEntropy + λ·Σ gates  (L1 pressure drives gates → 0)

Gradients flow through BOTH W and gate_scores via PyTorch autograd.
No monkey-patching of nn.Linear — PrunableLinear is built from scratch.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — PrunableLinear Layer
# ═══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Custom linear layer with a learnable gating mechanism.

    For each weight W_ij, a companion gate_score s_ij is maintained.
    The effective weight used in the forward pass is:

            W̃_ij = W_ij · σ(s_ij)

    When s_ij → -∞,  σ(s_ij) → 0  → weight is effectively pruned.
    When s_ij → +∞,  σ(s_ij) → 1  → weight is fully active.

    The L1 sparsity loss penalises the sum of all gate values, creating
    gradient pressure to push s_ij downward (and therefore gates toward 0).
    The classification loss creates opposing pressure to keep useful gates open.
    This tension results in a bimodal gate distribution after training.

    Gradient flow:
      ∂L/∂W_ij       = ∂L/∂output · σ(s_ij)           (through pruned_weight)
      ∂L/∂s_ij       = ∂L/∂output · W_ij · σ'(s_ij) + λ · σ'(s_ij)
    Both gradients are computed automatically by autograd.

    Parameters
    ──────────
    in_features  : input dimension
    out_features : output dimension
    bias         : whether to include a bias term (default: True)
    gate_init    : initial value for gate_scores (default: 0.0 → gates ≈ 0.5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Primary weight  (Kaiming uniform init, same as nn.Linear) ─────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # ── Bias ───────────────────────────────────────────────────────────
        if bias:
            bound     = 1.0 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # ── Gate scores — SAME shape as weight, also a trainable parameter ─
        # Initialised to gate_init so σ(gate_init) = 0.5 at the start.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(gate_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: squash gate_scores → (0,1) via sigmoid  [differentiable]
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: element-wise gating of weights
        pruned_weight = self.weight * gates              # shape: (out, in)

        # Step 3: standard linear transform  y = x W̃ᵀ + b
        return F.linear(x, pruned_weight, self.bias)

    # ── Utility properties ────────────────────────────────────────────────────

    @property
    def gates(self) -> torch.Tensor:
        """Gate values detached from the computation graph."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values (added to total loss during training)."""
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_fraction(self, threshold: float = 1e-2) -> float:
        """Fraction of connections with gate < threshold (effectively pruned)."""
        return (self.gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


# ═══════════════════════════════════════════════════════════════════════════════
# Network Definition
# ═══════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10  (3×32×32 input → 10-class output).

    Architecture:  Flatten → [PrunableLinear → BN → GELU → Dropout] × 3 → PrunableLinear

    Design choices
    ──────────────
    • GELU instead of ReLU: smoother gradient landscape aids gate optimisation.
    • BatchNorm after prunable linear: stabilises activations as gates collapse.
    • Dropout in hidden layers only: regularises the classifier, not the gates.
    • All nn.Linear replaced by PrunableLinear — every weight can be pruned.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        dropout:     float           = 0.3,
        gate_init:   float           = 0.0,
    ) -> None:
        super().__init__()
        dims   = [3 * 32 * 32] + list(hidden_dims) + [10]
        layers: List[nn.Module] = []

        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(PrunableLinear(d_in, d_out, gate_init=gate_init))
            if i < len(hidden_dims):                  # hidden layers only
                layers.append(nn.BatchNorm1d(d_out))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))         # flatten → (B, 3072)

    # ── Pruning analytics ─────────────────────────────────────────────────────

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate norms across every PrunableLinear layer."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Network-wide fraction of pruned connections."""
        layers = self.prunable_layers()
        total  = sum(l.weight.numel() for l in layers)
        pruned = sum((l.gates < threshold).sum().item() for l in layers)
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Concatenated gate values from all prunable layers (for plotting)."""
        return torch.cat(
            [l.gates.flatten() for l in self.prunable_layers()]
        ).numpy()

    def layer_sparsities(self, threshold: float = 1e-2) -> Dict[str, float]:
        return {
            f"layer_{i}": round(l.sparsity_fraction(threshold), 4)
            for i, l in enumerate(self.prunable_layers())
        }

    def active_weight_count(self, threshold: float = 1e-2) -> int:
        return sum(
            (l.gates >= threshold).sum().item()
            for l in self.prunable_layers()
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def get_loaders(
    batch_size: int = 256,
    data_root:  str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 with standard normalisation and mild training augmentation."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 & 3 — Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model:         SelfPruningNet,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    criterion:     nn.Module,
    lambda_sparse: float,
    device:        torch.device,
) -> Dict[str, float]:
    """One full training epoch. Returns averaged loss components."""
    model.train()
    cls_sum = spar_sum = total_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits    = model(images)
        cls_loss  = criterion(logits, labels)

        # ── Sparsity loss: L1 norm of ALL gate values across the network ───
        spar_loss = model.total_sparsity_loss()

        # ── Total loss: classification + sparsity trade-off ────────────────
        loss = cls_loss + lambda_sparse * spar_loss
        loss.backward()

        # Gradient clipping prevents gate_scores exploding in early epochs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        cls_sum   += cls_loss.item()
        spar_sum  += spar_loss.item()
        total_sum += loss.item()

    n = len(loader)
    return {
        "loss":          total_sum / n,
        "cls_loss":      cls_sum   / n,
        "sparsity_loss": spar_sum  / n,
    }


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    """Returns top-1 accuracy on the given loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(
    lambda_sparse: float,
    epochs:        int,
    train_loader:  DataLoader,
    test_loader:   DataLoader,
    device:        torch.device,
) -> Dict:
    """Full training run for one lambda value. Returns complete result dict."""
    print(f"\n{'─'*62}")
    print(f"  Experiment  λ = {lambda_sparse:.1e}  ({epochs} epochs)")
    print(f"{'─'*62}")

    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    history = []
    t0      = time.time()

    for epoch in range(1, epochs + 1):
        metrics  = train_epoch(model, train_loader, optimizer, criterion, lambda_sparse, device)
        scheduler.step()
        acc      = evaluate(model, test_loader, device)
        sparsity = model.global_sparsity()
        history.append({**metrics, "accuracy": acc, "sparsity": sparsity, "epoch": epoch})

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  ep {epoch:3d}/{epochs} | "
                f"loss {metrics['loss']:.4f} | "
                f"acc {acc*100:5.2f}% | "
                f"sparsity {sparsity*100:5.2f}%"
            )

    elapsed = time.time() - t0
    print(f"\n  ✓  Done in {elapsed:.1f}s")
    print(f"  ✓  Final test accuracy : {history[-1]['accuracy']*100:.2f}%")
    print(f"  ✓  Final sparsity      : {history[-1]['sparsity']*100:.2f}%")
    print(f"  ✓  Per-layer sparsities: {model.layer_sparsities()}")

    return {
        "lambda":           lambda_sparse,
        "test_accuracy":    history[-1]["accuracy"],
        "sparsity_level":   history[-1]["sparsity"],
        "layer_sparsities": model.layer_sparsities(),
        "active_weights":   model.active_weight_count(),
        "gate_values":      model.all_gate_values(),
        "history":          history,
        "elapsed_s":        round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: List[Dict], best_idx: int, out_dir: Path) -> None:
    """
    Two-row figure:
      Row 1 — Gate-value histograms (one per λ)        ← main deliverable
      Row 2 — Accuracy & sparsity training curves
    """
    n      = len(results)
    fig    = plt.figure(figsize=(6 * n, 11), constrained_layout=True)
    gs     = gridspec.GridSpec(2, n, figure=fig)
    colors = ["#27ae60", "#2980b9", "#c0392b", "#8e44ad"]

    # ── Row 1: gate distributions ─────────────────────────────────────────────
    for i, res in enumerate(results):
        ax  = fig.add_subplot(gs[0, i])
        col = colors[i % len(colors)]
        g   = res["gate_values"]

        ax.hist(g, bins=120, color=col, alpha=0.82, edgecolor="white", linewidth=0.2)
        ax.axvline(0.01, color="dimgray", ls="--", lw=1.2, label="prune threshold (0.01)")

        pct = (g < 0.01).mean() * 100
        ax.text(
            0.60, 0.87,
            f"{pct:.1f}% pruned",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=col, alpha=0.22),
        )
        star = "★ Best  " if i == best_idx else ""
        ax.set_title(
            f"{star}λ = {res['lambda']:.1e}\n"
            f"Acc {res['test_accuracy']*100:.1f}%  ·  "
            f"Sparsity {res['sparsity_level']*100:.1f}%",
            fontweight="bold", fontsize=11,
        )
        ax.set_xlabel("Gate value", fontsize=9)
        ax.set_ylabel("Count",      fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.22)

    # ── Row 2: training curves ─────────────────────────────────────────────────
    for i, res in enumerate(results):
        ax  = fig.add_subplot(gs[1, i])
        ax2 = ax.twinx()
        col = colors[i % len(colors)]
        h   = res["history"]
        xs  = [r["epoch"]        for r in h]
        acc = [r["accuracy"]*100 for r in h]
        spr = [r["sparsity"]*100 for r in h]

        ax.plot( xs, acc, color=col,           lw=2.0,  label="Test Acc %")
        ax2.plot(xs, spr, color=col, ls="--",  lw=1.5, alpha=0.65, label="Sparsity %")

        ax.set_xlabel("Epoch",       fontsize=9)
        ax.set_ylabel("Accuracy %",  fontsize=9, color=col)
        ax2.set_ylabel("Sparsity %", fontsize=9, color=col, alpha=0.75)
        ax.set_title(f"Training curve  λ={res['lambda']:.1e}", fontsize=10)
        ax.grid(alpha=0.2)

        lines  = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    fig.suptitle(
        "Self-Pruning Neural Network — CIFAR-10\n"
        "Gate Value Distributions & Training Dynamics",
        fontsize=14, fontweight="bold",
    )
    out_path = out_dir / "gate_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  [Saved] {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Console + JSON output
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[Dict]) -> None:
    print("\n" + "═" * 72)
    print(f"{'Lambda':<12} {'Test Acc %':<15} {'Sparsity %':<15} {'Active Wts':<14} {'Time(s)'}")
    print("─" * 72)
    for r in results:
        print(
            f"{r['lambda']:<12.1e}"
            f"{r['test_accuracy']*100:<15.2f}"
            f"{r['sparsity_level']*100:<15.2f}"
            f"{r['active_weights']:<14,}"
            f"{r['elapsed_s']}"
        )
    print("═" * 72)


def save_json(results: List[Dict], out_dir: Path) -> None:
    serialisable = [
        {k: v for k, v in r.items() if k not in ("gate_values", "history")}
        for r in results
    ]
    path = out_dir / "results.json"
    path.write_text(json.dumps(serialisable, indent=2))
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network — CIFAR-10")
    p.add_argument("--epochs",     type=int,   default=30,
                   help="Epochs per λ run  (default 30; use 50 for higher accuracy)")
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lambdas",    type=float, nargs="+", default=[1e-5, 1e-4, 1e-3],
                   help="λ values to sweep  (default: 1e-5 1e-4 1e-3)")
    p.add_argument("--out_dir",    type=str,   default=".")
    p.add_argument("--data_root",  type=str,   default="./data")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs}")
    print(f"λ sweep : {args.lambdas}")

    train_loader, test_loader = get_loaders(args.batch_size, args.data_root)

    results = []
    for lam in args.lambdas:
        results.append(
            run_experiment(lam, args.epochs, train_loader, test_loader, device)
        )

    print_table(results)
    save_json(results, out_dir)

    best_idx = max(range(len(results)), key=lambda i: results[i]["test_accuracy"])
    print(
        f"\n  Best: λ={results[best_idx]['lambda']:.1e}  "
        f"acc={results[best_idx]['test_accuracy']*100:.2f}%  "
        f"sparsity={results[best_idx]['sparsity_level']*100:.2f}%"
    )

    plot_results(results, best_idx, out_dir)


if __name__ == "__main__":
    main()
