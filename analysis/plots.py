import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collect.dataset import TimingDataset

FIGURE_DPI = 150
FIGURE_DIR = "figures"


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


def plot_timing_distributions(dataset: TimingDataset, title: str = "",
                              save_path: Optional[str] = None) -> plt.Figure:
    groups = dataset.get_timing_by_action()
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    data = []
    for action in sorted(groups.keys()):
        labels.append(f"Action {action}")
        data.append(groups[action])

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Inference Cycles")
    ax.set_title(title or "Timing Distribution by Action")
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_mi_heatmap(mi_data: Dict[str, Dict[str, float]],
                    title: str = "",
                    save_path: Optional[str] = None) -> plt.Figure:
    labels = list(mi_data.keys())
    sub_keys = list(next(iter(mi_data.values())).keys()) if mi_data else []

    if not sub_keys:
        matrix = np.array([[mi_data[l] for l in labels]])
        row_labels = ["MI (bits)"]
    else:
        matrix = np.array([[mi_data[l][sk] for sk in sub_keys] for l in labels])
        row_labels = labels

    fig, ax = plt.subplots(figsize=(max(8, len(sub_keys)), max(4, len(labels) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(sub_keys)))
    ax.set_xticklabels(sub_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="black" if matrix[i, j] < matrix.max() * 0.7 else "white",
                    fontsize=8)

    fig.colorbar(im, ax=ax, label="MI (bits)")
    ax.set_title(title or "Mutual Information Heatmap")

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "",
                          save_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Pred {i}" for i in range(n)])
    ax.set_yticklabels([f"True {i}" for i in range(n)])
    ax.set_xlabel("Predicted Action")
    ax.set_ylabel("True Action")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    ax.set_title(title or "Confusion Matrix: Timing → Action Classifier")

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return fig


def plot_per_layer_breakdown(dataset: TimingDataset, n_ops: int = 6,
                             title: str = "",
                             save_path: Optional[str] = None) -> plt.Figure:
    arrays = dataset.to_arrays()
    per_op = arrays["per_op_cycles"].astype(np.float64)[:, :n_ops]
    actions = arrays["actions"]

    fig, axes = plt.subplots(1, n_ops, figsize=(4 * n_ops, 5), sharey=True)
    if n_ops == 1:
        axes = [axes]

    for op_idx in range(n_ops):
        ax = axes[op_idx]
        unique_actions = sorted(np.unique(actions))
        data = [per_op[actions == a, op_idx] for a in unique_actions]
        data = [d for d in data if len(d) > 0]
        if data:
            ax.violinplot(data, showmeans=True)
        ax.set_title(f"Op {op_idx}")
        ax.set_xlabel("Action")
        if op_idx == 0:
            ax.set_ylabel("Cycles")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title or "Per-Operator Timing by Action")
    fig.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")

    return fig
