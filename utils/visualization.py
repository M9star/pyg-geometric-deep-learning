"""
Shared visualization helpers
============================
Small, dependency-light plotting functions reused across the modules.
Every function saves a PNG to the `assets/` folder so each experiment
leaves a visual record (great for a GitHub README).

We use a non-interactive matplotlib backend so scripts run on headless
machines / CI without a display.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# Repo root = one level up from this file's folder (utils/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)


def _save(fig, filename: str) -> str:
    """Save a figure into assets/ and return its path."""
    path = os.path.join(ASSETS, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved] {os.path.relpath(path, ROOT)}")
    return path


def plot_training_curves(history: dict, filename: str, title: str = "Training") -> str:
    """
    history: dict with optional keys 'loss', 'train_acc', 'val_acc', 'test_acc'
             (each a list, one value per epoch).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "loss" in history:
        axes[0].plot(history["loss"], color="crimson")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].grid(alpha=0.3)

    for key, color in (("train_acc", "tab:blue"),
                       ("val_acc", "tab:orange"),
                       ("test_acc", "tab:green")):
        if key in history:
            axes[1].plot(history[key], label=key, color=color)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontweight="bold")
    return _save(fig, filename)


def plot_embeddings_tsne(embeddings, labels, filename: str,
                         title: str = "Node embeddings (t-SNE)") -> str:
    """
    Project high-dim node embeddings to 2D with t-SNE and color by label.
    `embeddings`: (N, D) array/tensor. `labels`: (N,) array/tensor of class ids.
    """
    import numpy as np
    from sklearn.manifold import TSNE

    emb = _to_numpy(embeddings)
    lab = _to_numpy(labels)

    # t-SNE perplexity must be < n_samples; clamp for tiny graphs.
    perplexity = min(30, max(5, emb.shape[0] // 4))
    proj = TSNE(n_components=2, perplexity=perplexity,
                init="pca", random_state=42).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap="tab10", s=18, alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    legend = ax.legend(*scatter.legend_elements(), title="class",
                       loc="best", fontsize=8)
    ax.add_artist(legend)
    return _save(fig, filename)


def draw_graph(edge_index, filename: str, node_color=None,
               title: str = "Graph", num_nodes: int | None = None) -> str:
    """Draw a (small) graph with networkx, optionally coloring nodes."""
    import networkx as nx

    ei = _to_numpy(edge_index)
    G = nx.Graph()
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(ei[0], ei[1]))

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G, pos, ax=ax, with_labels=True, node_size=500,
        node_color=(_to_numpy(node_color) if node_color is not None else "#9ecae1"),
        cmap="tab10", edge_color="#888", font_size=9,
    )
    ax.set_title(title, fontweight="bold")
    ax.axis("off")
    return _save(fig, filename)


def _to_numpy(x):
    """Accept torch tensors or numpy arrays."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    import numpy as np
    return np.asarray(x)
