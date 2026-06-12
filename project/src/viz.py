"""
Project-specific plots (confusion matrix) + thin wrappers that save into
the project's own run directory rather than the repo-wide assets/ folder.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_curves(history: dict, path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], color="crimson")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_acc"], label="train", color="tab:blue")
    axes[1].plot(history["val_acc"], label="val", color="tab:orange")
    axes[1].plot(history["test_acc"], label="test", color="tab:green")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_confusion_matrix(y_true, y_pred, num_classes, path: str, title: str):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_title(title, fontweight="bold")
    # annotate counts
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_tsne(embeddings, labels, path: str, title: str):
    from sklearn.manifold import TSNE

    emb = embeddings.detach().cpu().numpy()
    lab = labels.detach().cpu().numpy()
    perplexity = min(30, max(5, emb.shape[0] // 4))
    proj = TSNE(n_components=2, perplexity=perplexity,
                init="pca", random_state=42).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap="tab10", s=18, alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ax.legend(*sc.legend_elements(), title="class", fontsize=8))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
