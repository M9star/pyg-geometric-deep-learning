"""
04 · Link prediction with a Graph Autoencoder
=============================================
Hide some edges of Cora, learn node embeddings whose dot product recovers
the missing links. Self-supervised — no node labels used for training.

    python 04_link_prediction/train.py
"""

import os
import sys

# Let unsupported ops fall back to CPU on Apple MPS. Must be set before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device import get_device  # noqa: E402
from utils.visualization import plot_embeddings_tsne, plot_training_curves  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class GCNEncoder(torch.nn.Module):
    """Maps node features -> low-dim embeddings via 2 GCN layers."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)        # latent embeddings z


def main():
    device = get_device()
    print(f"Using device: {device.type.upper()}")
    torch.manual_seed(42)

    # RandomLinkSplit carves the edges into train/val/test sets and creates
    # negative samples for evaluation automatically.
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(os.path.join(HERE, "data", "Cora"), name="Cora",
                        transform=transform)
    train_data, val_data, test_data = dataset[0]
    print(f"Cora link prediction | {dataset.num_features} features")
    print(f"train edges: {train_data.edge_index.size(1)} | "
          f"val pos: {val_data.pos_edge_label_index.size(1)} | "
          f"test pos: {test_data.pos_edge_label_index.size(1)}\n")

    # GAE wraps our encoder and provides recon_loss() + test() (AUC/AP).
    model = GAE(GCNEncoder(dataset.num_features, 32, 16)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    @torch.no_grad()
    def evaluate(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    history = {"val_auc": [], "val_ap": []}
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        # reconstruction loss over positive (real) train edges + sampled negatives
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss.backward()
        optimizer.step()

        auc, ap = evaluate(val_data)
        history["val_auc"].append(auc)
        history["val_ap"].append(ap)
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss:.4f} | val AUC {auc:.4f} | val AP {ap:.4f}")

    test_auc, test_ap = evaluate(test_data)
    print(f"\nTEST  AUC {test_auc:.4f} | AP {test_ap:.4f}")

    # --- visualizations ---
    # rename keys so the shared plotter labels them on the accuracy axis
    plot_training_curves(
        {"val_acc": history["val_auc"], "test_acc": history["val_ap"]},
        "04_gae_curves.png",
        title="Link prediction (GAE) — val AUC (blue) & AP (green)",
    )

    # color the learned embeddings by the (unused-in-training) paper topic
    raw = Planetoid(os.path.join(HERE, "data", "Cora"), name="Cora")[0]
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
    plot_embeddings_tsne(z, raw.y, "04_gae_tsne.png",
                         title="GAE node embeddings — t-SNE (color = paper topic)")


if __name__ == "__main__":
    main()
