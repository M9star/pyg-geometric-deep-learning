"""
02 · Node classification on Cora
================================
Train GCN / GAT / GraphSAGE to classify papers in a citation network.

    python 02_node_classification/train.py --model gcn   # or gat / sage

The Cora dataset downloads automatically on first run (~few MB) into ./data.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import plot_embeddings_tsne, plot_training_curves  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class GNN(torch.nn.Module):
    """A generic 2-layer GNN; swap the conv type via `kind`."""

    def __init__(self, in_dim, hidden, out_dim, kind="gcn", heads=8):
        super().__init__()
        self.kind = kind
        self.dropout = Dropout(0.5)

        if kind == "gcn":
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)
        elif kind == "sage":
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, out_dim)
        elif kind == "gat":
            # multi-head attention; concat heads in layer 1, average in layer 2.
            self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=0.5)
            self.conv2 = GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=0.5)
        else:
            raise ValueError(f"unknown model kind: {kind}")

    def forward(self, x, edge_index, return_embedding=False):
        h = self.conv1(x, edge_index)
        h = F.elu(h) if self.kind == "gat" else F.relu(h)
        h = self.dropout(h)
        emb = self.conv2(h, edge_index)        # final node embeddings (= logits here)
        if return_embedding:
            return emb
        return F.log_softmax(emb, dim=1)


def accuracy(logits, y, mask):
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gcn", "gat", "sage"], default="gcn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # --- data ---
    dataset = Planetoid(root=os.path.join(HERE, "data", "Cora"), name="Cora")
    data = dataset[0].to(device)
    print(f"Cora: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{dataset.num_features} features, {dataset.num_classes} classes")
    print(f"labeled for training: {int(data.train_mask.sum())} nodes\n")

    # --- model ---
    model = GNN(dataset.num_features, args.hidden, dataset.num_classes,
                kind=args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    history = {"loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    best_val, best_test = 0.0, 0.0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # evaluate
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            tr = accuracy(logits, data.y, data.train_mask)
            va = accuracy(logits, data.y, data.val_mask)
            te = accuracy(logits, data.y, data.test_mask)

        history["loss"].append(loss.item())
        history["train_acc"].append(tr)
        history["val_acc"].append(va)
        history["test_acc"].append(te)

        # keep the test acc at the best validation epoch (proper model selection)
        if va > best_val:
            best_val, best_test = va, te

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {loss:.4f} | "
                  f"train {tr:.3f} | val {va:.3f} | test {te:.3f}")

    print(f"\nBest val acc {best_val:.3f}  ->  test acc {best_test:.3f}")

    # --- visualizations ---
    plot_training_curves(history, f"02_{args.model}_curves.png",
                         title=f"Cora node classification — {args.model.upper()}")

    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index, return_embedding=True)
    plot_embeddings_tsne(emb, data.y, f"02_{args.model}_tsne.png",
                         title=f"Learned node embeddings ({args.model.upper()}) — t-SNE")


if __name__ == "__main__":
    main()
