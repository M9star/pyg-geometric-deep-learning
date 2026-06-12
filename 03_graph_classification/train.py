"""
03 · Graph classification on MUTAG
==================================
Classify whole molecules as mutagenic / not using a GIN.

    python 03_graph_classification/train.py

Demonstrates the two new ideas vs. node classification:
  1. mini-batching many graphs   (torch_geometric.loader.DataLoader)
  2. readout / global pooling    (nodes -> one vector per graph)
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import plot_embeddings_tsne, plot_training_curves  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hidden] * num_layers
        for i in range(num_layers):
            # GIN's UPDATE is a small MLP applied after sum-aggregation.
            mlp = Sequential(
                Linear(dims[i], hidden), BatchNorm1d(hidden), ReLU(),
                Linear(hidden, hidden), ReLU(),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, return_embedding=False):
        for conv in self.convs:
            x = conv(x, edge_index)
        # READOUT: collapse all nodes of each graph into one vector.
        graph_emb = global_add_pool(x, batch)          # [num_graphs, hidden]
        if return_embedding:
            return graph_emb
        h = F.relu(self.lin1(graph_emb))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    dataset = TUDataset(root=os.path.join(HERE, "data"), name="MUTAG").shuffle()
    print(f"MUTAG: {len(dataset)} graphs | "
          f"{dataset.num_features} node features | {dataset.num_classes} classes")

    # 80/20 split
    n_train = int(len(dataset) * 0.8)
    train_ds, test_ds = dataset[:n_train], dataset[n_train:]
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # peek at one batch to see graph-batching in action
    batch = next(iter(train_loader))
    print(f"one batch -> {batch.num_graphs} graphs glued into "
          f"{batch.num_nodes} total nodes; batch vector shape {tuple(batch.batch.shape)}\n")

    model = GIN(dataset.num_features, 32, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.num_graphs

        epoch_loss /= len(train_ds)
        tr = evaluate(model, train_loader, device)
        te = evaluate(model, test_loader, device)
        history["loss"].append(epoch_loss)
        history["train_acc"].append(tr)
        history["test_acc"].append(te)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"train {tr:.3f} | test {te:.3f}")

    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.3f} "
          f"(best {max(history['test_acc']):.3f})")

    # --- visualizations ---
    plot_training_curves(history, "03_gin_curves.png",
                         title="MUTAG graph classification — GIN")

    # collect per-graph embeddings across the whole dataset for t-SNE
    full_loader = DataLoader(dataset, batch_size=64)
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for data in full_loader:
            data = data.to(device)
            embs.append(model(data.x, data.edge_index, data.batch,
                              return_embedding=True).cpu())
            labels.append(data.y.cpu())
    embs = torch.cat(embs)
    labels = torch.cat(labels)
    plot_embeddings_tsne(embs, labels, "03_gin_graph_embeddings.png",
                         title="Whole-graph embeddings (GIN) — t-SNE")


if __name__ == "__main__":
    main()
