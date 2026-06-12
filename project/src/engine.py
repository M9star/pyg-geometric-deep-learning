"""
Train / evaluate loops for graph classification.
"""

import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy over a loader."""
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.num_graphs
    return correct / total


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (y_true, y_pred) numpy arrays for the whole loader."""
    model.eval()
    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
        ys.append(data.y.cpu())
        preds.append(pred.cpu())
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def collect_embeddings(model, loader, device):
    """Return (graph_embeddings, labels) for t-SNE."""
    model.eval()
    embs, ys = [], []
    for data in loader:
        data = data.to(device)
        embs.append(model(data.x, data.edge_index, data.batch,
                          return_embedding=True).cpu())
        ys.append(data.y.cpu())
    return torch.cat(embs), torch.cat(ys)
