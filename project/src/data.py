"""
Data loading & splitting for graph classification.
"""

import os

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Cache datasets next to the project so re-runs don't re-download.
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_dataset(name: str):
    """
    Load a TUDataset. Some social datasets (e.g. IMDB-BINARY) have no node
    features; in that case we synthesize them from node degree so GNNs still work.
    """
    # First load without transform to detect missing features.
    probe = TUDataset(root=DATA_ROOT, name=name)
    transform = None
    if probe.num_node_features == 0:
        # OneHotDegree needs a max-degree bound; pick a safe cap.
        transform = T.OneHotDegree(max_degree=135)
    dataset = TUDataset(root=DATA_ROOT, name=name, transform=transform).shuffle()
    return dataset


def make_loaders(dataset, batch_size: int = 32, seed: int = 42):
    """80/10/10 train/val/test split -> three DataLoaders."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    n = len(dataset)
    n_train, n_val = int(0.8 * n), int(0.1 * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]
    test_ds = dataset[test_idx]

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )
