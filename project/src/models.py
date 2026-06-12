"""
GNN architectures for graph-level classification.

All models share the same shape:
    [stack of message-passing layers] -> global pooling (readout) -> MLP head
They differ only in the convolution used — exactly the lesson from modules 01-03.
"""

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import (GATConv, GCNConv, GINConv, global_add_pool,
                                 global_mean_pool)


class GraphClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden, num_classes, kind="gin",
                 num_layers=3, heads=4):
        super().__init__()
        self.kind = kind
        self.convs = torch.nn.ModuleList()

        if kind == "gin":
            dims = [in_dim] + [hidden] * num_layers
            for i in range(num_layers):
                mlp = Sequential(
                    Linear(dims[i], hidden), BatchNorm1d(hidden), ReLU(),
                    Linear(hidden, hidden), ReLU(),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            self.pool = global_add_pool
            out_dim = hidden

        elif kind == "gcn":
            dims = [in_dim] + [hidden] * num_layers
            for i in range(num_layers):
                self.convs.append(GCNConv(dims[i], hidden))
            self.pool = global_mean_pool
            out_dim = hidden

        elif kind == "gat":
            self.convs.append(GATConv(in_dim, hidden, heads=heads))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
            self.pool = global_mean_pool
            out_dim = hidden * heads

        else:
            raise ValueError(f"unknown model kind: {kind}")

        self.lin1 = Linear(out_dim, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, return_embedding=False):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        graph_emb = self.pool(x, batch)        # readout: one vector per graph
        if return_embedding:
            return graph_emb
        h = F.relu(self.lin1(graph_emb))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)
