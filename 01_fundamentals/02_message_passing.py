"""
02 · Message passing FROM SCRATCH
=================================
Implement one Graph Convolution layer ourselves using PyG's MessagePassing
base class, then confirm it matches the official GCNConv.

    python 01_fundamentals/02_message_passing.py

This is the single most important concept in graph ML. Read the comments
carefully — every other architecture is a variation on this template.

The GCN layer (Kipf & Welling, 2017) computes, for each node v:

    h_v' = W · sum_{u in N(v) ∪ {v}}  ( 1 / sqrt(deg(u)·deg(v)) ) · h_u

i.e. a *symmetric-normalized* sum of neighbor features, then a linear map.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MyGCNConv(MessagePassing):
    """A from-scratch re-implementation of GCNConv."""

    def __init__(self, in_channels: int, out_channels: int):
        # aggr="add" => the AGGREGATE step sums incoming messages.
        super().__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # Step 1: add self-loops so each node also keeps its own features.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: linearly transform node features.  h = x · W
        x = self.lin(x)

        # Step 3: compute symmetric normalization coefficient per edge.
        #         norm(u->v) = 1 / sqrt(deg(u) * deg(v))
        row, col = edge_index            # row = source, col = target
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: start propagating. This internally calls message() then
        #         aggregates (sum) into each target node.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j = features of the SOURCE node of each edge (the neighbor).
        # Scale each message by its normalization coefficient.
        return norm.view(-1, 1) * x_j


def main() -> None:
    torch.manual_seed(0)

    # A small graph: 4 nodes, a little path + a branch.
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)

    mine = MyGCNConv(3, 2)
    official = GCNConv(3, 2, bias=False)

    # Copy the official layer's weights into ours so we can compare outputs.
    with torch.no_grad():
        mine.lin.weight.copy_(official.lin.weight)

    out_mine = mine(x, edge_index)
    out_official = official(x, edge_index)

    print("Our  MyGCNConv output:\n", out_mine)
    print("\nOfficial GCNConv output:\n", out_official)

    close = torch.allclose(out_mine, out_official, atol=1e-5)
    diff = (out_mine - out_official).abs().max().item()
    print(f"\nMax abs difference: {diff:.2e}")
    print("MATCH ✔  — our from-scratch layer reproduces GCNConv!"
          if close else "Mismatch — check the implementation.")

    # --- Bonus: demonstrate permutation invariance of aggregation ---
    print("\n" + "-" * 60)
    print("Permutation check: relabel nodes -> embeddings just get relabeled,")
    print("the *set* of outputs is unchanged (GNNs respect graph symmetry).")
    perm = torch.tensor([2, 0, 3, 1])           # a node relabeling
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(4)
    x_perm = x[perm]
    ei_perm = inv[edge_index]                   # remap edge endpoints
    out_perm = mine(x_perm, ei_perm)
    # Undo the permutation on the output and compare.
    same = torch.allclose(out_perm[inv], out_mine, atol=1e-5)
    print("Equivariant under relabeling:", same)


if __name__ == "__main__":
    main()
