"""
01 · The graph Data object
==========================
Build a tiny social-network-style graph BY HAND and learn how PyG stores it.

    python 01_fundamentals/01_graph_data_structure.py

Outputs a drawing to assets/01_hand_built_graph.png
"""

import os
import sys

import torch
from torch_geometric.data import Data

# Make the shared utils importable when running this file directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import draw_graph  # noqa: E402


def build_graph() -> Data:
    """A 5-person friendship graph. Each person has 3 features."""

    # Node features: [age(normalized), is_student, hours_online] for 5 people.
    x = torch.tensor([
        [0.2, 1, 0.8],   # 0: young student, very online
        [0.5, 0, 0.3],   # 1
        [0.3, 1, 0.6],   # 2: student
        [0.8, 0, 0.1],   # 3: older, rarely online
        [0.4, 0, 0.5],   # 4
    ], dtype=torch.float)

    # Friendships (undirected -> we list BOTH directions for every edge).
    # 0-1, 0-2, 1-2, 2-3, 3-4
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    undirected = edges + [(b, a) for a, b in edges]
    edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()

    # A node-level label: e.g. community membership (0 or 1).
    y = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def inspect(data: Data) -> None:
    print("=" * 60)
    print("The Data object:", data)
    print("=" * 60)
    print(f"num_nodes        : {data.num_nodes}")
    print(f"num_edges        : {data.num_edges}   (each undirected edge counted twice)")
    print(f"num_node_features: {data.num_node_features}")
    print(f"has self-loops   : {data.has_self_loops()}")
    print(f"is undirected    : {data.is_undirected()}")
    print()
    print("x (node feature matrix):")
    print(data.x)
    print()
    print("edge_index  [2, num_edges] — row 0 = source, row 1 = target:")
    print(data.edge_index)
    print()
    print("y (node labels):", data.y.tolist())

    # Degree of each node = how many edges touch it.
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    print("node degrees     :", deg.tolist())


def main() -> None:
    data = build_graph()
    inspect(data)
    draw_graph(
        data.edge_index,
        filename="01_hand_built_graph.png",
        node_color=data.y,
        num_nodes=data.num_nodes,
        title="Hand-built friendship graph (color = community label)",
    )
    print("\nDone. Open assets/01_hand_built_graph.png to see your graph.")


if __name__ == "__main__":
    main()
