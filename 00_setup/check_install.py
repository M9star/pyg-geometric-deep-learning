"""
00 · Setup check
================
Run this first to confirm your environment is ready.

    python 00_setup/check_install.py

It prints the versions of everything we use and runs a tiny end-to-end
sanity test (build a graph, push it through one GCN layer). If this script
finishes with "ALL GOOD", you're ready for module 01.
"""

import sys


def main() -> None:
    print("=" * 60)
    print("PyG environment check")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]}")

    # --- torch ---
    try:
        import torch
        print(f"torch       : {torch.__version__}")
        print(f"CUDA avail  : {torch.cuda.is_available()}")
        print(f"Apple MPS   : {torch.backends.mps.is_available()}")
    except ImportError:
        sys.exit("[X] PyTorch not installed.  ->  pip install torch")

    # --- torch_geometric ---
    try:
        import torch_geometric
        print(f"torch_geom. : {torch_geometric.__version__}")
    except ImportError:
        sys.exit("[X] PyTorch Geometric not installed.  ->  pip install torch-geometric")

    # --- supporting libs ---
    for mod in ("numpy", "sklearn", "matplotlib", "networkx"):
        try:
            __import__(mod)
            print(f"{mod:<12}: OK")
        except ImportError:
            print(f"{mod:<12}: MISSING  (pip install {mod})")

    # --- tiny end-to-end test: a triangle graph through one GCN layer ---
    print("-" * 60)
    print("Running a tiny GCN forward pass...")
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    # 3 nodes, each with a 2-dim feature; edges form a triangle (undirected).
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],
                               [1, 0, 2, 1, 0, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    conv = GCNConv(in_channels=2, out_channels=4)
    out = conv(data.x, data.edge_index)
    print(f"Input  node features shape : {tuple(data.x.shape)}")
    print(f"Output node features shape : {tuple(out.shape)}  (3 nodes, 4 features)")

    assert out.shape == (3, 4), "Unexpected output shape!"
    print("=" * 60)
    print("ALL GOOD ✔  — head to 01_fundamentals next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
