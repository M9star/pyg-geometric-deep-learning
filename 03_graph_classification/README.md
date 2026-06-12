# 03 · Graph Classification — GIN on MUTAG molecules

> **Goal:** classify *entire graphs*. Each graph is a molecule; predict whether it's mutagenic.

This is a different task from module 02. There, we labeled **nodes** inside one big graph. Here we have **many small
graphs** and one label **per graph**. That changes two things: **batching** and **readout**.

## The dataset: MUTAG
- **188 molecules** (small graphs)
- nodes = atoms (with a one-hot atom-type feature), edges = chemical bonds
- binary label: mutagenic on a bacterium, or not

## New concept 1 — Mini-batching graphs
You can't stack graphs of different sizes into a tensor like images. PyG's trick: **glue a batch of graphs into one
big disconnected graph**. Node features are concatenated; `edge_index` is offset so edges never cross between graphs.
A `batch` vector records which graph each node belongs to:

```
batch = [0,0,0, 1,1, 2,2,2,2]   # 3 graphs with 3, 2, 4 nodes
```

`DataLoader` from `torch_geometric.loader` does this for you automatically.

## New concept 2 — Readout / global pooling
After message passing we have *node* embeddings, but we need *one* vector per graph. We **pool** all nodes of a graph
into a single vector using the `batch` vector:

- `global_mean_pool` — average node embeddings
- `global_add_pool` — sum (the GIN paper's choice; preserves more info)
- `global_max_pool` — element-wise max

## The model: GIN (Graph Isomorphism Network)
GIN is the **most expressive** message-passing GNN under a key theoretical result: a GNN can distinguish two graphs
*at most* as well as the **Weisfeiler–Lehman (WL) graph isomorphism test**. GIN is built to match WL's power by using
**sum** aggregation followed by an MLP:

```
h_v' = MLP( (1 + ε) · h_v  +  sum_{u ∈ N(v)} h_u )
```

The injective sum + MLP is what gives GIN its discriminative edge over mean/max-based GNNs.

## Run it

```bash
python 03_graph_classification/train.py
```

Outputs:
- console: train/test accuracy per epoch (expect **~75–85% test acc**; MUTAG is small so it's noisy)
- `assets/03_gin_curves.png` — training curves
- `assets/03_gin_graph_embeddings.png` — t-SNE of whole-**graph** embeddings, colored by mutagenicity

## What to notice
- Because MUTAG has only 188 graphs, accuracy varies run-to-run — that's normal for tiny datasets.
- Swap `global_add_pool` for `global_mean_pool` in the code and compare; sum often wins for molecules.
