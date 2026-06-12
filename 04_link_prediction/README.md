# 04 · Link Prediction — Graph Autoencoder (GAE)

> **Goal:** predict *missing* edges. "Which two people are likely to become friends?" / "which papers should cite
> each other?" This powers recommendation and knowledge-graph completion.

## The task
Hide a fraction of the graph's real edges. Train a model on the remaining edges, then ask it to score *all* possible
node pairs and check whether it ranks the hidden ("positive") edges above random ("negative") non-edges.

This is **self-supervised**: the labels (edge exists / doesn't) come from the graph itself — no human annotation.

## The model: Graph Autoencoder (Kipf & Welling, 2016)
Two parts:

1. **Encoder** — a GCN maps each node to a low-dim embedding `z_v`.
2. **Decoder** — there are *no* learned parameters; the probability of an edge between `u` and `v` is just the
   **dot product** of their embeddings, squashed by a sigmoid:

   ```
   p(edge u–v) = σ( z_u · z_v )
   ```

The intuition: the encoder learns to place nodes that *should* be connected close together in embedding space, so
their dot product is large.

## Training & evaluation
- **Loss:** binary cross-entropy over positive edges (real) vs. sampled negative edges (random non-edges).
- **Metrics:** **AUC** and **Average Precision (AP)** on a held-out set of positive/negative edges.
  PyG's `RandomLinkSplit` builds these train/val/test edge sets for us.

## Run it

```bash
python 04_link_prediction/train.py
```

Outputs:
- console: val/test **AUC** and **AP** per epoch (expect **AUC ~0.90+** on Cora)
- `assets/04_gae_curves.png` — AUC/AP over training
- `assets/04_gae_tsne.png` — t-SNE of the learned node embeddings (colored by the papers' true class — note we never
  trained on those labels, yet structure emerges)

## What to notice
- We optimize *edges*, not labels, yet the embeddings still cluster by topic — graph structure carries the signal.
- Try **VGAE** (the variational version) for a probabilistic embedding; PyG ships both `GAE` and `VGAE`.
