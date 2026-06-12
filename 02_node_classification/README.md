# 02 · Node Classification — GCN / GAT / GraphSAGE on Cora

> **Goal:** given a citation network where most papers are unlabeled, predict the topic of every paper.

## The dataset: Cora
- **2,708 nodes** = scientific papers
- **edges** = citations between papers
- **node features** = a 1,433-dim bag-of-words of the paper's abstract
- **7 classes** = research topics
- Only a *handful* of nodes per class are labeled (140 total) — yet we classify all 2,708.

This is **transductive** learning: the whole graph (including test nodes' *features* and *edges*) is visible during
training; only their *labels* are hidden. PyG gives us boolean masks `train_mask` / `val_mask` / `test_mask`.

## The models (all just stacks of message-passing layers)

| Model | MESSAGE / AGGREGATE idea | One-line intuition |
|-------|--------------------------|--------------------|
| **GCN** | symmetric-normalized **sum** of neighbors | "average your neighbors, weighted by degree" |
| **GraphSAGE** | concat(self, **mean** of neighbors) | "learn from a sampled neighborhood; works on unseen nodes" |
| **GAT** | **attention-weighted** sum | "learn *how much* to listen to each neighbor" |

All three are 2-layer networks here: `Conv → ReLU → Dropout → Conv → log-softmax`.

## Why 2 layers?
Each layer = 1 hop. Two layers let a node see its neighbors-of-neighbors — enough signal for Cora. Going deeper
often *hurts* (over-smoothing: all embeddings converge). Try editing `hidden_channels`/adding layers to see it.

## Run it

```bash
# default is GCN
python 02_node_classification/train.py

# try the others
python 02_node_classification/train.py --model gat
python 02_node_classification/train.py --model sage
```

Outputs:
- console: per-epoch loss + train/val/test accuracy (expect **~80% test acc** for GCN)
- `assets/02_<model>_curves.png` — training curves
- `assets/02_<model>_tsne.png` — t-SNE of the learned node embeddings, colored by true class.
  **This is the payoff:** watch the 7 topics separate into clean clusters — visual proof the GNN learned
  meaningful representations from graph structure + text features.

## What to notice
- GAT and GCN usually land within a point or two of each other on Cora.
- The t-SNE plot of raw features (before training) is a blob; after training it's 7 islands.
- Validation accuracy is used for **early model selection** — we report the test accuracy at the best val epoch.
