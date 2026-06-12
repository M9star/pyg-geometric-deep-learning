# ★ Capstone Project — Graph Classifier CLI

An end-to-end, reusable **graph classification** pipeline that ties together everything from modules 01–03 into a
proper, packaged project with a command-line interface. Train a GNN on any
[TUDataset](https://chrsmrrs.github.io/datasets/) (molecules, proteins, social graphs), evaluate it, save the model,
and generate publication-style plots.

This is the part you can show off / fork / build on.

## Features
- **Three architectures** to choose from: `gin`, `gcn`, `gat` (selectable via `--model`).
- Works on **any TUDataset** via `--dataset` (default `PROTEINS`).
- Clean train/val/test split with early model selection on validation.
- Saves a **model checkpoint** (`runs/<name>/model.pt`) and a **metrics JSON**.
- Auto-generates:
  - training curves
  - a **confusion matrix** on the test set
  - a t-SNE of learned graph embeddings
- Tidy, documented `src/` package — easy to read and extend.

## Install
From the repo root (`pyg-geometric-deep-learning/`):
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train the default model (GIN on PROTEINS)
python project/main.py train

# Pick a dataset + architecture + hyperparameters
python project/main.py train --dataset MUTAG --model gat --epochs 150 --hidden 64

# Evaluate a saved checkpoint
python project/main.py evaluate --run runs/PROTEINS_gin
```

All outputs land in `runs/<dataset>_<model>/`:
```
runs/PROTEINS_gin/
├── model.pt           # trained weights + config
├── metrics.json       # final train/val/test accuracy
├── curves.png
├── confusion_matrix.png
└── embeddings_tsne.png
```

## Suggested datasets to try
| `--dataset` | Domain | Graphs | Note |
|-------------|--------|--------|------|
| `MUTAG`     | molecules | 188 | tiny, fast |
| `PROTEINS`  | bioinformatics | 1,113 | good default |
| `ENZYMES`   | bioinformatics | 600 | 6 classes (harder) |
| `IMDB-BINARY` | social | 1,000 | no node features → uses degree |

## How it maps to the curriculum
- `src/models.py` — the GNN architectures from modules 02 & 03.
- `src/data.py` — dataset loading, splitting, batching (module 03 concepts).
- `src/engine.py` — the train / evaluate loops.
- `src/viz.py` — confusion matrix + reuse of the shared t-SNE/curves plots.
- `main.py` — the CLI that wires it together.

## Extend it (good practice exercises)
- Add a `predict` subcommand that classifies a single graph.
- Add hierarchical pooling (`TopKPooling` / `SAGPooling`) for larger graphs.
- Log to TensorBoard or Weights & Biases.
- Swap in an OGB dataset for a real benchmark.
