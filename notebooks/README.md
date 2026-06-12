# 📓 Notebooks — stepwise, run-as-you-learn

Interactive versions of the curriculum. Each notebook is **self-contained**, mixes **rendered math + notes** with
**runnable code**, and ends with "what to notice" + exercises. They reuse the datasets already downloaded by the
matching modules (no duplicate downloads).

| Notebook | Mirrors module | Learn |
|----------|----------------|-------|
| `01_fundamentals.ipynb` | 01 | graphs, `edge_index`, the `Data` object, message passing from scratch |
| `02_node_classification.ipynb` | 02 | GCN on Cora, training loop, t-SNE of embeddings |
| `03_graph_classification.ipynb` | 03 | GIN, graph mini-batching, global pooling |
| `04_link_prediction.ipynb` | 04 | Graph Autoencoder, AUC/AP |

## Run
```bash
uv run jupyter lab        # then open a notebook and run top-to-bottom
```
The first cell auto-detects your device (CUDA → MPS → CPU) and makes the repo's `utils/` importable, so notebooks
work whether you launch Jupyter from the repo root or from this folder.

## Regenerating
These `.ipynb` files are generated from `_build_notebooks.py` for consistency. To change them in bulk, edit that
script and run:
```bash
uv run python notebooks/_build_notebooks.py
```
(You can also just edit the `.ipynb` directly in Jupyter — both work.)

## Want the animated explainer?
The message-passing idea in notebook 01 has a narrated video version — see [`../animations`](../animations).
