# Geometric Deep Learning with PyTorch Geometric (PyG)

A hands-on, from-scratch learning path for **Graph Machine Learning** and **Geometric Deep Learning** using
[PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/).

Every module pairs **theory** (the `README.md` in each folder) with **runnable code** (`.py` files) and
**visualizations** so you can *see* what graph neural networks actually learn.

> This repo is designed to be cloned and run by anyone who wants to learn GNNs. No prior graph experience required —
> only basic PyTorch and Python.

---

## Why graphs?

Most real data isn't a grid (like images) or a sequence (like text) — it's a **graph**: molecules, social networks,
road maps, citation networks, recommendation systems, knowledge graphs, 3D meshes. Geometric Deep Learning is the
field that generalizes deep learning to these irregular structures. PyG is the most widely used library for it.

---

## Learning path

Work through the modules **in order**. Each builds on the previous one.

| # | Module | What you learn | Key concept |
|---|--------|----------------|-------------|
| 0 | [`00_setup`](./00_setup) | Install & verify PyG; detect CPU/CUDA/MPS | Environment |
| 1 | [`01_fundamentals`](./01_fundamentals) | How graphs are represented; the `Data` object; message passing from scratch | `edge_index`, MPNN |
| 2 | [`02_node_classification`](./02_node_classification) | Classify nodes in a citation graph (Cora) with GCN / GAT / GraphSAGE | Transductive learning |
| 3 | [`03_graph_classification`](./03_graph_classification) | Classify whole molecules (MUTAG) with GIN + pooling | Mini-batching, readout |
| 4 | [`04_link_prediction`](./04_link_prediction) | Predict missing edges with a Graph Autoencoder | Self-supervised, embeddings |
| ★ | [`project`](./project) | **Capstone:** end-to-end molecular property prediction pipeline (CLI + plots) | Putting it together |

**Two ways to learn each topic:**
- 📓 **[`notebooks/`](./notebooks)** — run it stepwise in Jupyter, with rendered math, notes, and inline references.
- 🎬 **[`animations/`](./animations)** — narrated Manim videos of the trickiest ideas (start with *message passing*).
  See **[`VIDEOS.md`](./VIDEOS.md)** for what each video covers (also YouTube-ready titles/descriptions).
- 📚 **[`REFERENCES.md`](./REFERENCES.md)** — exact paper for every method (GCN, GAT, GraphSAGE, GIN, GAE …).

---

## What you need to learn (concept checklist)

Tick these off as you go — they are the core of graph ML:

**Representation**
- [ ] Nodes, edges, adjacency matrix vs. **edge list** (`edge_index`)
- [ ] Node features `x`, edge features `edge_attr`, graph-level targets `y`
- [ ] PyG's `Data` and `Batch` objects; sparse representation

**The core idea — Message Passing**
- [ ] Aggregate → Update: each node gathers info from neighbors
- [ ] Permutation invariance & why GNNs respect graph symmetry
- [ ] The `MessagePassing` base class (`message`, `aggregate`, `update`)

**Architectures**
- [ ] **GCN** — spectral-inspired, symmetric normalization
- [ ] **GraphSAGE** — sampling + inductive learning
- [ ] **GAT** — attention over neighbors
- [ ] **GIN** — maximally expressive, theory link to Weisfeiler–Lehman test

**Tasks**
- [ ] Node-level (classification / regression)
- [ ] Graph-level (needs **pooling / readout**)
- [ ] Edge-level (link prediction, recommendation)

**Practical**
- [ ] Mini-batching many graphs into one big disconnected graph
- [ ] Over-smoothing (why deep GNNs can get *worse*)
- [ ] Train/val/test masks vs. dataset splits
- [ ] Visualizing embeddings with t-SNE

---

## Quick start (with `uv`)

This repo uses [**uv**](https://docs.astral.sh/uv/) and is pinned to **Python 3.12** (the most compatible version
for the torch + PyG + manim stack; newer 3.13/3.14 often lack wheels).

```bash
# 1. Clone & enter
git clone <your-repo-url> && cd pyg-geometric-deep-learning

# 2. Create the environment + install everything (uv reads pyproject.toml & .python-version)
uv sync                       # makes .venv with Python 3.12 and all deps

# 3. Verify the install + see your device (CPU / CUDA / MPS)
uv run python 00_setup/check_install.py

# 4. Start learning — run the fundamentals
uv run python 01_fundamentals/01_graph_data_structure.py

# ...or open the notebooks
uv run jupyter lab
```

> Don't have `uv`? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux), or use plain
> pip: `pip install -r requirements.txt` inside a Python 3.12 virtualenv.

### Runs anywhere — CPU, NVIDIA, or Apple Silicon
Every script and notebook calls [`utils/device.py`](./utils/device.py), which auto-selects the best backend:
**CUDA → MPS (Apple Metal) → CPU**. On Apple Silicon it uses the GPU via MPS, with an automatic CPU fallback for the
few graph ops Metal doesn't support yet (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set for you).

Each script saves its plots into [`assets/`](./assets) so you have a visual record of every experiment.

---

## Resources (the "official" tutorials worth your time)

- **PyG docs & Colab notebooks** — https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
- **Stanford CS224W: Machine Learning with Graphs** — the canonical course (free lectures on YouTube)
- **Book:** *Geometric Deep Learning* (Bronstein, Bruna, Cohen, Veličković) — https://geometricdeeplearning.com/
- **Distill: A Gentle Introduction to GNNs** — https://distill.pub/2021/gnn-intro/
- **Open Graph Benchmark (OGB)** — real, large-scale datasets — https://ogb.stanford.edu/

---

## Repo layout

```
pyg-geometric-deep-learning/
├── README.md                  ← you are here
├── REFERENCES.md              ← exact paper sources for every method
├── pyproject.toml             ← uv project (Python 3.12 + all deps)
├── .python-version            ← pins Python 3.12
├── requirements.txt           ← pip fallback
├── 00_setup/                  ← install + device check
├── 01_fundamentals/           ← graphs + message passing from scratch
├── 02_node_classification/    ← GCN/GAT/SAGE on Cora
├── 03_graph_classification/   ← GIN on MUTAG molecules
├── 04_link_prediction/        ← Graph Autoencoder
├── notebooks/                 ← 📓 stepwise Jupyter notebooks (math + notes)
├── animations/                ← 🎬 narrated Manim explainers (gTTS voiceover)
├── utils/                     ← device detection + visualization helpers
├── project/                   ← capstone CLI project
└── assets/                    ← generated plots + the sample animation
```

## License

MIT — see [LICENSE](./LICENSE). Free to use, fork, and learn from.
