"""
Generator for the learning notebooks.
=====================================
We build the .ipynb files programmatically with nbformat so they stay
consistent and easy to regenerate.

    uv run python notebooks/_build_notebooks.py

This (re)creates 01–04 .ipynb in this folder. Edit here, not the .ipynb,
if you want to change them in bulk.
"""

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared first cell: make the repo root importable + enable MPS fallback.
SETUP = """\
import sys, os
ROOT = os.getcwd()
if os.path.basename(ROOT) == "notebooks":
    ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safe Apple-GPU fallback

import torch
from utils.device import get_device, device_report
device = get_device()
print(device_report())
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = os.path.join(HERE, name)
    nbf.write(nb, path)
    print("wrote", os.path.relpath(path))


# ======================================================================
# 01 · Fundamentals
# ======================================================================
def build_01():
    cells = [
        md(r"""# 01 · Fundamentals — Graphs & Message Passing

> **Run me top to bottom.** Each cell builds on the last.
> 🎬 **Watch first:** [Message Passing — how every GNN works](https://youtu.be/eZ8AdilDccQ) ·
> bonus: [Over-smoothing — why deep GNNs fail](https://youtu.be/Xd7QVCdCr5I)

## What is a graph, in code?
A graph is **nodes** connected by **edges**. For learning we attach:
- a feature vector to each node → matrix $X \in \mathbb{R}^{N \times F}$ (`x`)
- a target $y$ (per node, or one per graph)

Connectivity is stored **sparsely** as an *edge list* `edge_index` of shape $[2, E]$ — not a dense $N\times N$
adjacency matrix. Row 0 = source nodes, row 1 = target nodes. Undirected edges are stored **both ways**.

**Source for the framework:** Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017
([arXiv:1704.01212](https://arxiv.org/abs/1704.01212))."""),
        code(SETUP),
        md("## Build a tiny graph by hand and inspect the `Data` object"),
        code("""\
from torch_geometric.data import Data

# 5 people, 3 features each: [age(norm), is_student, hours_online]
x = torch.tensor([[0.2,1,0.8],[0.5,0,0.3],[0.3,1,0.6],[0.8,0,0.1],[0.4,0,0.5]], dtype=torch.float)
edges = [(0,1),(0,2),(1,2),(2,3),(3,4)]
undirected = edges + [(b,a) for a,b in edges]          # store both directions
edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()
y = torch.tensor([0,0,0,1,1])                            # community label

data = Data(x=x, edge_index=edge_index, y=y)
print(data)
print("nodes:", data.num_nodes, "| edges:", data.num_edges, "| features:", data.num_node_features)
print("undirected:", data.is_undirected())"""),
        md("## Draw it"),
        code("""\
import networkx as nx, matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(5,5))
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), node_color=y.tolist(),
                 cmap="tab10", node_size=600, font_color="white")
plt.title("Hand-built friendship graph (color = community)"); plt.axis("off"); plt.show()"""),
        md(r"""## The core idea: Message Passing
One GNN layer updates each node $v$ from its neighbours $N(v)$:

$$ h_v' = \text{UPDATE}\Big(h_v,\ \underbrace{\textstyle\bigoplus_{u \in N(v)} \text{MESSAGE}(h_u)}_{\text{permutation-invariant aggregation}}\Big) $$

The aggregation $\bigoplus$ (sum / mean / max) must **not depend on neighbour order** — that's what makes a GNN
respect the graph's symmetry. Stacking $k$ layers lets information travel $k$ hops.

Below we implement **one GCN layer from scratch** and check it against PyG's official `GCNConv`. The GCN rule is:

$$ h_v' = W \!\!\sum_{u \in N(v)\cup\{v\}} \frac{1}{\sqrt{d_u\, d_v}}\, h_u $$

**Source:** Kipf & Welling, ICLR 2017 ([arXiv:1609.02907](https://arxiv.org/abs/1609.02907))."""),
        code("""\
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree

class MyGCNConv(MessagePassing):
    def __init__(self, in_c, out_c):
        super().__init__(aggr="add")            # AGGREGATE = sum
        self.lin = Linear(in_c, out_c, bias=False)
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)                          # MESSAGE prep: h = xW
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        dis = deg.pow(-0.5); dis[dis == float("inf")] = 0
        norm = dis[row] * dis[col]               # 1/sqrt(d_u d_v)
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j             # scaled neighbour features

mine = MyGCNConv(3, 2); official = GCNConv(3, 2, bias=False)
with torch.no_grad(): mine.lin.weight.copy_(official.lin.weight)
out_mine, out_off = mine(x, edge_index), official(x, edge_index)
print("max abs diff:", (out_mine - out_off).abs().max().item())
print("MATCH" if torch.allclose(out_mine, out_off, atol=1e-5) else "mismatch")"""),
        md("""## ✅ Takeaways
- Graphs = `x` + `edge_index` (+ optional `y`), stored sparsely.
- One GNN layer = **Message → Aggregate → Update**; aggregation is permutation-invariant.
- Depth = number of hops; too deep → *over-smoothing* (embeddings collapse).

**Exercises:** (1) add a self-loop check; (2) swap `aggr="add"`→`"mean"` and compare; (3) make the graph directed
and watch `is_undirected()` change.

➡ Next: **02 · Node Classification**."""),
    ]
    write("01_fundamentals.ipynb", cells)


# ======================================================================
# 02 · Node classification
# ======================================================================
def build_02():
    cells = [
        md(r"""# 02 · Node Classification — GCN on Cora

> 🎬 **Watch first:** [The GCN formula, term by term](https://youtu.be/eqmp7V0I0cw) ·
> [GAT — which neighbors matter](https://youtu.be/0EdsNkNlaZo)

**Task:** given a citation graph where only ~140 of 2,708 papers are labelled, predict every paper's topic.
This is **transductive** learning — the whole graph (features + edges) is visible; only test *labels* are hidden.

**Models & sources**
- **GCN** — Kipf & Welling, ICLR 2017 ([arXiv:1609.02907](https://arxiv.org/abs/1609.02907))
- **GAT** (attention over neighbours) — Veličković et al., ICLR 2018 ([arXiv:1710.10903](https://arxiv.org/abs/1710.10903))
- **GraphSAGE** (inductive, sampling) — Hamilton et al., NeurIPS 2017 ([arXiv:1706.02216](https://arxiv.org/abs/1706.02216))
- **Cora** dataset — Planetoid splits, Yang et al., ICML 2016 ([arXiv:1603.08861](https://arxiv.org/abs/1603.08861))"""),
        code(SETUP),
        code("""\
from torch_geometric.datasets import Planetoid
# reuse the dataset already downloaded by the 02 module (no re-download)
dataset = Planetoid(root=os.path.join(ROOT, "02_node_classification", "data", "Cora"), name="Cora")
data = dataset[0].to(device)
print(f"{data.num_nodes} nodes | {data.num_edges} edges | "
      f"{dataset.num_features} features | {dataset.num_classes} classes")
print("labelled for training:", int(data.train_mask.sum()), "nodes")"""),
        md(r"""## A 2-layer GCN
Two layers = each node sees its 2-hop neighbourhood. We use `log_softmax` + `nll_loss`, dropout for regularization,
and select the model by **validation** accuracy (reporting test accuracy at the best val epoch)."""),
        code("""\
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
    def forward(self, x, edge_index, return_emb=False):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        emb = self.conv2(h, edge_index)
        return emb if return_emb else F.log_softmax(emb, dim=1)

model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def acc(logits, mask): return (logits[mask].argmax(1) == data.y[mask]).float().mean().item()

hist = {"loss": [], "val": [], "test": []}; best_val = best_test = 0
for epoch in range(1, 201):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): logits = model(data.x, data.edge_index)
    va, te = acc(logits, data.val_mask), acc(logits, data.test_mask)
    hist["loss"].append(loss.item()); hist["val"].append(va); hist["test"].append(te)
    if va > best_val: best_val, best_test = va, te
    if epoch % 40 == 0: print(f"epoch {epoch:3d} | loss {loss:.3f} | val {va:.3f} | test {te:.3f}")
print(f"\\nBest val {best_val:.3f} -> test {best_test:.3f}")"""),
        md("## Training curves"),
        code("""\
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11,4))
ax[0].plot(hist["loss"], color="crimson"); ax[0].set_title("loss"); ax[0].grid(alpha=.3)
ax[1].plot(hist["val"], label="val"); ax[1].plot(hist["test"], label="test")
ax[1].set_title("accuracy"); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].grid(alpha=.3); plt.show()"""),
        md("""## The payoff: do the learned embeddings separate the 7 topics?
We project the final-layer node embeddings to 2D with **t-SNE**. Before training this is a blob; after training it's
7 clean islands — visual proof the GNN learned meaningful representations from graph structure + text features."""),
        code("""\
from sklearn.manifold import TSNE
model.eval()
with torch.no_grad():
    emb = model(data.x, data.edge_index, return_emb=True).cpu().numpy()
proj = TSNE(n_components=2, init="pca", random_state=42).fit_transform(emb)
plt.figure(figsize=(7,6))
sc = plt.scatter(proj[:,0], proj[:,1], c=data.y.cpu(), cmap="tab10", s=12)
plt.legend(*sc.legend_elements(), title="topic", fontsize=8); plt.xticks([]); plt.yticks([])
plt.title("Learned node embeddings (t-SNE)"); plt.show()"""),
        md("""## ✅ Notes & exercises
- Expect **~80% test accuracy**. Try editing the model to `GATConv`/`SAGEConv` and compare.
- Add a third `GCNConv` layer → accuracy often *drops* (over-smoothing).
- The CLI version with GAT/SAGE + saved plots: `python 02_node_classification/train.py --model gat`.

➡ Next: **03 · Graph Classification**."""),
    ]
    write("02_node_classification.ipynb", cells)


# ======================================================================
# 03 · Graph classification
# ======================================================================
def build_03():
    cells = [
        md(r"""# 03 · Graph Classification — GIN on MUTAG

> 🎬 **Watch first:** [Batching graphs in PyG](https://youtu.be/bD7IDjEJ16M) ·
> [Why sum beats mean — GIN & the WL test](https://youtu.be/sJ-m8AIJC-s)

**Task:** classify *whole graphs*. Each graph is a molecule; predict whether it's mutagenic. Two new ideas vs.
node classification:

1. **Mini-batching graphs** — many small graphs are glued into one big *disconnected* graph; a `batch` vector
   records which graph each node belongs to. `DataLoader` does this automatically.
2. **Readout / global pooling** — collapse node embeddings into **one vector per graph** (`global_add_pool`).

**Model — GIN.** The most *expressive* message-passing GNN: its power matches the **Weisfeiler–Lehman** graph
isomorphism test. Update rule:

$$ h_v' = \text{MLP}\Big( (1+\epsilon)\, h_v + \sum_{u\in N(v)} h_u \Big) $$

**Source:** Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019
([arXiv:1810.00826](https://arxiv.org/abs/1810.00826)). Dataset: Morris et al., TUDataset 2020
([arXiv:2007.08663](https://arxiv.org/abs/2007.08663))."""),
        code(SETUP),
        code("""\
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root=os.path.join(ROOT, "03_graph_classification", "data"), name="MUTAG").shuffle()
print(len(dataset), "graphs |", dataset.num_features, "node features |", dataset.num_classes, "classes")
n = int(len(dataset)*0.8)
train_loader = DataLoader(dataset[:n], batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset[n:], batch_size=32)

b = next(iter(train_loader))
print(f"one batch -> {b.num_graphs} graphs glued into {b.num_nodes} nodes; batch vec {tuple(b.batch.shape)}")"""),
        code("""\
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_cls, layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        dims = [in_dim] + [hidden]*layers
        for i in range(layers):
            mlp = Sequential(Linear(dims[i], hidden), BatchNorm1d(hidden), ReLU(),
                             Linear(hidden, hidden), ReLU())
            self.convs.append(GINConv(mlp, train_eps=True))
        self.lin1 = Linear(hidden, hidden); self.lin2 = Linear(hidden, n_cls)
    def forward(self, x, edge_index, batch):
        for c in self.convs: x = c(x, edge_index)
        x = global_add_pool(x, batch)            # READOUT: one vector per graph
        x = F.relu(self.lin1(x)); x = F.dropout(x, 0.5, training=self.training)
        return self.lin2(x)

model = GIN(dataset.num_features, 32, dataset.num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

@torch.no_grad()
def test(loader):
    model.eval(); correct = total = 0
    for d in loader:
        d = d.to(device)
        correct += int((model(d.x, d.edge_index, d.batch).argmax(1) == d.y).sum()); total += d.num_graphs
    return correct/total

for epoch in range(1, 101):
    model.train()
    for d in train_loader:
        d = d.to(device); opt.zero_grad()
        F.cross_entropy(model(d.x, d.edge_index, d.batch), d.y).backward(); opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | train {test(train_loader):.3f} | test {test(test_loader):.3f}")
print("final test acc:", round(test(test_loader), 3))"""),
        md("""## ✅ Notes
- MUTAG has only 188 graphs, so accuracy is noisy run-to-run (expect ~75–90%).
- Swap `global_add_pool` → `global_mean_pool` and compare; **sum** preserves more info (GIN's choice).
- CLI version with confusion matrix + checkpoints: `python project/main.py train --dataset MUTAG --model gin`.

➡ Next: **04 · Link Prediction**."""),
    ]
    write("03_graph_classification.ipynb", cells)


# ======================================================================
# 04 · Link prediction
# ======================================================================
def build_04():
    cells = [
        md(r"""# 04 · Link Prediction — Graph Autoencoder (GAE)

> 🎬 The full animated series (message passing → over-smoothing) is linked in the
> [main README](../README.md)'s video table — worth a watch before this final module.

**Task:** predict *missing* edges ("who should connect to whom?"). This is **self-supervised** — labels come from
the graph itself. Hide some real edges, train on the rest, then score all node pairs.

**Model — GAE.** An encoder (GCN) maps nodes to embeddings $z_v$; the decoder is parameter-free — the probability
of an edge is the **dot product** of embeddings:

$$ p(u\!\sim\! v) = \sigma(z_u^\top z_v) $$

We optimise binary cross-entropy over real (positive) vs. sampled (negative) edges, and measure **AUC** and
**Average Precision**.

**Source:** Kipf & Welling, *Variational Graph Auto-Encoders*, NeurIPS BDL 2016
([arXiv:1611.07308](https://arxiv.org/abs/1611.07308))."""),
        code(SETUP),
        code("""\
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, GCNConv

tf = T.Compose([T.NormalizeFeatures(), T.ToDevice(device),
                T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                  split_labels=True, add_negative_train_samples=False)])
ds = Planetoid(os.path.join(ROOT, "04_link_prediction", "data", "Cora"), name="Cora", transform=tf)
train_data, val_data, test_data = ds[0]
print("train edges:", train_data.edge_index.size(1), "| test pos edges:", test_data.pos_edge_label_index.size(1))

class Encoder(torch.nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.c1 = GCNConv(i, h); self.c2 = GCNConv(h, o)
    def forward(self, x, ei): return self.c2(self.c1(x, ei).relu(), ei)

model = GAE(Encoder(ds.num_features, 32, 16)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

@torch.no_grad()
def evaluate(d):
    model.eval(); z = model.encode(d.x, d.edge_index)
    return model.test(z, d.pos_edge_label_index, d.neg_edge_label_index)

for epoch in range(1, 201):
    model.train(); opt.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss.backward(); opt.step()
    if epoch % 40 == 0:
        auc, ap = evaluate(val_data); print(f"epoch {epoch:3d} | loss {loss:.3f} | val AUC {auc:.3f} AP {ap:.3f}")
auc, ap = evaluate(test_data); print(f"\\nTEST AUC {auc:.3f} | AP {ap:.3f}")"""),
        md("""## ✅ Notes
- Expect **AUC ~0.90+**. We never used node *labels* — structure alone carries the signal.
- The learned embeddings still cluster by topic; colour a t-SNE of `model.encode(...)` by `Planetoid(...).y` to see it.
- Try PyG's **VGAE** (variational version) for probabilistic embeddings.

➡ You've covered node-, graph-, and edge-level tasks. Build something in `project/`!"""),
    ]
    write("04_link_prediction.ipynb", cells)


if __name__ == "__main__":
    build_01(); build_02(); build_03(); build_04()
    print("\nDone. Open the .ipynb files in Jupyter:  uv run jupyter lab")
