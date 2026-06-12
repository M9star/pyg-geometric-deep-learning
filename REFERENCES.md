# References & Sources

Every model and idea in this repo traces back to a specific paper. This is where the methods *come from* — read the
originals; they're more readable than people expect.

## The architectures we implement

| Used in | Method | Paper | Year | Link |
|---------|--------|-------|------|------|
| 02 | **GCN** (Graph Convolutional Network) | Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* | ICLR 2017 | [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) |
| 02 | **GAT** (Graph Attention Network) | Veličković et al., *Graph Attention Networks* | ICLR 2018 | [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) |
| 02 | **GraphSAGE** | Hamilton, Ying & Leskovec, *Inductive Representation Learning on Large Graphs* | NeurIPS 2017 | [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) |
| 03 | **GIN** (Graph Isomorphism Network) | Xu et al., *How Powerful are Graph Neural Networks?* | ICLR 2019 | [arXiv:1810.00826](https://arxiv.org/abs/1810.00826) |
| 04 | **GAE / VGAE** (Graph Auto-Encoders) | Kipf & Welling, *Variational Graph Auto-Encoders* | NeurIPS BDL 2016 | [arXiv:1611.07308](https://arxiv.org/abs/1611.07308) |

## The unifying framework

| Idea | Paper | Link |
|------|-------|------|
| **Message Passing Neural Networks (MPNN)** — the template all of the above fit into | Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017 | [arXiv:1704.01212](https://arxiv.org/abs/1704.01212) |
| **Geometric Deep Learning** — the "why" (symmetry, invariance, the bigger picture) | Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*, 2021 | [arXiv:2104.13478](https://arxiv.org/abs/2104.13478) · [site](https://geometricdeeplearning.com/) |
| **Weisfeiler–Lehman test** — the theory bound GIN matches | Weisfeiler & Leman (1968); modern treatment in the GIN paper above | — |

## The library

| Tool | Reference | Link |
|------|-----------|------|
| **PyTorch Geometric (PyG)** | Fey & Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*, ICLR-W 2019 | [arXiv:1903.02428](https://arxiv.org/abs/1903.02428) · [docs](https://pytorch-geometric.readthedocs.io/) |

## Datasets

| Dataset | Source |
|---------|--------|
| **Cora / CiteSeer / PubMed** | Sen et al., *Collective Classification in Network Data*, AI Magazine 2008. Planetoid splits from Yang et al., ICML 2016 ([arXiv:1603.08861](https://arxiv.org/abs/1603.08861)) |
| **MUTAG / PROTEINS / ENZYMES** (TUDatasets) | Morris et al., *TUDataset: A collection of benchmark datasets for graph classification*, 2020 ([arXiv:2007.08663](https://arxiv.org/abs/2007.08663)) · [site](https://chrsmrrs.github.io/datasets/) |

## Courses & expository writing (start here if you're new)

- **Stanford CS224W — Machine Learning with Graphs** (Jure Leskovec). Free lectures + slides: http://web.stanford.edu/class/cs224w/
- **Distill — A Gentle Introduction to Graph Neural Networks** (Sanchez-Lengeling et al., 2021): https://distill.pub/2021/gnn-intro/
- **Distill — Understanding Convolutions on Graphs** (Daigavane et al., 2021): https://distill.pub/2021/understanding-gnns/
- **PyG official Colab notebooks**: https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html

## How to cite this learning repo
This repo is educational and reimplements the methods above; please cite the **original papers**, not this repo.
