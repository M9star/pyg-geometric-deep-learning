# 01 · Fundamentals — Graphs & Message Passing

> **Goal:** understand how a graph is stored in PyG, and *the* core operation of every GNN: message passing.

🎬 **Watch first:** [Message Passing — how every GNN works](https://youtu.be/eZ8AdilDccQ) ·
also relevant: [Over-smoothing — why deep GNNs fail](https://youtu.be/Xd7QVCdCr5I)

## 1. What is a graph, in code?

A graph is **nodes** (vertices) connected by **edges**. In deep learning we attach:

- a **feature vector** to every node → matrix `x` of shape `[num_nodes, num_node_features]`
- optionally features to edges → `edge_attr`
- optionally a **label** `y` (per node, or one per whole graph)

The connectivity itself is *not* stored as a dense adjacency matrix (that would be `N×N` — huge and mostly zeros).
Instead PyG uses an **edge list** called `edge_index`, a `[2, num_edges]` integer tensor:

```
edge_index = [[0, 1, 1, 2],     # source nodes
              [1, 0, 2, 1]]      # target nodes
# means edges: 0->1, 1->0, 1->2, 2->1
```

Undirected edges are stored as **two directed edges** (both directions). This sparse format is why GNNs scale to
millions of nodes.

Everything bundles into a single **`torch_geometric.data.Data`** object:

```python
from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index, y=y)
```

➡ Run **[`01_graph_data_structure.py`](./01_graph_data_structure.py)** — it builds a small graph by hand,
inspects the `Data` object, and draws it.

## 2. The core idea: Message Passing

Every modern GNN is a **Message Passing Neural Network (MPNN)**. One layer updates each node by looking at its
neighbors. For node *v*:

1. **Message:** each neighbor *u* sends a message (a function of its features).
2. **Aggregate:** combine all incoming messages with a *permutation-invariant* op (sum / mean / max).
   This is crucial — the result must not depend on neighbor ordering.
3. **Update:** combine the aggregated message with the node's own features to get its new features.

Formally, one layer is:

```
h_v^(k) = UPDATE( h_v^(k-1) ,  AGGREGATE_{u ∈ N(v)}  MESSAGE( h_u^(k-1) ) )
```

Stacking *k* layers lets information flow from *k* hops away. That's it — GCN, GAT, GraphSAGE, GIN all differ only
in *how* they define MESSAGE / AGGREGATE / UPDATE.

### Why this respects graph structure
A graph has no canonical node ordering. Because AGGREGATE is permutation-invariant, relabeling nodes doesn't change
the output — the network is **permutation equivariant**. This is the "geometric" prior that makes GNNs the right tool.

### Watch out: over-smoothing
Stack too many layers and every node ends up aggregating the whole graph → all embeddings collapse to the same
vector. In practice 2–4 layers is often best. (We demonstrate this idea in later modules.)

➡ Run **[`02_message_passing.py`](./02_message_passing.py)** — it implements one message-passing layer **from
scratch** using PyG's `MessagePassing` base class, then checks it against the official `GCNConv`.

## Key takeaways
- Graphs = `x` + `edge_index` (+ optional `edge_attr`, `y`), stored sparsely.
- One GNN layer = Message → Aggregate → Update.
- Aggregation must be permutation-invariant.
- Depth = how many hops information travels; too deep → over-smoothing.
