# 🎬 Video Guide — what each animation presents

Six short, narrated explainers covering the core concepts of Graph Neural Networks. Each maps to a module of this
repo. This page doubles as **YouTube metadata** (title + description ready to paste) and as the **in-repo viewing
guide**. All videos: 1080p, gTTS narration, made with [Manim](https://www.manim.community/) — source in
[`animations/`](./animations).

Suggested upload order = the numbering below (each builds on the previous one).

> **Note:** the video files themselves are not tracked in git (see `.gitignore`) — watch them on YouTube via the links below, or regenerate locally with `uv run python animations/render_all.py --quality h`. Thumbnails + paste-ready upload descriptions live in the local `youtube/` folder (`uv run python animations/make_thumbnails.py`).

---

## 1. Message Passing — the one idea behind every GNN
**File:** `assets/sample_message_passing.mp4` · ~53s · pairs with module 01
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_1 *(placeholder — updated after upload)*

**What it shows:** a center node `v` with three neighbors. The three-step loop plays out visually:
**(1) MESSAGE** — neighbors light up as they prepare messages; **(2) AGGREGATE** — orange pulses flow along the
edges into `v` and are summed (order-independent); **(3) UPDATE** — `v` changes color, showing it now holds a new
representation. Ends with the general MPNN formula `h_v' = UPDATE(h_v, Σ MSG(h_u))`.

**YouTube title:** `Message Passing in 60 Seconds — How Every Graph Neural Network Works`

**YouTube description:**
> Every GNN — GCN, GAT, GraphSAGE, GIN — is built on one loop: Message → Aggregate → Update. This animation shows
> the loop happening on an actual graph, then gives the general formula. Part 1 of a 6-part series on Graph Neural
> Networks. Code & notebooks: <REPO_URL>. Reference: Gilmer et al., "Neural Message Passing for Quantum Chemistry,"
> ICML 2017 (arXiv:1704.01212).

---

## 2. The GCN Layer, Term by Term
**File:** `assets/sample_gcn_explained.mp4` · ~53s · pairs with module 02
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_2 *(placeholder — updated after upload)*

**What it shows:** the full GCN update formula on screen; each term is highlighted in its own color while the
narration explains it — the neighbor features `h_u` (green), the sum over `N(v) ∪ {v}` with the self-loop (yellow),
the `1/√(d_u·d_v)` symmetric degree normalization that stops hub nodes from dominating (orange), and the shared
weight matrix `W` (blue). Ends with the plain-words takeaway: *a degree-normalized average of neighbors, then a
linear map.*

**YouTube title:** `The GCN Formula Explained Term by Term (Kipf & Welling, 2017)`

**YouTube description:**
> The Graph Convolutional Network update rule looks scary; it isn't. We walk through every symbol: neighbor
> features, self-loops, symmetric degree normalization, and the shared weight matrix that makes it a "convolution."
> Part 2 of the GNN series. Code: <REPO_URL>. Paper: Kipf & Welling, ICLR 2017 (arXiv:1609.02907).

---

## 3. GAT — Attention on Graphs
**File:** `assets/sample_gat_attention.mp4` · ~85s · pairs with module 02
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_3 *(placeholder — updated after upload)*

**What it shows:** the same node-with-three-neighbors setup, but now the **edges visibly thicken and thin** to their
learned attention scores — α = 0.65 (listen closely, thick orange), 0.25 (medium, yellow), 0.10 (mostly ignored,
thin grey). Shows that the scores softmax to 1, builds the attention formula
`α_vu = softmax(LeakyReLU(aᵀ[Wh_v ‖ Wh_u]))`, the weighted-sum update, and finishes with multi-head attention.
Opens by contrasting with GCN's fixed degree-based weights.

**YouTube title:** `Graph Attention Networks (GAT) — Learning Which Neighbors Matter`

**YouTube description:**
> GCN treats every neighbor the same. GAT learns an attention weight per edge — watch the edges physically thicken
> as the network decides which neighbors to listen to. Covers the attention formula, softmax normalization, and
> multi-head attention. Part 3 of the GNN series. Code: <REPO_URL>. Paper: Veličković et al., ICLR 2018
> (arXiv:1710.10903).

---

## 4. Batching Graphs — the Disconnected-Graph Trick
**File:** `assets/sample_graph_batching.mp4` · ~69s · pairs with module 03
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_4 *(placeholder — updated after upload)*

**What it shows:** three differently-sized graphs (blue triangle, green pair, orange square) **slide together into
one big disconnected graph** inside a "Batch" box — PyG's actual batching mechanism. The color-coded batch vector
`batch = [0,0,0, 1,1, 2,2,2,2]` appears, mapping every node to its graph. Then **pooling**: each graph's nodes
physically collapse into a single embedding dot (z₀, z₁, z₂) via `global_add_pool` — three graphs in, three
vectors out.

**YouTube title:** `How PyTorch Geometric Batches Graphs (and Why Pooling Works)`

**YouTube description:**
> You can't stack graphs of different sizes like images — so PyG glues them into one big disconnected graph and
> tracks node ownership with a batch vector. This animation shows the whole pipeline, ending with global pooling
> that turns node embeddings into one vector per graph. Part 4 of the GNN series. Code: <REPO_URL>.

---

## 5. Why GIN Uses SUM — the Weisfeiler–Lehman Connection
**File:** `assets/sample_gin_wl_test.mp4` · ~82s · pairs with module 03
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_5 *(placeholder — updated after upload)*

**What it shows:** two star graphs — node `v` with **2** neighbors and node `w` with **3**, every neighbor feature
equal to 1. With **mean** aggregation both give exactly 1 → *"identical → MEAN cannot tell them apart!"* (red).
With **sum**: 2 ≠ 3 → distinguishable (green). Then the theory: message-passing GNNs are bounded by the
Weisfeiler–Lehman isomorphism test, and the GIN layer `h_v' = MLP((1+ε)h_v + Σ h_u)` reaches that bound.

**YouTube title:** `Why Sum Beats Mean: GIN and the Weisfeiler–Lehman Test`

**YouTube description:**
> A 2-neighbor node and a 3-neighbor node can be *provably indistinguishable* to a mean-aggregating GNN. This
> animation shows the counterexample, then explains the expressiveness ceiling (the WL test) and the GIN
> architecture that reaches it. Part 5 of the GNN series. Code: <REPO_URL>. Paper: Xu et al., "How Powerful are
> Graph Neural Networks?", ICLR 2019 (arXiv:1810.00826).

---

## 6. Over-smoothing — Why Deep GNNs Fail
**File:** `assets/sample_over_smoothing.mp4` · ~62s · applies to every module
**▶ Watch on YouTube:** https://youtu.be/REPLACE_ME_6 *(placeholder — updated after upload)*

**What it shows:** six nodes on a hexagon, each with its own vivid color (= its embedding). Each message-passing
layer blends every node with its neighbors: layer 1 is the sweet spot (distinct but structure-aware), layer 2–3
visibly converge, and by **layer 6 all nodes are the same beige** — embeddings collapsed, nothing left to classify.
Ends with the practical rule: *keep GNNs 2–4 layers; need depth → residual connections / JumpingKnowledge.*

**YouTube title:** `Over-smoothing: The Reason GNNs Stay Shallow`

**YouTube description:**
> In CNNs, deeper is usually better. In GNNs, stacking layers makes every node average the entire graph until all
> embeddings collapse to one point. Watch it happen in color. Part 6 (finale) of the GNN series.
> Code: <REPO_URL>. Analysis: Li, Han & Wu, AAAI 2018 (arXiv:1801.07606).

---

## Re-rendering / customizing

```bash
uv run python animations/render_all.py --quality h   # 1080p, all scenes
uv run manim -qh animations/gat_attention.py GATAttention   # one scene
```

Voice is gTTS (free); to swap in another voice (ElevenLabs, Azure, offline `say`), change the
`set_speech_service(...)` line in any scene — see [`animations/README.md`](./animations/README.md).

> Replace `<REPO_URL>` in the descriptions with your GitHub link after pushing.
