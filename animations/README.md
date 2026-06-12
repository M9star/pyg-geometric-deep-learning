# 🎬 Animations (narrated, with Manim)

Short animated explainers for the trickiest ideas, built with
[Manim](https://www.manim.community/) and narrated using **gTTS** (Google Text-to-Speech — free, needs an internet
connection at render time). The narration text lives right next to the visuals in each scene, so the audio and the
animation stay in sync.

> **Note on the voice:** the original request mentioned `voiceinthemachine.com` for narration — that turned out to be
> a *blog* about conversational AI, not a TTS service, so it can't generate speech. We use gTTS instead, wired through
> [`manim-voiceover`](https://voiceover.manim.community/). To swap in a different voice (Azure, ElevenLabs, OpenAI,
> or offline macOS `say` via `pyttsx3`), change the `set_speech_service(...)` line in each scene — that's the only edit.

## Scenes

| File | Scene class | Explains |
|------|-------------|----------|
| `message_passing.py` | `MessagePassing` | The Message → Aggregate → Update loop (the core of every GNN) |
| `gcn_explained.py` | `GCNExplained` | The GCN formula, term by term, incl. symmetric normalization |
| `gat_attention.py` | `GATAttention` | GAT: learned attention weights per edge (vs. GCN's fixed degree weights), softmax, multi-head |
| `graph_batching.py` | `GraphBatching` | Mini-batching graphs: the "one big disconnected graph" trick, the `batch` vector, global pooling |
| `gin_wl_test.py` | `GINWLTest` | Why GIN uses SUM: a neighbourhood mean can't distinguish but sum can; the Weisfeiler–Lehman bound |
| `over_smoothing.py` | `OverSmoothing` | Over-smoothing: node embeddings (as colors) collapsing as layers stack — why GNNs stay 2–4 layers deep |

## Render

Requirements (already in `pyproject.toml`): `manim`, `manim-voiceover[gtts]`, plus system **ffmpeg** and a **LaTeX**
distribution (for the math). Then:

```bash
# one scene, fast/low quality (good enough to learn)
uv run manim -ql animations/message_passing.py MessagePassing

# high quality
uv run manim -qh animations/gcn_explained.py GCNExplained

# render everything at once
uv run python animations/render_all.py
```

Output videos land in `media/videos/<scene>/<quality>/`. The `media/` folder is git-ignored (videos are large) —
**except** the one pre-rendered sample in `assets/` so you can see what to expect without rendering anything.

## Quality flags
| Flag | Resolution | Use |
|------|-----------|-----|
| `-ql` | 480p | quick iteration / learning |
| `-qm` | 720p | |
| `-qh` | 1080p | sharing |
| `-qk` | 4K | overkill |
