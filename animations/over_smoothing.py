"""
Animation: Over-smoothing — why deep GNNs get WORSE.
====================================================
Each message-passing layer mixes a node with its neighbours. Stack too many
layers and every node ends up averaging the whole graph: all embeddings
collapse to the same point. We visualize embeddings as colors and watch
them blend into uniform grey, layer by layer.

This is why the models in this repo use only 2-4 layers.
(See e.g. Li, Han & Wu, "Deeper Insights into GCNs", AAAI 2018,
arXiv:1801.07606, for the formal analysis.)

Render:
    uv run manim -ql animations/over_smoothing.py OverSmoothing
"""

from manim import (BLUE, GREEN, GREY, ORANGE, PURPLE, RED, WHITE, YELLOW,
                   Create, Dot, FadeIn, FadeOut, Line, Text, VGroup, Write,
                   average_color, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class OverSmoothing(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("Over-smoothing: why deep GNNs fail", font_size=40).to_edge(np.array([0, 1, 0]))
        with self.voiceover(text="Deeper is better, right? Not for graph neural networks. "
                                 "This is over-smoothing."):
            self.play(Write(title))
        self.play(FadeOut(title))

        # --- a hexagon graph with a couple of chords; colors = embeddings ---
        n = 6
        radius = 2.0
        centers = [radius * np.array([np.cos(2 * np.pi * i / n + np.pi / 2),
                                      np.sin(2 * np.pi * i / n + np.pi / 2), 0])
                   + np.array([0, 0.2, 0])
                   for i in range(n)]
        edge_pairs = [(i, (i + 1) % n) for i in range(n)] + [(0, 3), (1, 4)]
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

        dots = [Dot(c, radius=0.26, color=col) for c, col in zip(centers, colors)]
        lines = [Line(centers[a], centers[b], color=GREY, stroke_width=3) for a, b in edge_pairs]

        layer_label = Text("Layer 0 — every node has its own distinct embedding (color)",
                           font_size=24).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="Think of each node's embedding as a color. At the start, "
                                 "every node is distinct."):
            self.play(Create(VGroup(*lines)), FadeIn(VGroup(*dots)))
            self.play(Write(layer_label))

        # adjacency list for averaging (include self, like a GCN with self-loops)
        neigh = {i: [i] for i in range(n)}
        for a, b in edge_pairs:
            neigh[a].append(b)
            neigh[b].append(a)

        # --- simulate layers: each layer averages a node's color with neighbours ---
        narrations = [
            "Apply one message-passing layer: every node blends with its neighbours. "
            "Still distinguishable, and now structure-aware. This is the sweet spot.",
            "Another layer. The colors are converging — nodes are starting to look alike.",
            "Layer three. Almost uniform.",
            "By layer six, every node has averaged the entire graph. All embeddings have "
            "collapsed to the same point. The classifier has nothing left to separate.",
        ]
        layer_numbers = [1, 2, 3, 6]

        current = list(colors)
        for note, k in zip(narrations, layer_numbers):
            steps = 1 if k <= 3 else 3  # jump 3 diffusion steps for the "layer 6" beat
            for _ in range(steps):
                current = [average_color(*[current[j] for j in neigh[i]]) for i in range(n)]

            new_label = Text(
                f"Layer {k}" + (" — embeddings collapsed (over-smoothed)" if k == 6 else ""),
                font_size=24, color=(RED if k == 6 else WHITE),
            ).to_edge(np.array([0, -1, 0]))
            with self.voiceover(text=note):
                self.play(layer_label.animate.become(new_label))
                self.play(*[d.animate.set_color(c) for d, c in zip(dots, current)], run_time=1.5)

        # --- takeaway ---
        with self.voiceover(text="The fix is simple: stay shallow. Two to four layers is usually "
                                 "best, which is exactly what every model in this repo uses. "
                                 "If you truly need depth, look up residual connections and "
                                 "jumping knowledge networks."):
            takeaway = Text("Keep GNNs shallow: 2–4 layers.\nNeed depth? → residuals, JumpingKnowledge",
                            font_size=28, color=YELLOW, line_spacing=1.2).shift(np.array([0, 0.2, 0]))
            self.play(FadeOut(VGroup(*dots)), FadeOut(VGroup(*lines)), FadeOut(layer_label))
            self.play(Write(takeaway))

        self.wait(1)
