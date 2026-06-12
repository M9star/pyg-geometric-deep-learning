"""
Animation: Why GIN uses SUM — the Weisfeiler–Lehman connection.
===============================================================
Shows two different neighbourhoods that MEAN aggregation cannot tell apart
but SUM can — the core insight of the GIN paper, which proves message-passing
GNNs are at most as powerful as the Weisfeiler–Lehman isomorphism test, and
that sum + MLP reaches that bound.

Source: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019
(arXiv:1810.00826).

Render:
    uv run manim -ql animations/gin_wl_test.py GINWLTest
"""

from manim import (BLUE, GREEN, GREY, ORANGE, RED, WHITE, YELLOW, Create, Dot,
                   FadeIn, FadeOut, Line, MathTex, Text, VGroup, Write, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


def star(center, n_neighbors, radius, color):
    """A star graph: center dot + n neighbours on a circle, all feature '1'."""
    cdot = Dot(center, radius=0.24, color=color)
    angles = [np.pi / 2 + 2 * np.pi * i / n_neighbors for i in range(n_neighbors)]
    npos = [center + radius * np.array([np.cos(a), np.sin(a), 0]) for a in angles]
    ndots = [Dot(p, radius=0.18, color=GREEN) for p in npos]
    edges = [Line(center, p, color=GREY, stroke_width=3) for p in npos]
    feats = [MathTex("1", font_size=26).move_to(p + 0.38 * (p - center) / np.linalg.norm(p - center))
             for p in npos]
    return VGroup(*edges, cdot, *ndots, *feats)


class GINWLTest(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("Why GIN uses SUM", font_size=40).to_edge(np.array([0, 1, 0]))
        source = Text("Xu et al., ICLR 2019 — “How Powerful are GNNs?”",
                      font_size=22, color=BLUE).next_to(title, np.array([0, -1, 0]))
        with self.voiceover(text="How powerful is a graph neural network? It depends on one tiny "
                                 "choice: how you aggregate the neighbours."):
            self.play(Write(title), FadeIn(source))
        self.play(FadeOut(title), FadeOut(source))

        # --- two stars: 2 neighbours vs 3 neighbours, all features = 1 ---
        left_c = np.array([-3.2, 0.9, 0])
        right_c = np.array([3.2, 0.9, 0])
        s_left = star(left_c, 2, 1.4, BLUE)
        s_right = star(right_c, 3, 1.4, ORANGE)
        lab_left = MathTex("v", font_size=30).next_to(Dot(left_c), 0.2 * np.array([0, -1, 0]))
        lab_right = MathTex("w", font_size=30).next_to(Dot(right_c), 0.2 * np.array([0, -1, 0]))

        with self.voiceover(text="Look at these two nodes. v has two neighbours, w has three. "
                                 "Every neighbour has the same feature: one. "
                                 "These neighbourhoods are clearly different."):
            self.play(Create(s_left), Create(s_right), Write(lab_left), Write(lab_right))

        # --- MEAN fails ---
        mean_l = MathTex(r"\text{mean} = \tfrac{1+1}{2} = 1", font_size=32,
                         color=RED).next_to(s_left, np.array([0, -1, 0]), buff=0.5)
        mean_r = MathTex(r"\text{mean} = \tfrac{1+1+1}{3} = 1", font_size=32,
                         color=RED).next_to(s_right, np.array([0, -1, 0]), buff=0.5)
        verdict_mean = Text("identical → MEAN cannot tell them apart!",
                            font_size=26, color=RED).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="Aggregate with the mean, and both give exactly one. "
                                 "Identical embeddings. The network literally cannot tell "
                                 "these two structures apart."):
            self.play(Write(mean_l), Write(mean_r))
            self.play(Write(verdict_mean))
        self.play(FadeOut(mean_l), FadeOut(mean_r), FadeOut(verdict_mean))

        # --- SUM works ---
        sum_l = MathTex(r"\text{sum} = 1+1 = 2", font_size=32,
                        color=GREEN).next_to(s_left, np.array([0, -1, 0]), buff=0.5)
        sum_r = MathTex(r"\text{sum} = 1+1+1 = 3", font_size=32,
                        color=GREEN).next_to(s_right, np.array([0, -1, 0]), buff=0.5)
        verdict_sum = Text("2 ≠ 3 → SUM distinguishes them ✓",
                           font_size=26, color=GREEN).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="Aggregate with the sum, and you get two versus three. "
                                 "Different embeddings. The sum preserves the size and "
                                 "multiset structure of the neighbourhood; mean and max throw it away."):
            self.play(Write(sum_l), Write(sum_r))
            self.play(Write(verdict_sum))
        self.play(FadeOut(sum_l), FadeOut(sum_r), FadeOut(verdict_sum),
                  FadeOut(s_left), FadeOut(s_right), FadeOut(lab_left), FadeOut(lab_right))

        # --- the theory + the GIN layer ---
        theory = Text("A message-passing GNN can distinguish graphs at most as well as\n"
                      "the Weisfeiler–Lehman (WL) isomorphism test.",
                      font_size=26, line_spacing=1.1).shift(np.array([0, 1.6, 0]))
        with self.voiceover(text="The GIN paper proved the general result: no message-passing network "
                                 "can distinguish graphs better than the classic Weisfeiler-Lehman "
                                 "isomorphism test. That is the ceiling."):
            self.play(Write(theory))

        gin = MathTex(
            r"h_v' = \text{MLP}\Big((1+\epsilon)\, h_v + \sum_{u \in N(v)} h_u\Big)",
            font_size=40,
        ).shift(np.array([0, -0.6, 0]))
        with self.voiceover(text="And G I N reaches that ceiling: sum the neighbours, keep a "
                                 "weighted copy of yourself, and push it through an M L P. "
                                 "The injective sum plus M L P is what makes it maximally expressive."):
            self.play(Write(gin))
            self.play(gin.animate.set_color(YELLOW))

        takeaway = Text("This is the model that classifies molecules in module 03.",
                        font_size=24, color=GREEN).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="This is exactly the model you trained on the mutagenic molecules "
                                 "in module three."):
            self.play(Write(takeaway))

        self.wait(1)
