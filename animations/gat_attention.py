"""
Animation: GAT — learning HOW MUCH to listen to each neighbour.
===============================================================
Shows the difference between GCN's fixed degree-based weights and GAT's
learned attention coefficients, narrated with gTTS.

Source of the method: Veličković et al., "Graph Attention Networks",
ICLR 2018 (arXiv:1710.10903).

Render:
    uv run manim -ql animations/gat_attention.py GATAttention
"""

from manim import (BLUE, GREEN, GREY, ORANGE, RED, WHITE, YELLOW, Create, Dot,
                   FadeIn, FadeOut, Indicate, Line, MathTex, Text, VGroup,
                   Write, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class GATAttention(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("Graph Attention Networks (GAT)", font_size=40).to_edge(np.array([0, 1, 0]))
        source = Text("Veličković et al., ICLR 2018", font_size=22, color=BLUE).next_to(
            title, np.array([0, -1, 0]))
        with self.voiceover(text="Graph attention networks, from Velichkovich and colleagues, "
                                 "twenty eighteen."):
            self.play(Write(title), FadeIn(source))
        self.play(FadeOut(title), FadeOut(source))

        # --- a center node with three neighbours ---
        center = Dot(point=[0, -0.3, 0], radius=0.28, color=BLUE)
        center_label = Text("v", font_size=26).next_to(center, 0.15 * np.array([0, -1, 0]))

        neighbor_pos = [np.array([-3.0, 1.4, 0]),
                        np.array([0.0, 2.2, 0]),
                        np.array([3.0, 1.4, 0])]
        neighbors = VGroup(*[Dot(p, radius=0.24, color=GREEN) for p in neighbor_pos])
        n_labels = VGroup(*[Text(f"u{i+1}", font_size=22).next_to(d, 0.15 * np.array([0, 1, 0]))
                            for i, d in enumerate(neighbors)])
        edges = [Line(center.get_center(), p, color=GREY, stroke_width=4)
                 for p in neighbor_pos]

        with self.voiceover(text="Here is a node v with three neighbours."):
            self.play(Create(VGroup(*edges)), FadeIn(center), FadeIn(neighbors))
            self.play(Write(center_label), Write(n_labels))

        # --- the GCN limitation: equal/fixed weights ---
        gcn_note = Text("GCN: weights fixed by node degree — every neighbour treated the same",
                        font_size=24, color=GREY).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="In a G C N, each neighbour's weight is fixed by the graph's degrees. "
                                 "Every neighbour is treated the same, whether it is relevant or not."):
            self.play(Write(gcn_note))
            self.play(*[Indicate(e, color=WHITE, scale_factor=1.0) for e in edges])

        # --- GAT: learned attention, visualized as edge thickness ---
        with self.voiceover(text="GAT instead learns an attention score, alpha, for every edge. "
                                 "Watch the edges: a thick edge means v listens closely; "
                                 "a thin one is mostly ignored."):
            self.play(FadeOut(gcn_note))
            # alpha values that sum to 1: u1 important, u2 medium, u3 ignored
            alphas = [0.65, 0.25, 0.10]
            widths = [12, 6, 1.5]
            colors = [ORANGE, YELLOW, GREY]
            self.play(*[e.animate.set_stroke(width=w, color=c)
                        for e, w, c in zip(edges, widths, colors)], run_time=2)

            alpha_labels = VGroup(*[
                MathTex(rf"\alpha = {a:.2f}", font_size=30, color=c).move_to(
                    0.55 * e.get_center() + 0.45 * p + np.array([0.0, 0.35, 0]))
                for e, p, a, c in zip(edges, neighbor_pos, alphas, colors)
            ])
            self.play(Write(alpha_labels))

        with self.voiceover(text="The scores are normalized with a softmax over the neighbourhood, "
                                 "so they always sum to one."):
            sum_note = MathTex(r"\sum_{u \in N(v)} \alpha_{vu} = 1", font_size=34,
                               color=WHITE).to_edge(np.array([0, -1, 0]))
            self.play(Write(sum_note))
        self.play(FadeOut(sum_note))

        # --- how alpha is computed ---
        attn_formula = MathTex(
            r"\alpha_{vu}", r"=", r"\mathrm{softmax}_u\!\Big(",
            r"\mathrm{LeakyReLU}\big(\,\mathbf{a}^{\top}[\,W h_v \,\|\, W h_u\,]\big)", r"\Big)",
            font_size=34,
        ).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="Where does alpha come from? A tiny neural network. It looks at the "
                                 "features of both endpoints, transformed by W and concatenated, "
                                 "scores the pair with a learned vector a, and passes it through "
                                 "a leaky ReLU before the softmax."):
            self.play(Write(attn_formula))
            self.play(attn_formula[3].animate.set_color(YELLOW))
        self.play(FadeOut(attn_formula))

        # --- the update rule ---
        update = MathTex(
            r"h_v' = \sigma\!\Big(\sum_{u \in N(v)} \alpha_{vu}\, W\, h_u\Big)",
            font_size=38,
        ).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="The update is then just a weighted sum: each neighbour's message, "
                                 "scaled by its attention score. Same message passing template, "
                                 "smarter weights."):
            self.play(Write(update))
            self.play(center.animate.set_color(ORANGE))

        # --- multi-head note ---
        with self.voiceover(text="In practice GAT runs several attention heads in parallel and "
                                 "concatenates them, so different heads can focus on different "
                                 "kinds of neighbours. That is the model you trained on Cora."):
            heads = Text("× K heads, concatenated (multi-head attention)",
                         font_size=24, color=GREEN).next_to(update, np.array([0, 1, 0]), buff=0.4)
            self.play(Write(heads))

        self.wait(1)
