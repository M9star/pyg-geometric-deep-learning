"""
Animation: Batching graphs — the "one big disconnected graph" trick.
====================================================================
The #1 confusion point in PyG: how do you mini-batch graphs of different
sizes? Answer: glue them into ONE disconnected graph and remember which
node belongs to which graph with a `batch` vector. Then pooling (readout)
collapses each graph's nodes into a single vector.

Used in module 03 (graph classification) and the capstone project.

Render:
    uv run manim -ql animations/graph_batching.py GraphBatching
"""

from manim import (BLUE, GREEN, GREY, ORANGE, WHITE, YELLOW, Create, Dot,
                   FadeIn, FadeOut, Line, MathTex, SurroundingRectangle, Text,
                   VGroup, Write, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


def make_graph(node_positions, edge_pairs, color):
    """Build a small graph (dots + lines) as a VGroup."""
    dots = [Dot(p, radius=0.16, color=color) for p in node_positions]
    lines = [Line(node_positions[a], node_positions[b], color=GREY, stroke_width=3)
             for a, b in edge_pairs]
    return VGroup(*lines, *dots), dots


class GraphBatching(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("Batching graphs in PyG", font_size=40).to_edge(np.array([0, 1, 0]))
        with self.voiceover(text="How do you put many graphs of different sizes into one batch? "
                                 "You can't stack them like images."):
            self.play(Write(title))
        self.play(FadeOut(title))

        # --- three small graphs of different sizes ---
        # Graph 0: triangle (3 nodes), blue, on the left
        tri_pos = [np.array([-4.5, 1.8, 0]), np.array([-5.3, 0.6, 0]), np.array([-3.7, 0.6, 0])]
        g0, g0_dots = make_graph(tri_pos, [(0, 1), (1, 2), (2, 0)], BLUE)

        # Graph 1: a 2-node pair, green, middle
        pair_pos = [np.array([-0.6, 1.6, 0]), np.array([0.6, 0.8, 0])]
        g1, g1_dots = make_graph(pair_pos, [(0, 1)], GREEN)

        # Graph 2: a 4-node square, orange, right
        sq_pos = [np.array([3.4, 1.9, 0]), np.array([4.8, 1.9, 0]),
                  np.array([4.8, 0.5, 0]), np.array([3.4, 0.5, 0])]
        g2, g2_dots = make_graph(sq_pos, [(0, 1), (1, 2), (2, 3), (3, 0)], ORANGE)

        labels = VGroup(
            Text("graph 0", font_size=22, color=BLUE).next_to(g0, np.array([0, 1, 0])),
            Text("graph 1", font_size=22, color=GREEN).next_to(g1, np.array([0, 1, 0])),
            Text("graph 2", font_size=22, color=ORANGE).next_to(g2, np.array([0, 1, 0])),
        )

        with self.voiceover(text="Say we have three graphs: a triangle with three nodes, "
                                 "a pair with two, and a square with four."):
            self.play(Create(g0), Create(g1), Create(g2), Write(labels))

        # --- glue them into one big disconnected graph ---
        with self.voiceover(text="PyG's trick: glue them into one big disconnected graph. "
                                 "Node features are simply concatenated, and the edge indices are "
                                 "shifted so edges never cross between graphs."):
            self.play(FadeOut(labels))
            self.play(g0.animate.shift(np.array([1.2, -0.8, 0])),
                      g1.animate.shift(np.array([0.0, -0.6, 0])),
                      g2.animate.shift(np.array([-1.2, -0.8, 0])), run_time=1.5)
            box = SurroundingRectangle(VGroup(g0, g1, g2), color=WHITE, buff=0.4)
            box_label = Text("one Batch = one big disconnected graph",
                             font_size=24).next_to(box, np.array([0, 1, 0]))
            self.play(Create(box), Write(box_label))

        # --- the batch vector ---
        batch_vec = MathTex(
            r"\texttt{batch} = [\,",
            r"0,0,0,", r"\ 1,1,", r"\ 2,2,2,2", r"\,]",
            font_size=36,
        ).to_edge(np.array([0, -1, 0])).shift(np.array([0, 0.6, 0]))
        batch_vec[1].set_color(BLUE)
        batch_vec[2].set_color(GREEN)
        batch_vec[3].set_color(ORANGE)
        with self.voiceover(text="A batch vector remembers which graph each node came from: "
                                 "zero zero zero for the triangle, one one for the pair, "
                                 "and two two two two for the square. "
                                 "The data loader builds all of this automatically."):
            self.play(Write(batch_vec))

        # --- pooling / readout ---
        with self.voiceover(text="After message passing we have node embeddings, but we need one "
                                 "vector per graph. Global pooling collapses each graph's nodes, "
                                 "guided by the batch vector."):
            self.play(FadeOut(box), FadeOut(box_label))
            # collapse each graph's dots to its centroid
            c0 = sum(d.get_center() for d in g0_dots) / len(g0_dots)
            c1 = sum(d.get_center() for d in g1_dots) / len(g1_dots)
            c2 = sum(d.get_center() for d in g2_dots) / len(g2_dots)
            self.play(*[d.animate.move_to(c0) for d in g0_dots],
                      *[d.animate.move_to(c1) for d in g1_dots],
                      *[d.animate.move_to(c2) for d in g2_dots],
                      *[FadeOut(m) for g in (g0, g1, g2) for m in g if isinstance(m, Line)],
                      run_time=1.5)
            pooled = VGroup(
                Dot(c0, radius=0.3, color=BLUE),
                Dot(c1, radius=0.3, color=GREEN),
                Dot(c2, radius=0.3, color=ORANGE),
            )
            pool_labels = VGroup(
                MathTex(r"z_0", font_size=30, color=BLUE).next_to(pooled[0], np.array([0, 1, 0])),
                MathTex(r"z_1", font_size=30, color=GREEN).next_to(pooled[1], np.array([0, 1, 0])),
                MathTex(r"z_2", font_size=30, color=ORANGE).next_to(pooled[2], np.array([0, 1, 0])),
            )
            self.play(FadeIn(pooled), Write(pool_labels))

        pool_formula = MathTex(
            r"z_g = \texttt{global\_add\_pool}(h,\ \texttt{batch})",
            font_size=34, color=YELLOW,
        ).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="That's the readout: global add pool sums each graph's node "
                                 "embeddings into a single vector, ready for the classifier. "
                                 "Three graphs in, three vectors out."):
            self.play(FadeOut(batch_vec))
            self.play(Write(pool_formula))

        self.wait(1)
