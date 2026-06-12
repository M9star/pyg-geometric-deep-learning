"""
Animation: Message Passing — the core operation of every GNN.
=============================================================
Narrated with gTTS (free, needs internet at render time).

Render (low quality, fast — this is the pre-rendered sample):
    uv run manim -ql animations/message_passing.py MessagePassing

Render high quality:
    uv run manim -qh animations/message_passing.py MessagePassing

The voiceover narration is generated from the text in each
`with self.voiceover(...)` block, and the visuals are timed to it.
"""

from manim import (BLUE, GREEN, GREY, ORANGE, WHITE, YELLOW, Circle, Create,
                   Dot, FadeIn, FadeOut, Indicate, Line, MathTex, Text, VGroup,
                   Write, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class MessagePassing(VoiceoverScene):
    def construct(self):
        # Use Google Text-to-Speech for narration.
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("Message Passing", font_size=44).to_edge(0.5 * np.array([0, 1, 0]))
        with self.voiceover(text="Every graph neural network is built on one idea: message passing."):
            self.play(Write(title))
        self.play(FadeOut(title))

        # --- build a small star graph: a center node and three neighbours ---
        center = Dot(point=[0, 0, 0], radius=0.28, color=BLUE)
        center_label = Text("v", font_size=26).next_to(center, 0.1 * np.array([0, -1, 0]))

        neighbor_pos = [np.array([-2.6, 1.3, 0]),
                        np.array([2.6, 1.3, 0]),
                        np.array([0, -2.2, 0])]
        neighbors = VGroup(*[Dot(p, radius=0.24, color=GREEN) for p in neighbor_pos])
        n_labels = VGroup(*[Text(f"u{i+1}", font_size=22).next_to(d, 0.1 * np.array([0, 1, 0]))
                            for i, d in enumerate(neighbors)])
        edges = VGroup(*[Line(center.get_center(), p, color=GREY, stroke_width=3)
                         for p in neighbor_pos])

        with self.voiceover(text="Take a node, v, and its neighbours in the graph."):
            self.play(Create(edges), FadeIn(center), FadeIn(neighbors))
            self.play(Write(center_label), Write(n_labels))

        # --- STEP 1: MESSAGE ---
        step = Text("1. MESSAGE", font_size=30, color=YELLOW).to_corner(np.array([-1, 1, 0]))
        with self.voiceover(text="Step one: message. Each neighbour prepares a message from its own features."):
            self.play(Write(step))
            self.play(Indicate(neighbors, color=YELLOW, scale_factor=1.3))

        # --- STEP 2: AGGREGATE ---
        with self.voiceover(text="Step two: aggregate. Those messages travel along the edges and are "
                                 "combined with a sum that does not depend on their order."):
            self.play(step.animate.become(
                Text("2. AGGREGATE", font_size=30, color=YELLOW).to_corner(np.array([-1, 1, 0]))))
            # animate dots flowing from each neighbour into the center
            pulses = [Dot(p, radius=0.12, color=ORANGE) for p in neighbor_pos]
            self.play(FadeIn(VGroup(*pulses)))
            self.play(*[d.animate.move_to(center.get_center()) for d in pulses], run_time=1.5)
            self.play(FadeOut(VGroup(*pulses)))
            agg = MathTex(r"\sum_{u \in N(v)} m_u", font_size=34).next_to(center, np.array([1, 0, 0]), buff=1.0)
            self.play(Write(agg))

        # --- STEP 3: UPDATE ---
        with self.voiceover(text="Step three: update. The node combines that aggregated message with its own "
                                 "features to get a new, richer representation."):
            self.play(step.animate.become(
                Text("3. UPDATE", font_size=30, color=YELLOW).to_corner(np.array([-1, 1, 0]))))
            new_ring = Circle(radius=0.42, color=ORANGE).move_to(center.get_center())
            self.play(center.animate.set_color(ORANGE), Create(new_ring))
            self.play(FadeOut(agg))

        # --- the formula ---
        formula = MathTex(
            r"h_v' = \text{UPDATE}\Big(h_v,\ \sum_{u \in N(v)} \text{MSG}(h_u)\Big)",
            font_size=36,
        ).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="That is one layer. Stack a few, and information flows across the whole graph. "
                                 "G C N, G A T, GraphSAGE and G I N all just change how the message and the "
                                 "update are defined."):
            self.play(Write(formula))
            self.wait(0.5)

        self.wait(1)
