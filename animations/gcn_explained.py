"""
Animation: What the GCN formula actually does.
==============================================
Builds up the Graph Convolutional Network update rule term by term and
explains the symmetric normalization, narrated with gTTS.

Source of the method: Kipf & Welling, "Semi-Supervised Classification with
Graph Convolutional Networks", ICLR 2017 (arXiv:1609.02907).

Render:
    uv run manim -ql animations/gcn_explained.py GCNExplained
"""

from manim import (BLUE, GREEN, ORANGE, WHITE, YELLOW, FadeIn, FadeOut,
                   MathTex, Text, VGroup, Write, np)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class GCNExplained(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService(lang="en"))

        title = Text("The GCN layer", font_size=44).to_edge(np.array([0, 1, 0]))
        source = Text("Kipf & Welling, ICLR 2017", font_size=22, color=BLUE).next_to(
            title, np.array([0, -1, 0]))
        with self.voiceover(text="Let's read the graph convolution formula, from Kipf and Welling, "
                                 "twenty seventeen."):
            self.play(Write(title), FadeIn(source))

        # The full formula, then we highlight parts.
        formula = MathTex(
            r"h_v'", r"=", r"W", r"\sum_{u \in N(v) \cup \{v\}}",
            r"\frac{1}{\sqrt{d_u\, d_v}}", r"h_u",
            font_size=46,
        )
        with self.voiceover(text="The new features of node v are computed like this."):
            self.play(Write(formula))
        self.wait(0.3)

        def explain(index, color, text):
            with self.voiceover(text=text):
                self.play(formula[index].animate.set_color(color))
                self.wait(0.2)

        explain(5, GREEN,
                "h u: the features of each neighbour u, and of v itself thanks to the self-loop.")
        explain(3, YELLOW,
                "We sum over all neighbours of v, plus v, so a node never forgets itself.")
        explain(4, ORANGE,
                "Each term is scaled by one over the square root of the degrees of u and v. "
                "This symmetric normalization stops high-degree nodes from dominating.")
        explain(2, BLUE,
                "Finally W is a learned weight matrix, shared across every node in the graph.")

        with self.voiceover(text="That weight sharing is the graph version of a convolution: "
                                 "the same filter, applied everywhere."):
            self.play(formula.animate.set_color(WHITE))

        takeaway = Text("= a degree-normalized average of neighbours, then a linear map",
                        font_size=26, color=GREEN).to_edge(np.array([0, -1, 0]))
        with self.voiceover(text="In plain words: average your neighbours, weighted by degree, "
                                 "then transform."):
            self.play(Write(takeaway))
        self.wait(1)
