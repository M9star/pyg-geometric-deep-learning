"""
Render every animation scene in one go.

    uv run python animations/render_all.py            # low quality (fast)
    uv run python animations/render_all.py --quality h # high quality

Needs internet (gTTS narration) the first time each scene is rendered;
manim-voiceover caches the audio afterwards.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (filename, SceneClassName)
SCENES = [
    ("message_passing.py", "MessagePassing"),
    ("gcn_explained.py", "GCNExplained"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=["l", "m", "h", "k"], default="l",
                        help="manim quality: l(ow)/m/h(igh)/k(4k)")
    args = parser.parse_args()

    for filename, scene in SCENES:
        path = HERE / filename
        print(f"\n=== rendering {scene} ({filename}) ===")
        cmd = ["manim", f"-q{args.quality}", str(path), scene]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[!] {scene} failed (exit {result.returncode}).", file=sys.stderr)
            sys.exit(result.returncode)

    print("\nAll scenes rendered. See media/videos/.")


if __name__ == "__main__":
    main()
