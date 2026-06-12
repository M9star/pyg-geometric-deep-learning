"""
YouTube thumbnail generator
===========================
Builds a 1280x720 thumbnail for each video in the series: a key frame from
the video as the backdrop, darkened, with bold title typography and a series
badge overlaid.

    uv run python animations/make_thumbnails.py

Output: youtube/thumbnails/NN_<name>.png   (the youtube/ folder is
git-ignored — it's the upload package, not repo content).
Requires the 1080p videos in assets/ (render via render_all.py) and ffmpeg.
"""

import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "youtube", "thumbnails")
os.makedirs(OUT, exist_ok=True)

W, H = 1280, 720
YELLOW = (255, 211, 67)
WHITE = (245, 245, 245)
GREY = (200, 200, 200)
BADGE_BG = (30, 30, 30)

# (video file, frame timestamp seconds, big title lines, subtitle, series number)
SPECS = [
    ("sample_message_passing.mp4", 43, ["MESSAGE", "PASSING"],
     "how every GNN works", 1),
    ("sample_gcn_explained.mp4", 45, ["THE GCN", "FORMULA"],
     "explained term by term", 2),
    ("sample_gat_attention.mp4", 35, ["GRAPH", "ATTENTION"],
     "GAT: which neighbors matter?", 3),
    ("sample_graph_batching.mp4", 28, ["BATCHING", "GRAPHS"],
     "PyG's disconnected-graph trick", 4),
    ("sample_gin_wl_test.mp4", 33, ["SUM BEATS", "MEAN"],
     "GIN & the Weisfeiler-Lehman test", 5),
    ("sample_over_smoothing.mp4", 9, ["OVER-", "SMOOTHING"],
     "why deep GNNs fail", 6),
]

FONT_CANDIDATES_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_CANDIDATES_REG = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def grab_frame(video: str, ts: float, dest: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(ts), "-i", video,
         "-frames:v", "1", "-y", dest],
        check=True,
    )


def build(spec) -> str:
    video, ts, title_lines, subtitle, num = spec
    name = video.replace("sample_", "").replace(".mp4", "")
    frame_tmp = os.path.join(OUT, f"_frame_{name}.png")
    grab_frame(os.path.join(ASSETS, video), ts, frame_tmp)

    # backdrop: the video frame, resized to fill 1280x720
    img = Image.open(frame_tmp).convert("RGB").resize((W, H))
    os.remove(frame_tmp)

    # darken with a left-heavy horizontal gradient so the text pops but the
    # graph visual stays recognizable on the right
    overlay = Image.new("L", (W, 1))
    for x in range(W):
        alpha = int(200 - 150 * (x / W))      # 200 (left) -> 50 (right)
        overlay.putpixel((x, 0), alpha)
    overlay = overlay.resize((W, H))
    img = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), img, overlay)

    draw = ImageDraw.Draw(img)
    f_title = load_font(FONT_CANDIDATES_BOLD, 110)
    f_sub = load_font(FONT_CANDIDATES_REG, 44)
    f_badge = load_font(FONT_CANDIDATES_BOLD, 36)

    # big title, upper-left
    y = 90
    for line in title_lines:
        draw.text((64, y), line, font=f_title, fill=YELLOW,
                  stroke_width=4, stroke_fill=(0, 0, 0))
        y += 122

    # subtitle under the title
    draw.text((68, y + 16), subtitle, font=f_sub, fill=WHITE,
              stroke_width=2, stroke_fill=(0, 0, 0))

    # series badge, bottom-left
    badge = f"GNN SERIES · {num}/6"
    bb = draw.textbbox((0, 0), badge, font=f_badge)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    bx, by = 64, H - 110
    draw.rounded_rectangle([bx - 18, by - 14, bx + bw + 18, by + bh + 22],
                           radius=14, fill=BADGE_BG, outline=YELLOW, width=3)
    draw.text((bx, by), badge, font=f_badge, fill=WHITE)

    # library tag, bottom-right
    tag = "PyTorch Geometric"
    tb = draw.textbbox((0, 0), tag, font=f_sub)
    draw.text((W - (tb[2] - tb[0]) - 56, H - 90), tag, font=f_sub, fill=GREY,
              stroke_width=2, stroke_fill=(0, 0, 0))

    out_path = os.path.join(OUT, f"{num:02d}_{name}.png")
    img.save(out_path)
    print("wrote", os.path.relpath(out_path, ROOT))
    return out_path


if __name__ == "__main__":
    for spec in SPECS:
        build(spec)
    print("\nDone. Thumbnails are in youtube/thumbnails/ (1280x720, YouTube spec).")
