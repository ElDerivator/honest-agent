#!/usr/bin/env python3
"""Render assets/social-card.png — the repo banner and GitHub social-preview
image. GitHub refuses to proxy a relative-path SVG in a README <img>, and it uses
a raster og:image for link thumbnails; a PNG serves both. Rerun after edits:

    python3 assets/make_social_card.py

Reproduces assets/social-card.svg at 2x (2560x1280) with DejaVu Sans Mono. Colors
are GitHub-dark tokens so the card matches the terminal aesthetic of the README.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

SCALE = 2
W, H = 1280 * SCALE, 640 * SCALE

BG = "#0d1117"
BORDER = "#21262d"
TERM_BG = "#010409"
FG = "#e6edf3"
MUTED = "#8b949e"
FAINT = "#6e7681"
BLUE = "#a5d6ff"
AMBER = "#e3b341"
GREEN = "#3fb950"
RED = "#f85149"
LIGHTS = ("#ff5f56", "#ffbd2e", "#27c93f")

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size * SCALE)


def s(v: int) -> int:
    return v * SCALE


def segments(draw: ImageDraw.ImageDraw, x: int, y: int, parts, fnt) -> None:
    """Draw colored text runs left-to-right from a shared baseline (anchor 'ls')."""
    for text, color, bold in parts:
        f = FONT_CODE_BOLD if bold else fnt
        draw.text((x, y), text, font=f, fill=color, anchor="ls")
        x += round(f.getlength(text))


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

FONT_TITLE = font(MONO_BOLD, 76)
FONT_SUB = font(MONO, 30)
FONT_CODE = font(MONO, 27)
FONT_CODE_BOLD = font(MONO_BOLD, 27)
FONT_FOOT = font(MONO, 26)
FONT_APACHE = font(MONO, 24)

# frame + terminal panel
d.rounded_rectangle([s(20), s(20), s(1260), s(620)], radius=s(16), outline=BORDER, width=s(2))
d.text((s(80), s(170)), "honest-agent", font=FONT_TITLE, fill=FG, anchor="ls")
d.text((s(82), s(232)), "Evidence-gated task completion for autonomous agents",
       font=FONT_SUB, fill=MUTED, anchor="ls")
d.rounded_rectangle([s(80), s(290), s(1200), s(500)], radius=s(12), outline=BORDER,
                    width=max(1, round(1.5 * SCALE)), fill=TERM_BG)
for i, col in enumerate(LIGHTS):
    cx = s(112 + i * 24)
    d.ellipse([cx - s(7), s(320) - s(7), cx + s(7), s(320) + s(7)], fill=col)

# the rule, in three lines — status column fixed so the arrows align in mono
ARROW_X = s(720)
rows = [
    (290 + 85, [("close_episode(", MUTED, False), ('"COMPLETED"', BLUE, False), (")", MUTED, False)],
     "UNVERIFIED", AMBER),
    (290 + 130, [("close_episode(", MUTED, False), ('"COMPLETED"', BLUE, False), (", proof)", MUTED, False)],
     "COMPLETED", GREEN),
    (290 + 175, [("raise RuntimeError", MUTED, False)], "FAILED", RED),
]
for y, left, status, color in rows:
    segments(d, s(112), s(y), left, FONT_CODE)
    d.text((ARROW_X, s(y)), "→ ", font=FONT_CODE, fill=MUTED, anchor="ls")
    d.text((ARROW_X + round(FONT_CODE.getlength("→ ")), s(y)), status,
           font=FONT_CODE_BOLD, fill=color, anchor="ls")

d.text((s(80), s(560)), "The oracle of truth is the environment, not the model.",
       font=FONT_FOOT, fill=FAINT, anchor="ls")
d.text((s(1200), s(560)), "Apache-2.0", font=FONT_APACHE, fill=FAINT, anchor="rs")

out = os.path.join(os.path.dirname(__file__), "social-card.png")
img.save(out, "PNG", optimize=True)
print(f"wrote {out} ({W}x{H}, {os.path.getsize(out) // 1024} KB)")
