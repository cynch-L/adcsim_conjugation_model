#!/usr/bin/env python3
"""Paste the overview and close-up renders side by side, add title + legend.

Called by render_readme_figures.py (same interpreter, has Pillow).
"""
import argparse
import json

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial.ttf",
]
MARGIN = 70
GAP = 60
TITLE_SIZE = 72
LEGEND_SIZE = 44
TITLE = "Trastuzumab + vc-MMAE  |  {tag}  |  collapsed pose"


def load_font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_legend(draw, x, y, entries, font):
    swatch = int(font.size * 0.72)
    line = int(font.size * 1.55)
    for (rgb, text) in entries:
        colour = tuple(int(round(c * 255)) for c in rgb)
        draw.rectangle([x, y, x + swatch, y + swatch], fill=colour)
        draw.text((x + swatch + 18, y + swatch // 2), text,
                  fill=(30, 30, 30), font=font, anchor="lm")
        y += line
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overview", required=True)
    ap.add_argument("--closeup", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--legend", required=True)
    args = ap.parse_args()

    ov = Image.open(args.overview).convert("RGB")
    cu = Image.open(args.closeup).convert("RGB")
    if ov.size != cu.size:
        cu = cu.resize(ov.size)

    title_font = load_font(TITLE_SIZE)
    legend_font = load_font(LEGEND_SIZE)

    W = MARGIN * 2 + GAP + ov.width * 2
    # room for: title + two panels + captions + legend row + site list
    H = MARGIN * 2 + TITLE_SIZE + GAP + ov.height + LEGEND_SIZE * 5 + 60
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    y0 = MARGIN
    draw.text((MARGIN, y0), TITLE.format(tag=args.tag),
              fill=(20, 20, 20), font=title_font)
    y1 = y0 + TITLE_SIZE + GAP

    x_ov = MARGIN
    x_cu = x_ov + ov.width + GAP
    canvas.paste(ov, (x_ov, y1))
    canvas.paste(cu, (x_cu, y1))
    draw.line([(x_cu - GAP // 2, y1), (x_cu - GAP // 2, y1 + ov.height)],
              fill=(200, 200, 200), width=3)
    draw.text((x_cu, y1 + ov.height + 8),
              "Conjugation site close-up (ball-and-stick payload)",
              fill=(90, 90, 90), font=legend_font)
    draw.text((x_ov, y1 + ov.height + 8),
              "Whole antibody, conjugation site facing the viewer",
              fill=(90, 90, 90), font=legend_font)

    sites = json.loads(args.sites)
    legend = json.loads(args.legend)
    sites_text = ("Conjugated sites: "
                  + ", ".join("%s-Cys%d" % (c, i) for c, i in sites)
                  + "   (each SG-C91 bond = 1.82 A)")

    y2 = y1 + ov.height + LEGEND_SIZE * 2
    legend_y = y2 + LEGEND_SIZE * 2 + 6
    sites_y = legend_y + LEGEND_SIZE * 2
    x = MARGIN
    entry_w = max(draw.textlength(txt, font=legend_font) for _, txt in legend) + LEGEND_SIZE + 40
    for rgb, txt in legend:
        swatch = int(legend_font.size * 0.7)
        colour = tuple(int(round(c * 255)) for c in rgb)
        draw.rectangle([x, legend_y, x + swatch, legend_y + swatch], fill=colour)
        draw.text((x + swatch + 14, legend_y + swatch // 2), txt,
                  fill=(30, 30, 30), font=legend_font, anchor="lm")
        x += entry_w
    draw.text((MARGIN, sites_y), sites_text,
              fill=(30, 30, 30), font=legend_font)

    canvas.save(args.output, dpi=(220, 220))
    print("canvas", canvas.size)


if __name__ == "__main__":
    main()
