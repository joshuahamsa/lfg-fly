"""The before → after card (spec §4.5).

A link-free 1200×675 card built from the hero's real on-chain images (the
metadata `image` before and after the move): the day number, the two renders
side by side with an arrow between, and the changes spelled out underneath
("Head: Crown → Pirate Hat"). Only PIL; fonts are DejaVu when the system has it,
else Pillow's bundled default. Nothing here draws a URL.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lfg_fly.brain.senses import NONE, SLOTS

CARD_SIZE = (1200, 675)
PANEL = 480  # each render is fitted into a PANEL×PANEL square
MARGIN_X = 60
HEADER = 85  # the panels start here
GAP = CARD_SIZE[0] - 2 * MARGIN_X - 2 * PANEL  # 120 px between the panels, for the arrow
STAY_LINE = "Kept the look; nothing beat it"
NOTHING = "nothing"  # how a None value reads

BG = (16, 16, 20)
PANEL_BG = (34, 34, 40)
INK = (240, 240, 240)
MUTED = (160, 160, 170)
ACCENT = (255, 200, 60)

FONT_CANDIDATES = {
    False: ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",),
    True: ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",),
}


def panel_centres() -> tuple[tuple[int, int], tuple[int, int]]:
    """Centre pixel of the before (left) and after (right) panels."""
    y = HEADER + PANEL // 2
    return (MARGIN_X + PANEL // 2, y), (MARGIN_X + PANEL + GAP + PANEL // 2, y)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[bold]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _display(value: object) -> str:
    return NOTHING if value == NONE or value is None else str(value)


def change_lines(changes: Sequence[dict]) -> list[str]:
    """One line per change: "Slot: from → value" when the change carries `from`, else
    "Slot → value"; a None value reads as "nothing". No changes → [STAY_LINE]."""
    lines = []
    for c in changes:
        slot, value = c["slot"], _display(c["value"])
        if "from" in c and c["from"] is not None:
            lines.append(f"{slot}: {_display(c['from'])} → {value}")
        else:
            lines.append(f"{slot} → {value}")
    return lines or [STAY_LINE]


def with_from(changes: Sequence[dict], before: Sequence[str] | None) -> list[dict]:
    """Copies of `changes` with `from` filled from the `before` look (SLOTS order).
    Entries that already carry `from`, or name an unknown slot, are copied as they are."""
    if before is None:
        return [dict(c) for c in changes]
    out = []
    for c in changes:
        c = dict(c)
        if "from" not in c and c.get("slot") in SLOTS:
            c["from"] = before[SLOTS.index(c["slot"])]
        out.append(c)
    return out


def _fit(im: Image.Image, side: int) -> Image.Image:
    """`im` contained in a side×side transparent square, aspect preserved, centred."""
    rgba = im.convert("RGBA")
    w, h = rgba.size
    scale = min(side / w, side / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if size != rgba.size:
        rgba = rgba.resize(size, Image.LANCZOS)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.alpha_composite(rgba, ((side - size[0]) // 2, (side - size[1]) // 2))
    return out


def _wrap(draw: ImageDraw.ImageDraw, items: list[str], width: int,
          sizes: Sequence[int] = (28, 24, 20), max_lines: int = 2,
          sep: str = " · ") -> tuple[list[str], ImageFont.ImageFont | ImageFont.FreeTypeFont]:
    """Greedy wrap of `items` (joined by `sep`) into at most `max_lines` lines, trying
    the font sizes in order; the smallest size truncates with an ellipsis if needed."""
    font = _font(sizes[-1])
    lines: list[str] = []
    for size in sizes:
        font = _font(size)
        lines, cur = [], ""
        for item in items:
            trial = item if not cur else cur + sep + item
            if cur and draw.textlength(trial, font=font) > width:
                lines.append(cur)
                cur = item
            else:
                cur = trial
        if cur:
            lines.append(cur)
        if len(lines) <= max_lines and all(draw.textlength(ln, font=font) <= width
                                           for ln in lines):
            return lines, font
    lines = lines[:max_lines]
    last = lines[-1]
    while last and draw.textlength(last + "…", font=font) > width:
        last = last[:-1]
    lines[-1] = last + "…"
    return lines, font


def before_after(before: Image.Image, after: Image.Image, day: int,
                 changes: Sequence[dict]) -> Image.Image:
    """The §4.5 card, 1200×675 RGB: "Day N", the before and after renders with an
    arrow between, "before"/"after" under the panels and the change lines below.
    `changes` are the record's [{"slot", "value"}], optionally with "from" (see
    `with_from`)."""
    w, h = CARD_SIZE
    out = Image.new("RGB", CARD_SIZE, BG)
    draw = ImageDraw.Draw(out)

    # panels
    (lx, ly), (rx, ry) = panel_centres()
    for (cx, cy), im in (((lx, ly), before), ((rx, ry), after)):
        x0, y0 = cx - PANEL // 2, cy - PANEL // 2
        draw.rounded_rectangle([x0, y0, x0 + PANEL - 1, y0 + PANEL - 1], radius=16,
                               fill=PANEL_BG)
        tile = _fit(im, PANEL)
        out.paste(tile, (x0, y0), tile)

    # arrow in the gap
    gx0, gx1 = MARGIN_X + PANEL + 20, MARGIN_X + PANEL + GAP - 20
    ay = ly
    head = 22
    draw.line([(gx0, ay), (gx1 - head, ay)], fill=ACCENT, width=8)
    draw.polygon([(gx1, ay), (gx1 - head - 6, ay - head), (gx1 - head - 6, ay + head)],
                 fill=ACCENT)

    # header
    draw.text((MARGIN_X, 22), f"Day {int(day):,}", fill=INK, font=_font(40, bold=True))

    # panel labels
    label_font = _font(22)
    label_y = HEADER + PANEL + 8
    for cx, label in ((lx, "before"), (rx, "after")):
        tw = draw.textlength(label, font=label_font)
        draw.text((cx - tw / 2, label_y), label, fill=MUTED, font=label_font)

    # the changes
    lines, font = _wrap(draw, change_lines(changes), w - 2 * MARGIN_X)
    step = getattr(font, "size", 26) + 6
    y = label_y + 34
    for line in lines:
        draw.text((MARGIN_X, y), line, fill=INK, font=font)
        y += step
    return out
