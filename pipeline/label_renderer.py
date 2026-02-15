"""
Stage 4: Label Renderer
Renders text labels on the generated image using Pillow.
Supports 7 annotation styles. Also generates SVG with editable labels.

Key improvements (v3):
- Dynamic margins: computed from actual text widths to prevent overflow
- Bidirectional overlap resolution: pushes labels both up and down
- Edge clamping: guarantees no label box extends beyond image bounds
- Elbow-style leader lines: horizontal + diagonal segments avoid crossings
- Anti-crossing: labels sorted by y-coordinate within each side
- Balanced side assignment: prevents all labels from piling on one side
- Font size scales with image dimensions for consistent readability
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

import math

from PIL import Image, ImageDraw, ImageFont

from config import FONT_PATH, FONT_BOLD_PATH, GENERATED_DIR

# Type alias for annotation styles
LabelStyle = Literal[
    "plain_text", "boxed_text", "numbered",
    "color_coded", "minimal", "textbook", "professional",
]

# Color palette for color_coded style
COLORS = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#E67E22",  # deep orange
    "#2980B9",  # dark blue
    "#27AE60",  # dark green
    "#8E44AD",  # dark purple
    "#D35400",  # rust
    "#16A085",  # dark teal
    "#C0392B",  # dark red
    "#2C3E50",  # navy
    "#7F8C8D",  # gray
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font, falling back to default if not found."""
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _text_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """Get text width and height reliably across Pillow versions."""
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _auto_font_size(image_width: int, image_height: int) -> int:
    """Calculate appropriate label font size based on image dimensions."""
    min_dim = min(image_width, image_height)
    # Scale: ~12px for 512px images, ~14px for 1024px, ~18px for 2048px
    return max(11, min(20, int(min_dim * 0.014)))


# ─── Layout computation ────────────────────────────────────────────

def compute_label_layout(
    points: list[dict],
    image_width: int,
    image_height: int,
    label_side: str = "right",
    font_size: int = 0,
) -> list[dict]:
    """
    Given anatomical point positions from the vision model,
    compute optimal label text positions with smart side assignment.

    Strategy:
    1. Assign each label to left or right margin based on point position
    2. Sort labels by y-coordinate within each side (anti-crossing)
    3. Place labels at ideal y-positions (matching point_y)
    4. Resolve overlaps with bidirectional minimum-displacement spreading
    5. Edge-clamp to prevent label boxes from overflowing
    6. Return positions ready for rendering

    Returns list of dicts with: label, point_x, point_y, label_x, label_y, anchor
    """
    if not points:
        return []

    # Auto-scale font size if not specified
    if font_size <= 0:
        font_size = _auto_font_size(image_width, image_height)

    font = _load_font(font_size, bold=True)
    label_height = font_size + 16  # minimum vertical spacing between labels
    pad_x = 8   # horizontal padding inside label box
    pad_y = 5   # vertical padding inside label box

    # Measure text widths
    text_widths: dict[str, int] = {}
    max_tw = 0
    for p in points:
        tw, _ = _text_size(font, p["label"])
        text_widths[p["label"]] = tw
        max_tw = max(max_tw, tw)

    # Dynamic margins: ensure space for leader line stubs and comfortable spacing
    base_margin = max(28, int(min(image_width, image_height) * 0.028))
    label_column_width = max_tw + pad_x * 2 + 8  # box + small gap

    # Label anchor x-positions: comfortably inside the margin
    right_label_x = max(image_width - base_margin, image_width // 2 + 40)
    left_label_x = min(base_margin, image_width // 2 - 40)

    # Final edge-clamp safety
    right_label_x = min(right_label_x, image_width - 6)
    left_label_x = max(left_label_x, 6)

    # ── Side assignment ──────────────────────────────
    if label_side == "left":
        assigned_sides = {p["label"]: "left" for p in points}
    elif label_side == "right":
        assigned_sides = {p["label"]: "right" for p in points}
    else:
        # "integrated" or auto: assign based on point position relative to image
        mid_x = image_width / 2
        assigned_sides = {}
        for p in points:
            # Use a dead-zone around the center: points within 10% of center
            # get assigned to whichever side has fewer labels so far
            if p["point_x"] < mid_x * 0.85:
                assigned_sides[p["label"]] = "left"
            elif p["point_x"] > mid_x * 1.15:
                assigned_sides[p["label"]] = "right"
            else:
                # Near center — assign to side with fewer labels
                left_count = sum(1 for s in assigned_sides.values() if s == "left")
                right_count = sum(1 for s in assigned_sides.values() if s == "right")
                assigned_sides[p["label"]] = "left" if left_count <= right_count else "right"

        # Balance check: if one side is heavily overloaded, rebalance
        total = len(assigned_sides)
        if total > 4:
            left_count = sum(1 for s in assigned_sides.values() if s == "left")
            right_count = total - left_count

            if left_count > total * 0.70:
                # Move some left labels to right (choose those closest to center)
                left_pts = sorted(
                    [p for p in points if assigned_sides[p["label"]] == "left"],
                    key=lambda p: -p["point_x"],  # rightmost first
                )
                to_move = left_count - (total + 1) // 2
                for p in left_pts[:to_move]:
                    assigned_sides[p["label"]] = "right"

            elif right_count > total * 0.70:
                right_pts = sorted(
                    [p for p in points if assigned_sides[p["label"]] == "right"],
                    key=lambda p: p["point_x"],  # leftmost first
                )
                to_move = right_count - (total + 1) // 2
                for p in right_pts[:to_move]:
                    assigned_sides[p["label"]] = "left"

    # ── Split and sort by y-coordinate ─────────────
    left_points = sorted(
        [p for p in points if assigned_sides[p["label"]] == "left"],
        key=lambda p: p["point_y"],
    )
    right_points = sorted(
        [p for p in points if assigned_sides[p["label"]] == "right"],
        key=lambda p: p["point_y"],
    )

    # ── Compute label positions for each side ──────
    def _distribute_labels(
        sorted_pts: list[dict],
        side: str,
    ) -> list[dict]:
        """Compute label y-positions for one side with bidirectional overlap resolution."""
        if not sorted_pts:
            return []

        n = len(sorted_pts)
        min_y = base_margin + font_size
        max_y = image_height - base_margin - font_size
        available = max_y - min_y

        # Start with ideal y = point_y for each label (clamped)
        ideal_ys = [
            max(min_y, min(p["point_y"], max_y))
            for p in sorted_pts
        ]

        # Bidirectional overlap resolution
        resolved = _resolve_overlaps_bidirectional(ideal_ys, label_height, min_y, max_y)

        # Build result dicts
        results: list[dict] = []
        for i, pt in enumerate(sorted_pts):
            lx = right_label_x if side == "right" else left_label_x

            pos = {
                "label": pt["label"],
                "point_x": pt["point_x"],
                "point_y": pt["point_y"],
                "label_x": lx,
                "label_y": int(resolved[i]),
                "anchor": side,
            }
            if "cover_bbox" in pt:
                pos["cover_bbox"] = pt["cover_bbox"]
            results.append(pos)

        return results

    positions = (
        _distribute_labels(left_points, "left")
        + _distribute_labels(right_points, "right")
    )
    return positions


def _resolve_overlaps_bidirectional(
    ideal_ys: list[float],
    label_height: float,
    min_y: float,
    max_y: float,
) -> list[float]:
    """
    Resolve overlapping labels using bidirectional pushing.
    Unlike simple top-down pushing, this spreads labels both up and down
    from the centre of gravity, producing more balanced results.
    """
    n = len(ideal_ys)
    if n <= 1:
        return list(ideal_ys)

    resolved = list(ideal_ys)

    # Pass 1: Push down from top
    for i in range(1, n):
        if resolved[i] < resolved[i - 1] + label_height:
            resolved[i] = resolved[i - 1] + label_height

    # Check if we overflowed the bottom
    if resolved[-1] > max_y:
        available = max_y - min_y
        total_needed = (n - 1) * label_height

        if total_needed > available:
            # Not enough space — compress spacing
            spacing = available / max(n - 1, 1)
            # Centre the block vertically
            centre_y = sum(ideal_ys) / n
            total_height = (n - 1) * spacing
            start_y = centre_y - total_height / 2
            start_y = max(min_y, min(start_y, max_y - total_height))
            resolved = [start_y + i * spacing for i in range(n)]
        else:
            # Pass 2: Push up from bottom to re-centre
            resolved[-1] = min(resolved[-1], max_y)
            for i in range(n - 2, -1, -1):
                if resolved[i] > resolved[i + 1] - label_height:
                    resolved[i] = resolved[i + 1] - label_height

            # Pass 3: Clamp to top and push down again if needed
            resolved[0] = max(resolved[0], min_y)
            for i in range(1, n):
                if resolved[i] < resolved[i - 1] + label_height:
                    resolved[i] = resolved[i - 1] + label_height

    return resolved


def cover_placeholder_regions(img: Image.Image, label_positions: list[dict]) -> None:
    """
    Cover placeholder numbers (from image-gen) with sampled background color
    before drawing real labels.
    """
    if not label_positions:
        return
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for pos in label_positions:
        bbox = pos.get("cover_bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            samples = []
            for sx, sy in [
                (max(0, x1 - 3), max(0, y1 - 3)),
                (min(w - 1, x2 + 3), max(0, y1 - 3)),
                (max(0, x1 - 3), min(h - 1, y2 + 3)),
                (min(w - 1, x2 + 3), min(h - 1, y2 + 3)),
            ]:
                px = img.getpixel((sx, sy))
                if isinstance(px, int):
                    samples.append((px, px, px))
                else:
                    samples.append(px[:3])
            avg_r = sum(s[0] for s in samples) // len(samples)
            avg_g = sum(s[1] for s in samples) // len(samples)
            avg_b = sum(s[2] for s in samples) // len(samples)
            bg = (avg_r, avg_g, avg_b)
            draw.ellipse(bbox, fill=bg, outline=bg)


def render_labels_on_png(
    image_path: Path,
    label_positions: list[dict],
    style: LabelStyle = "boxed_text",
    session_id: str = "",
    cover_placeholders: bool = True,
) -> Path:
    """
    Render labels on the PNG image using Pillow.

    Args:
        image_path: Path to the raster image
        label_positions: List of dicts with label, point_x, point_y,
                         label_x, label_y, anchor
        style: Annotation style
        session_id: For output path
        cover_placeholders: If True, cover any placeholder regions first

    Returns:
        Path to the annotated PNG
    """
    img = Image.open(image_path).convert("RGB")
    if cover_placeholders:
        cover_placeholder_regions(img, label_positions)
    draw = ImageDraw.Draw(img)

    renderers = {
        "plain_text": _render_plain_text,
        "boxed_text": _render_boxed_text,
        "numbered": _render_numbered,
        "color_coded": _render_color_coded,
        "minimal": _render_minimal,
        "textbook": _render_textbook,
        "professional": _render_professional,
    }

    renderer = renderers.get(style, _render_boxed_text)
    renderer(draw, img, label_positions)

    # Save output
    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.png"
    img.save(str(out_path), "PNG")

    print(f"[label_renderer] Annotated PNG saved: {out_path}")
    return out_path


# ─── Leader line drawing helper ─────────────────────────────────


def _draw_leader_line(
    draw: ImageDraw.ImageDraw,
    px: int, py: int,
    lx: int, ly: int,
    anchor: str,
    color: str = "#555555",
    width: int = 1,
    label_text_width: int = 0,
    image_width: int = 0,
):
    """
    Draw a leader line from anatomical point (px,py) to label (lx,ly).

    Uses elbow routing when the vertical distance is large:
      point -> short horizontal to gutter -> diagonal to label level -> horizontal to label
    Falls back to a straight line when the angle is gentle.
    """
    # Determine the endpoint on the label side
    if anchor == "right":
        end_x = lx - label_text_width - 18 if label_text_width else lx - 12
    else:
        end_x = lx + label_text_width + 18 if label_text_width else lx + 12

    # Clamp end_x to image bounds
    if image_width > 0:
        end_x = max(4, min(end_x, image_width - 4))

    # Use elbow routing when the vertical distance is large relative to horizontal
    dy = abs(ly - py)
    dx = abs(end_x - px)

    if dy > 60 and dx > 30:
        # Elbow: point -> horizontal gutter -> diagonal to label
        # Gutter is 60% of the way from the point to the label column
        gutter_x = int(px + (end_x - px) * 0.35)
        # Three-segment line: point -> gutter at point_y -> gutter at label_y -> end
        draw.line([(px, py), (gutter_x, py)], fill=color, width=width)
        draw.line([(gutter_x, py), (end_x, ly)], fill=color, width=width)
    else:
        # Simple straight line for gentle angles
        draw.line([(px, py), (end_x, ly)], fill=color, width=width)


def _draw_dot(
    draw: ImageDraw.ImageDraw,
    px: int, py: int,
    radius: int = 4,
    fill: str = "#E74C3C",
    outline: str = "#FFFFFF",
):
    """Draw a small dot at the anatomical point."""
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        fill=fill,
        outline=outline,
        width=1,
    )


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    from_x: float, from_y: float,
    to_x: float, to_y: float,
    size: int = 10,
    fill: str = "#444444",
):
    """Draw a filled triangular arrowhead at (to_x, to_y) pointing from (from_x, from_y)."""
    dx = to_x - from_x
    dy = to_y - from_y
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux

    half_w = size * 0.38
    base_x = to_x - ux * size
    base_y = to_y - uy * size

    points = [
        (to_x, to_y),
        (base_x + perp_x * half_w, base_y + perp_y * half_w),
        (base_x - perp_x * half_w, base_y - perp_y * half_w),
    ]
    draw.polygon(points, fill=fill)


# ─── Rendering Styles ────────────────────────────────────────────


def _render_plain_text(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Simple text labels with clean leader lines and dots."""
    fs = _auto_font_size(*img.size)
    font = _load_font(fs)
    w, _ = img.size

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        anchor = pos.get("anchor", "right")

        tw, th = _text_size(font, text)

        # Leader line
        _draw_leader_line(draw, px, py, lx, ly, anchor,
                          color="#555555", width=1, label_text_width=tw,
                          image_width=w)

        # Dot at anatomical point
        _draw_dot(draw, px, py, radius=3, fill="#333333", outline="#333333")

        # Text (edge-clamped)
        if anchor == "right":
            text_x = max(4, lx - tw)
            draw.text((text_x, ly - th // 2), text, fill="#222222", font=font)
        else:
            text_x = min(w - tw - 4, lx)
            draw.text((text_x, ly - th // 2), text, fill="#222222", font=font)


def _render_boxed_text(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Text in rounded rectangles with leader lines — the default style."""
    fs = _auto_font_size(*img.size)
    font = _load_font(fs, bold=True)
    w, h = img.size

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        anchor = pos.get("anchor", "right")

        # Measure text
        tw, th = _text_size(font, text)
        pad_x = 7
        pad_y = 4

        # Box position (aligned to anchor side, edge-clamped)
        if anchor == "right":
            box_x2 = min(lx, w - 3)
            box_x1 = box_x2 - tw - pad_x * 2
            # Ensure box doesn't go off left edge
            if box_x1 < 4:
                box_x1 = 4
                box_x2 = box_x1 + tw + pad_x * 2
            text_x = box_x1 + pad_x
            line_end_x = box_x1  # leader connects to left edge of box
        else:
            box_x1 = max(lx, 3)
            box_x2 = box_x1 + tw + pad_x * 2
            # Ensure box doesn't go off right edge
            if box_x2 > w - 4:
                box_x2 = w - 4
                box_x1 = box_x2 - tw - pad_x * 2
            text_x = box_x1 + pad_x
            line_end_x = box_x2  # leader connects to right edge of box

        box_y1 = ly - th // 2 - pad_y
        box_y2 = ly + th // 2 + pad_y

        # Edge-clamp vertically
        if box_y1 < 2:
            shift = 2 - box_y1
            box_y1 += shift
            box_y2 += shift
            ly += shift
        if box_y2 > h - 2:
            shift = box_y2 - (h - 2)
            box_y1 -= shift
            box_y2 -= shift
            ly -= shift

        # Leader line (elbow-routed from point to box edge)
        dy = abs(ly - py)
        dx = abs(line_end_x - px)

        if dy > 60 and dx > 30:
            gutter_x = int(px + (line_end_x - px) * 0.35)
            draw.line([(px, py), (gutter_x, py)], fill="#666666", width=1)
            draw.line([(gutter_x, py), (line_end_x, ly)], fill="#666666", width=1)
        else:
            draw.line([(px, py), (line_end_x, ly)], fill="#666666", width=1)

        # Dot at anatomical point
        _draw_dot(draw, px, py, radius=4, fill="#E74C3C", outline="#FFFFFF")

        # Box background
        draw.rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=4,
            fill="white",
            outline="#888888",
            width=1,
        )

        # Text
        draw.text((text_x, box_y1 + pad_y), text, fill="#222222", font=font)


def _render_numbered(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Circled numbers on image with numbered legend below."""
    fs = _auto_font_size(*img.size)
    num_font = _load_font(max(11, fs - 2), bold=True)
    legend_font = _load_font(fs)
    legend_bold = _load_font(fs, bold=True)

    w, h = img.size

    # Draw numbered circles on the image at each point
    for i, pos in enumerate(positions):
        px, py = pos["point_x"], pos["point_y"]
        num = str(i + 1)

        # Circled number
        r = max(12, int(fs * 0.9))
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill="#1A56DB", outline="white", width=2,
        )

        # Centre the number text
        nw, nh = _text_size(num_font, num)
        draw.text((px - nw // 2, py - nh // 2), num, fill="white", font=num_font)

    # Legend at bottom
    line_h = fs + 8
    legend_height = len(positions) * line_h + 16
    legend_top = max(10, h - legend_height - 8)

    # Legend background
    draw.rectangle(
        [8, legend_top - 8, w - 8, h - 8],
        fill="white", outline="#CCCCCC",
    )

    for i, pos in enumerate(positions):
        y = legend_top + i * line_h
        num = str(i + 1)
        text = pos["label"]

        draw.text((20, y), f"{num}.", fill="#1A56DB", font=legend_bold)
        draw.text((48, y), text, fill="#222222", font=legend_font)


def _render_color_coded(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Coloured dots on image with matching coloured text labels."""
    fs = _auto_font_size(*img.size)
    font = _load_font(fs, bold=True)
    w, _ = img.size

    for i, pos in enumerate(positions):
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        color = COLORS[i % len(COLORS)]
        anchor = pos.get("anchor", "right")

        tw, _ = _text_size(font, text)

        # Leader line in matching colour
        _draw_leader_line(draw, px, py, lx, ly, anchor,
                          color=color, width=2, label_text_width=tw,
                          image_width=w)

        # Coloured dot on image
        _draw_dot(draw, px, py, radius=6, fill=color, outline="white")

        # Coloured text (edge-clamped)
        if anchor == "right":
            text_x = max(4, lx - tw)
            draw.text((text_x, ly - fs // 2), text, fill=color, font=font)
        else:
            text_x = min(w - tw - 4, lx)
            draw.text((text_x, ly - fs // 2), text, fill=color, font=font)


def _render_minimal(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Very thin lines, small text, minimal visual clutter."""
    fs = max(10, _auto_font_size(*img.size) - 2)
    font = _load_font(fs)
    w, _ = img.size

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        anchor = pos.get("anchor", "right")

        tw, _ = _text_size(font, text)

        # Thin leader line
        _draw_leader_line(draw, px, py, lx, ly, anchor,
                          color="#AAAAAA", width=1, label_text_width=tw,
                          image_width=w)

        # Small dot
        _draw_dot(draw, px, py, radius=2, fill="#999999", outline="#999999")

        # Small text (edge-clamped)
        if anchor == "right":
            text_x = max(4, lx - tw)
            draw.text((text_x, ly - fs // 2), text, fill="#666666", font=font)
        else:
            text_x = min(w - tw - 4, lx)
            draw.text((text_x, ly - fs // 2), text, fill="#666666", font=font)


def _render_textbook(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """Classic textbook look: bold labels with straight leader lines and underlines."""
    fs = _auto_font_size(*img.size)
    font = _load_font(fs, bold=True)
    w, h = img.size

    for pos in positions:
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = pos["label"]
        tw, th = _text_size(font, text)
        anchor = pos.get("anchor", "right")

        # Determine text position (edge-clamped)
        if anchor == "right":
            text_x = max(4, lx - tw)
            line_end_x = text_x - 8  # small gap before text
        else:
            text_x = min(w - tw - 4, lx)
            line_end_x = text_x + tw + 8  # small gap after text

        # Leader line from anatomical point to label (elbow-routed)
        dy = abs(ly - py)
        dx = abs(line_end_x - px)
        if dy > 60 and dx > 30:
            gutter_x = int(px + (line_end_x - px) * 0.35)
            draw.line([(px, py), (gutter_x, py)], fill="#333333", width=2)
            draw.line([(gutter_x, py), (line_end_x, ly)], fill="#333333", width=2)
        else:
            draw.line([(px, py), (line_end_x, ly)], fill="#333333", width=2)

        # Small square at anatomical point
        s = 3
        draw.rectangle([px - s, py - s, px + s, py + s], fill="#333333")

        # Bold text
        draw.text((text_x, ly - th // 2), text, fill="#1a1a1a", font=font)

        # Underline
        underline_y = ly + th // 2 + 2
        if anchor == "right":
            draw.line([(text_x, underline_y), (text_x + tw, underline_y)],
                      fill="#333333", width=1)
        else:
            draw.line([(text_x, underline_y), (text_x + tw, underline_y)],
                      fill="#333333", width=1)


def _render_professional(draw: ImageDraw.ImageDraw, img: Image.Image, positions: list[dict]):
    """
    Professional medical textbook annotation style.

    Designed to match high-quality scientific figure standards:
    - 2× supersampled rendering for razor-sharp text and lines
    - Thin leader lines: horizontal stub from label → clean diagonal to structure
    - Tiny endpoint dots at anatomical points — no arrowheads or colored markers
    - Bold text with clean white stroke outline (Pillow stroke_width) for
      readability on any background — replaces the old multi-offset halo
    - Generous spacing and precise alignment
    """
    w, h = img.size
    SCALE = 2  # Supersample factor for crisp anti-aliased rendering

    hi_w, hi_h = w * SCALE, h * SCALE

    # Transparent overlay at high resolution — draw annotations here
    overlay = Image.new("RGBA", (hi_w, hi_h), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)

    # ── Typography ──
    base_fs = _auto_font_size(w, h) + 3
    fs = base_fs * SCALE
    font = _load_font(fs, bold=True)

    # ── Palette (RGBA tuples for overlay drawing) ──
    LINE_CLR   = (50, 50, 50, 255)       # dark gray — thin leader lines
    TEXT_CLR   = (20, 20, 20, 255)       # near-black text
    DOT_CLR    = (50, 50, 50, 255)       # subtle endpoint dot
    STROKE_CLR = (255, 255, 255, 255)    # white text outline

    # ── Dimension constants (in hi-res pixels) ──
    lw       = SCALE                      # 1 px at final resolution
    dot_r    = SCALE + 1                  # tiny endpoint radius
    stroke_w = max(3, SCALE * 2)          # text outline width
    gap      = 5 * SCALE                  # text edge → line start
    stub     = 14 * SCALE                 # horizontal stub length
    margin   = 8 * SCALE                  # edge safety margin

    for pos in positions:
        px  = pos["point_x"] * SCALE
        py  = pos["point_y"] * SCALE
        lx  = pos["label_x"] * SCALE
        ly  = pos["label_y"] * SCALE
        text   = pos["label"]
        anchor = pos.get("anchor", "right")

        tw, th = _text_size(font, text)

        # ── Text position (edge-clamped) ──
        if anchor == "right":
            tx = max(margin, lx - tw)
        else:
            tx = min(hi_w - tw - margin, lx)
        ty = ly - th // 2
        ty = max(margin, min(hi_h - th - margin, ty))
        cy = ty + th // 2  # vertical centre of text line

        # ── Leader line: horizontal stub → diagonal to anatomical point ──
        if anchor == "right":
            sx = tx - gap           # stub start (text edge)
            ex = sx - stub          # stub end
        else:
            sx = tx + tw + gap
            ex = sx + stub
        ex = max(margin, min(hi_w - margin, ex))

        # Horizontal stub
        ov.line([(sx, cy), (ex, cy)], fill=LINE_CLR, width=lw)
        # Diagonal to anatomical point
        ov.line([(ex, cy), (px, py)], fill=LINE_CLR, width=lw)

        # Tiny dot at anatomical point
        ov.ellipse(
            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
            fill=DOT_CLR,
        )

        # ── Text with clean stroke-based outline ──
        ov.text(
            (tx, ty), text,
            fill=TEXT_CLR, font=font,
            stroke_width=stroke_w, stroke_fill=STROKE_CLR,
        )

    # ── Composite: downsample overlay and blend onto the original image ──
    overlay_sm = overlay.resize((w, h), Image.LANCZOS)
    base_rgba  = img.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay_sm)
    img.paste(composited.convert("RGB"))


# ─── SVG Export ──────────────────────────────────────────────────


def render_labels_as_svg(
    image_path: Path,
    label_positions: list[dict],
    style: LabelStyle = "boxed_text",
    session_id: str = "",
) -> Path:
    """
    Generate an SVG with the raster image embedded + vector label overlays.
    Labels are real <text> elements — editable, searchable, scalable.
    """
    with Image.open(image_path) as img:
        w, h = img.size

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    fs = _auto_font_size(w, h)

    # Build SVG
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                 f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">')

    # Embedded raster image
    lines.append(f'  <image width="{w}" height="{h}" '
                 f'href="data:image/png;base64,{b64}" />')

    # (No marker defs needed — professional style uses clean line endpoints)

    # Use larger font for professional style
    render_fs = fs + 3 if style == "professional" else fs

    # Style-dependent rendering
    lines.append('  <g id="annotations" font-family="Arial, Helvetica, sans-serif">')

    for i, pos in enumerate(label_positions):
        px, py = pos["point_x"], pos["point_y"]
        lx, ly = pos["label_x"], pos["label_y"]
        text = _svg_escape(pos["label"])
        anchor_side = pos.get("anchor", "right")
        anchor_attr = "end" if anchor_side == "right" else "start"

        if style == "professional":
            # ── Professional: clean stub→diagonal lines, tiny endpoints, crisp text ──
            approx_tw = len(text) * (render_fs * 0.6)
            gap_svg = 5
            stub_svg = 14

            if anchor_side == "right":
                sx = int(lx - approx_tw - gap_svg)   # stub start (near text)
                ex = sx - stub_svg                     # stub end
            else:
                sx = int(lx + approx_tw + gap_svg)
                ex = sx + stub_svg
            sx = max(4, min(sx, w - 4))
            ex = max(4, min(ex, w - 4))

            # Horizontal stub → diagonal to anatomical point
            lines.append(
                f'    <polyline points="{sx},{ly} {ex},{ly} {px},{py}" '
                f'fill="none" stroke="#333333" stroke-width="1" />'
            )

            # Tiny endpoint dot at anatomical point
            lines.append(
                f'    <circle cx="{px}" cy="{py}" r="2" fill="#333333" />'
            )

            # Bold text with white stroke outline
            lines.append(
                f'    <text x="{lx}" y="{ly + render_fs // 3}" '
                f'text-anchor="{anchor_attr}" '
                f'font-size="{render_fs}" font-weight="bold" '
                f'fill="#111111" stroke="white" stroke-width="4" '
                f'stroke-linejoin="round" paint-order="stroke">{text}</text>'
            )
        else:
            # ── All other styles: existing rendering ──
            # Colours
            if style == "color_coded":
                color = COLORS[i % len(COLORS)]
            else:
                color = "#555555"

            dot_color = COLORS[i % len(COLORS)] if style == "color_coded" else "#E74C3C"
            stroke_w = 2 if style in ("color_coded", "textbook") else 1

            # Leader line endpoint
            approx_tw = len(text) * (render_fs * 0.6)
            if anchor_side == "right":
                line_end_x = int(lx - approx_tw - 18)
            else:
                line_end_x = int(lx + approx_tw + 18)

            # Clamp to image bounds
            line_end_x = max(4, min(line_end_x, w - 4))

            # Elbow routing for SVG
            dy = abs(ly - py)
            dx = abs(line_end_x - px)
            if dy > 60 and dx > 30:
                gutter_x = int(px + (line_end_x - px) * 0.35)
                lines.append(f'    <polyline points="{px},{py} {gutter_x},{py} {line_end_x},{ly}" '
                             f'fill="none" stroke="{color}" stroke-width="{stroke_w}" />')
            else:
                lines.append(f'    <line x1="{px}" y1="{py}" x2="{line_end_x}" y2="{ly}" '
                             f'stroke="{color}" stroke-width="{stroke_w}" />')

            # Dot at anatomical point
            r = 4
            lines.append(f'    <circle cx="{px}" cy="{py}" r="{r}" '
                         f'fill="{dot_color}" stroke="white" stroke-width="1" />')

            # Text label
            text_color = color if style == "color_coded" else "#222222"
            font_weight = "bold" if style in ("boxed_text", "textbook", "color_coded") else "normal"

            # Background rect for boxed_text style
            if style == "boxed_text":
                padding = 5
                rw = int(approx_tw + padding * 2)
                rh = render_fs + padding * 2
                if anchor_attr == "end":
                    rx = int(lx - rw)
                else:
                    rx = int(lx - padding)
                ry = int(ly - render_fs // 2 - padding)
                # Edge-clamp
                rx = max(2, min(rx, w - rw - 2))
                ry = max(2, min(ry, h - rh - 2))
                lines.append(f'    <rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
                             f'rx="4" fill="white" fill-opacity="0.92" '
                             f'stroke="#888" stroke-width="1" />')

            lines.append(f'    <text x="{lx}" y="{ly + render_fs // 3}" '
                         f'text-anchor="{anchor_attr}" '
                         f'font-size="{render_fs}" font-weight="{font_weight}" '
                         f'fill="{text_color}">{text}</text>')

    lines.append('  </g>')
    lines.append('</svg>')

    svg_content = "\n".join(lines)

    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.svg"
    out_path.write_text(svg_content, encoding="utf-8")

    print(f"[label_renderer] Annotated SVG saved: {out_path}")
    return out_path


def _svg_escape(text: str) -> str:
    """Escape special characters for SVG text content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
