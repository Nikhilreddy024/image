"""
Stage 3: Placeholder Replacer
Detects numbered marker placeholders in the AI-generated image and replaces
them with the actual label text.

Architecture
────────────
Phase 1 · Marker Detection
    Use a Vision LLM to locate each numbered marker (1, 2, 3, …) in the
    generated image.  Returns centre-point coordinates for every marker.
    Falls back to EasyOCR if available, or grid-based heuristics.

Phase 2 · Marker Erasure
    Cover each detected marker with sampled background colour so the
    original number is invisible.

Phase 3 · Label Rendering
    Draw the real label text at each marker position using the model's
    chosen layout.  Leader lines and arrows are already baked into the
    image from generation — only the text is swapped.

Phase 4 · SVG Export
    Produce an SVG with the raster image embedded and real labels as
    editable <text> elements positioned over the marker locations.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
    FONT_PATH, FONT_BOLD_PATH, GENERATED_DIR,
)
from pipeline.utils import retry_on_rate_limit


# ────────────────────── Font Helpers ──────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD_PATH if bold else FONT_PATH
    try:
        return ImageFont.truetype(str(path), size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _text_size(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _auto_font_size(image_width: int, image_height: int) -> int:
    min_dim = min(image_width, image_height)
    return max(11, min(20, int(min_dim * 0.014)))


# ────────────────────── Detection Prompts ─────────────────────────

DETECT_MARKERS_PROMPT = """\
You are analysing a medical/scientific illustration that contains numbered \
annotation markers (1, 2, 3, …).  Each marker is a visible number in the \
image, connected to an anatomical structure via a leader line or arrow.

IMAGE DIMENSIONS: {width} × {height} pixels.
EXPECTED MARKERS: 1 through {num_markers}.

COORDINATE SYSTEM (0–1000 integer scale, BOTH axes):
  x = 0 → left edge      x = 1000 → right edge
  y = 0 → top edge       y = 1000 → bottom edge

TASK
For EACH numbered marker, locate the NUMBER TEXT in the image and report:
  1. The marker number.
  2. The centre coordinates of the marker number (NOT the anatomy it points to).

CRITICAL RULES
- Report the position of the displayed NUMBER, not the structure it labels.
- If a marker is inside a circle or box, report the centre of that circle/box.
- Every marker must have DIFFERENT coordinates.
- Coordinates are integers in 0–1000.

OUTPUT — JSON array with exactly {num_markers} entries, sorted by number:
[
  {{"number": 1, "x": <int>, "y": <int>}},
  {{"number": 2, "x": <int>, "y": <int>}},
  ...
]

Respond with ONLY the JSON array.  No markdown, no extra text."""


# ────────────────────── Main Entry Point ──────────────────────────

def detect_and_replace(
    image_path: Path,
    labels: list[str],
    session_id: str,
    provider: str | None = None,
) -> tuple[Path, Path, list[dict]]:
    """
    Full placeholder replacement pipeline:
      1. Detect numbered markers in the generated image.
      2. Erase markers from the raster.
      3. Render real label text at marker positions.
      4. Export as annotated PNG and SVG.

    Args:
        image_path: Path to the generated PNG with numbered placeholders.
        labels:     Ordered label strings (index 0 → marker 1, etc.).
        session_id: Session identifier for output directory.
        provider:   Vision LLM provider ('gemini' | 'openai').

    Returns:
        (annotated_png_path, annotated_svg_path, marker_positions)
        marker_positions is [{label, marker_x, marker_y, anchor}, …]
    """
    provider = provider or LLM_PROVIDER

    if not labels:
        # Nothing to replace — copy raster as-is
        out_dir = GENERATED_DIR / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        png_out = out_dir / "annotated.png"
        svg_out = out_dir / "annotated.svg"
        Image.open(image_path).save(str(png_out), "PNG")
        _write_passthrough_svg(image_path, png_out, svg_out)
        return png_out, svg_out, []

    with Image.open(image_path) as img:
        width, height = img.size

    # ── Phase 1: Detect marker positions ────────────────────────
    print(f"[placeholder_replacer] Phase 1: Detecting {len(labels)} markers via {provider}")
    markers = _detect_markers(image_path, len(labels), width, height, provider)
    print(f"[placeholder_replacer] Detected {len(markers)} / {len(labels)} markers")

    # Fill in any missing markers with heuristic positions
    markers = _fill_missing_markers(markers, labels, width, height)

    # Build label_positions: map marker number → label text
    label_positions = _build_label_positions(markers, labels, width, height)

    # ── Phase 2 + 3: Erase markers & render real labels (PNG) ──
    print("[placeholder_replacer] Phase 2+3: Replacing markers on PNG")
    png_out = _render_replacement_png(image_path, label_positions, session_id)

    # ── Phase 4: SVG export ────────────────────────────────────
    print("[placeholder_replacer] Phase 4: Generating SVG")
    svg_out = _render_replacement_svg(image_path, label_positions, session_id)

    print(f"[placeholder_replacer] Done — {len(label_positions)} labels replaced")
    return png_out, svg_out, label_positions


# ────────────────────── Marker Detection ──────────────────────────

def _detect_markers(
    image_path: Path,
    num_markers: int,
    width: int,
    height: int,
    provider: str,
) -> list[dict]:
    """
    Detect numbered markers using vision LLM.
    Returns list of {number, x_px, y_px} in pixel coordinates.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = DETECT_MARKERS_PROMPT.format(
        width=width, height=height, num_markers=num_markers,
    )

    # Run two detection rounds at different temperatures, take best
    best_result: list[dict] = []
    for temp in [0.05, 0.2]:
        try:
            raw = _call_vision(image_bytes, prompt, provider, temperature=temp)
            parsed = _parse_marker_result(raw, num_markers, width, height)
            if len(parsed) > len(best_result):
                best_result = parsed
            if len(parsed) >= num_markers:
                break
        except Exception as e:
            print(f"[placeholder_replacer] Detection round (temp={temp}) failed: {e}")

    # If vision didn't get enough, try OCR fallback
    if len(best_result) < num_markers:
        print("[placeholder_replacer] Trying OCR fallback…")
        ocr_result = _detect_markers_via_ocr(image_path, num_markers, width, height)
        # Merge: keep vision results, fill gaps from OCR
        found_nums = {m["number"] for m in best_result}
        for m in ocr_result:
            if m["number"] not in found_nums:
                best_result.append(m)
                found_nums.add(m["number"])

    return best_result


def _parse_marker_result(
    raw: list[dict],
    num_markers: int,
    width: int,
    height: int,
) -> list[dict]:
    """Parse vision LLM result into pixel-coordinate markers."""
    markers: list[dict] = []
    seen_nums: set[int] = set()

    for item in raw:
        num = item.get("number")
        x = item.get("x")
        y = item.get("y")
        if num is None or x is None or y is None:
            continue
        try:
            num = int(num)
            xi = max(0, min(1000, int(float(x))))
            yi = max(0, min(1000, int(float(y))))
        except (TypeError, ValueError):
            continue
        if num < 1 or num > num_markers or num in seen_nums:
            continue

        px = int(xi / 1000.0 * (width - 1))
        py = int(yi / 1000.0 * (height - 1))
        px = max(10, min(px, width - 10))
        py = max(10, min(py, height - 10))

        markers.append({"number": num, "x_px": px, "y_px": py})
        seen_nums.add(num)

    return sorted(markers, key=lambda m: m["number"])


def _detect_markers_via_ocr(
    image_path: Path,
    num_markers: int,
    width: int,
    height: int,
) -> list[dict]:
    """Fall back to EasyOCR for digit detection."""
    try:
        import easyocr
    except ImportError:
        print("[placeholder_replacer] EasyOCR not installed — skipping OCR fallback")
        return []

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(str(image_path), allowlist="0123456789", paragraph=False)

    markers: list[dict] = []
    seen: set[int] = set()
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text.isdigit():
            continue
        num = int(text)
        if num < 1 or num > num_markers or num in seen:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        cx = max(10, min(cx, width - 10))
        cy = max(10, min(cy, height - 10))
        markers.append({"number": num, "x_px": cx, "y_px": cy})
        seen.add(num)

    return sorted(markers, key=lambda m: m["number"])


def _fill_missing_markers(
    markers: list[dict],
    labels: list[str],
    width: int,
    height: int,
) -> list[dict]:
    """Fill in any undetected markers with heuristic grid positions."""
    found_nums = {m["number"] for m in markers}
    missing = [i + 1 for i in range(len(labels)) if (i + 1) not in found_nums]

    if not missing:
        return markers

    print(f"[placeholder_replacer] Filling {len(missing)} missing markers with heuristics")
    # Place missing markers in a column on the right margin
    margin_x = int(width * 0.88)
    n_missing = len(missing)
    for idx, num in enumerate(missing):
        frac_y = (idx + 1) / (n_missing + 1)
        y = int(height * (0.05 + 0.9 * frac_y))
        markers.append({"number": num, "x_px": margin_x, "y_px": y})

    return sorted(markers, key=lambda m: m["number"])


# ────────────────────── Label Position Building ───────────────────

def _build_label_positions(
    markers: list[dict],
    labels: list[str],
    width: int,
    height: int,
) -> list[dict]:
    """
    Map detected markers to label text and determine anchor direction.
    Returns [{label, marker_x, marker_y, anchor}, …].
    """
    mid_x = width / 2
    positions: list[dict] = []

    for marker in markers:
        num = marker["number"]
        if num < 1 or num > len(labels):
            continue
        label = labels[num - 1]
        mx, my = marker["x_px"], marker["y_px"]
        # Anchor direction: right-anchor if marker is on right half, else left
        anchor = "right" if mx > mid_x else "left"

        positions.append({
            "label": label,
            "marker_x": mx,
            "marker_y": my,
            "anchor": anchor,
        })

    return positions


# ────────────────────── PNG Replacement Rendering ─────────────────

def _render_replacement_png(
    image_path: Path,
    label_positions: list[dict],
    session_id: str,
) -> Path:
    """
    Replace numbered markers with real label text on the PNG.
    Uses 2× supersampled rendering for crisp text.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    SCALE = 2

    # ── Erase markers: cover each marker area with sampled background ──
    _erase_markers(img, label_positions)

    # ── Render labels at marker positions with supersampled overlay ──
    hi_w, hi_h = w * SCALE, h * SCALE
    overlay = Image.new("RGBA", (hi_w, hi_h), (0, 0, 0, 0))
    ov = ImageDraw.Draw(overlay)

    base_fs = _auto_font_size(w, h) + 2
    fs = base_fs * SCALE
    font = _load_font(fs, bold=True)

    LINE_CLR   = (50, 50, 50, 255)
    TEXT_CLR   = (20, 20, 20, 255)
    STROKE_CLR = (255, 255, 255, 255)
    BG_CLR     = (255, 255, 255, 220)
    BORDER_CLR = (180, 180, 180, 200)

    stroke_w = max(3, SCALE * 2)
    pad_x = 7 * SCALE
    pad_y = 4 * SCALE
    margin = 6 * SCALE

    for pos in label_positions:
        mx = pos["marker_x"] * SCALE
        my = pos["marker_y"] * SCALE
        text = pos["label"]
        anchor = pos["anchor"]

        tw, th = _text_size(font, text)

        # ── Position text box centred vertically on marker, anchored to side ──
        if anchor == "right":
            # Text extends to the left of the marker position
            box_x2 = min(mx + pad_x, hi_w - margin)
            box_x1 = box_x2 - tw - pad_x * 2
            if box_x1 < margin:
                box_x1 = margin
                box_x2 = box_x1 + tw + pad_x * 2
            tx = box_x1 + pad_x
        else:
            # Text extends to the right of the marker position
            box_x1 = max(mx - pad_x, margin)
            box_x2 = box_x1 + tw + pad_x * 2
            if box_x2 > hi_w - margin:
                box_x2 = hi_w - margin
                box_x1 = box_x2 - tw - pad_x * 2
            tx = box_x1 + pad_x

        box_y1 = my - th // 2 - pad_y
        box_y2 = my + th // 2 + pad_y

        # Edge-clamp vertically
        if box_y1 < margin:
            shift = margin - box_y1
            box_y1 += shift
            box_y2 += shift
        if box_y2 > hi_h - margin:
            shift = box_y2 - (hi_h - margin)
            box_y1 -= shift
            box_y2 -= shift

        ty = box_y1 + pad_y

        # ── Draw background box (semi-transparent white) ──
        ov.rounded_rectangle(
            [int(box_x1), int(box_y1), int(box_x2), int(box_y2)],
            radius=4 * SCALE,
            fill=BG_CLR,
            outline=BORDER_CLR,
            width=SCALE,
        )

        # ── Draw label text with stroke outline ──
        ov.text(
            (int(tx), int(ty)),
            text,
            fill=TEXT_CLR,
            font=font,
            stroke_width=stroke_w,
            stroke_fill=STROKE_CLR,
        )

    # Composite overlay onto image
    overlay_sm = overlay.resize((w, h), Image.LANCZOS)
    base_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay_sm)
    result = composited.convert("RGB")

    # Save
    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.png"
    result.save(str(out_path), "PNG")
    print(f"[placeholder_replacer] Annotated PNG saved: {out_path}")
    return out_path


def _erase_markers(img: Image.Image, label_positions: list[dict]) -> None:
    """
    Cover each numbered marker with sampled background colour.
    Uses a small circular region centred on the marker position.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Estimate marker size from image dimensions
    marker_radius = max(12, int(min(w, h) * 0.018))

    for pos in label_positions:
        mx, my = pos["marker_x"], pos["marker_y"]

        # Sample background colour from points around the marker
        samples: list[tuple[int, int, int]] = []
        offsets = [
            (-marker_radius - 4, -marker_radius - 4),
            (marker_radius + 4, -marker_radius - 4),
            (-marker_radius - 4, marker_radius + 4),
            (marker_radius + 4, marker_radius + 4),
            (-marker_radius - 6, 0),
            (marker_radius + 6, 0),
        ]
        for dx, dy in offsets:
            sx = max(0, min(w - 1, mx + dx))
            sy = max(0, min(h - 1, my + dy))
            px = img.getpixel((sx, sy))
            if isinstance(px, int):
                samples.append((px, px, px))
            else:
                samples.append(px[:3])

        # Use median colour for robustness
        avg_r = sorted(s[0] for s in samples)[len(samples) // 2]
        avg_g = sorted(s[1] for s in samples)[len(samples) // 2]
        avg_b = sorted(s[2] for s in samples)[len(samples) // 2]
        bg = (avg_r, avg_g, avg_b)

        # Cover the marker area with a filled ellipse
        r = marker_radius + 2
        draw.ellipse(
            [mx - r, my - r, mx + r, my + r],
            fill=bg,
            outline=bg,
        )


# ────────────────────── SVG Export ────────────────────────────────

def _render_replacement_svg(
    image_path: Path,
    label_positions: list[dict],
    session_id: str,
) -> Path:
    """
    Generate an SVG with the raster image embedded and real labels as
    editable <text> elements positioned over the marker locations.
    """
    with Image.open(image_path) as img:
        w, h = img.size

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    fs = _auto_font_size(w, h) + 2

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    )

    # Embedded raster image (original with markers — they'll be covered by rect+text)
    lines.append(
        f'  <image width="{w}" height="{h}" '
        f'href="data:image/png;base64,{b64}" />'
    )

    lines.append(
        '  <g id="label-replacements" '
        'font-family="Arial, Helvetica, sans-serif">'
    )

    pad = 5
    marker_r = max(10, int(min(w, h) * 0.016))

    for pos in label_positions:
        mx, my = pos["marker_x"], pos["marker_y"]
        text = _svg_escape(pos["label"])
        anchor = pos["anchor"]

        # Cover the original marker number with an opaque circle
        lines.append(
            f'    <circle cx="{mx}" cy="{my}" r="{marker_r}" '
            f'fill="white" stroke="none" />'
        )

        approx_tw = len(text) * (fs * 0.6)
        box_h = fs + pad * 2

        if anchor == "right":
            box_x = int(mx - approx_tw - pad * 2)
            box_x = max(2, box_x)
            text_anchor = "end"
            text_x = int(mx - pad)
        else:
            box_x = int(mx)
            text_anchor = "start"
            text_x = int(mx + pad)

        box_y = int(my - fs // 2 - pad)
        box_w = int(approx_tw + pad * 2)

        # Clamp to image bounds
        box_x = max(2, min(box_x, w - box_w - 2))
        box_y = max(2, min(box_y, h - box_h - 2))

        # Background rect for readability
        lines.append(
            f'    <rect x="{box_x}" y="{box_y}" '
            f'width="{box_w}" height="{box_h}" '
            f'rx="4" fill="white" fill-opacity="0.88" '
            f'stroke="#aaa" stroke-width="0.5" />'
        )

        # Editable text element
        lines.append(
            f'    <text x="{text_x}" y="{my + fs // 3}" '
            f'text-anchor="{text_anchor}" '
            f'font-size="{fs}" font-weight="bold" '
            f'fill="#111" stroke="white" stroke-width="3" '
            f'stroke-linejoin="round" paint-order="stroke">{text}</text>'
        )

    lines.append('  </g>')
    lines.append('</svg>')

    svg_content = "\n".join(lines)

    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotated.svg"
    out_path.write_text(svg_content, encoding="utf-8")
    print(f"[placeholder_replacer] Annotated SVG saved: {out_path}")
    return out_path


def _write_passthrough_svg(
    image_path: Path,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Write a minimal SVG wrapping just the raster image (no labels)."""
    with Image.open(image_path) as img:
        w, h = img.size
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<image width="{w}" height="{h}" href="data:image/png;base64,{b64}" />'
        f'</svg>'
    )
    svg_path.write_text(svg, encoding="utf-8")


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ────────────────────── Vision LLM Calls ─────────────────────────

def _call_vision(
    image_bytes: bytes,
    prompt: str,
    provider: str,
    temperature: float = 0.1,
) -> list[dict]:
    """Route to the appropriate vision model and parse JSON array result."""
    if provider == "gemini":
        return _call_gemini_vision(image_bytes, prompt, temperature)
    elif provider == "openai":
        return _call_openai_vision(image_bytes, prompt, temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _call_gemini_vision(
    image_bytes: bytes,
    prompt: str,
    temperature: float = 0.1,
) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return _parse_json_array(response.text or "")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _call_openai_vision(
    image_bytes: bytes,
    prompt: str,
    temperature: float = 0.1,
) -> list[dict]:
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        temperature=temperature,
        max_tokens=4096,
    )
    return _parse_json_array(response.choices[0].message.content or "")


def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from VLM response, with multiple fallbacks."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Attempt 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract JSON array from surrounding text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    # Attempt 3: Fix common issues (trailing commas)
    if start >= 0 and end > start:
        import re
        candidate = text[start:end + 1]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    print(f"[placeholder_replacer] Failed to parse JSON: {text[:300]}")
    return []
