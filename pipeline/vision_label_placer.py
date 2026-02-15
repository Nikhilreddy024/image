"""
Stage 3 – Vision-Based Label Placer  (Ensemble Detection + Iterative Verification)

Architecture
────────────
Phase 1 · Ensemble Detection
    Run *N* chain-of-thought detection calls at varying temperatures.
    Each call forces the VLM to *reason* about every structure's visual
    location before emitting coordinates.  Median coordinates across
    rounds give a robust consensus that smooths out per-call noise.

Phase 2 · Iterative Verification
    Overlay numbered dots at the consensus positions and send back to
    the VLM for strict verification.  Any corrections are applied, and
    if changes were made the verification repeats (up to *K* rounds)
    until positions stabilise or the budget is exhausted.

Phase 3 · Spatial Post-processing
    Algorithmic checks:  minimum inter-label distance, duplicate
    detection, edge clamping, and cluster dispersal.

Typical API cost:  N + K calls  (default 3 + 2 = 5 max).
Priority:  accuracy over cost — the user has explicitly requested this.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
    VISION_ENSEMBLE_ROUNDS, VISION_VERIFY_ROUNDS,
)
from pipeline.utils import retry_on_rate_limit

# ────────────── Tunable knobs ──────────────────────────────────
ENSEMBLE_ROUNDS = VISION_ENSEMBLE_ROUNDS     # from config / env
VERIFY_MAX_ROUNDS = VISION_VERIFY_ROUNDS     # from config / env
ENSEMBLE_TEMPERATURES = [0.05, 0.20, 0.40]   # One per ensemble round
MIN_POINT_DISTANCE_1000 = 25  # Minimum distance on 0-1000 scale between any two points
EDGE_CLAMP_PX = 10           # Keep points this many pixels from the image edge

# ─────────────────────────── Prompts ───────────────────────────

DETECT_PROMPT_COT = """\
You are an expert medical / scientific illustration analyst performing \
precise anatomical structure localisation.

CONTEXT
The image is a medical or scientific diagram.
It shows: {description}
Image dimensions: {width} x {height} pixels.

COORDINATE SYSTEM  (0-1000 integer scale, BOTH axes)
  x = 0   -> left edge       x = 1000 -> right edge
  y = 0   -> top edge        y = 1000 -> bottom edge

TASK
For EACH of the {num_labels} structures listed below, you must:
  1. FIRST describe where you see this structure in the image — which \
area of the image, which panel/view (if there are multiple), what visual \
features identify it (colour, shape, texture).
  2. THEN provide the precise coordinates of the geometric centre (or \
most representative visible point) of that structure AS DRAWN.

CRITICAL RULES
1. Study the actual visual content very carefully. Structures are visible \
drawn elements: coloured regions, distinct shapes, lines, or shaded areas.
2. MULTI-PANEL IMAGES: If the image has multiple panels or views side-by-side \
(e.g. skin view, muscle view, skeletal view), you MUST identify which panel \
each structure belongs to and place the coordinate WITHIN that panel.  \
Each panel occupies a different horizontal region of the image.
3. Every structure MUST have a DIFFERENT coordinate pair. Do NOT place \
multiple labels at the same point.  Coordinates should be SPREAD across \
the image reflecting actual anatomy.
4. For structures that span a large area, pick the visual centre of that \
specific structure.
5. If a structure has a clear boundary, the point should be INSIDE that \
boundary, not on its edge.
6. Coordinates MUST be integers in 0-1000.
7. Double-check each coordinate by mentally mapping it back to the image \
(e.g., x=500 is the horizontal centre, y=200 is near the top).

STRUCTURES (in order):
{label_list}

OUTPUT — JSON array with EXACTLY {num_labels} entries, same order:
[
  {{"label": "exact label text", "reasoning": "brief description of location", "x": <int>, "y": <int>}},
  ...
]

Respond with ONLY the JSON array.  No markdown fences, no extra text."""


VERIFY_PROMPT_V2 = """\
You are verifying label-placement accuracy on a medical illustration.

CONTEXT
The image shows a medical / scientific diagram with numbered RED DOTS \
overlaid at proposed label positions.  Each dot has a white number.

The illustration depicts: {description}

CURRENT PLACEMENTS (dot number -> label -> coordinate on 0-1000 scale):
{assignments}

VERIFICATION TASK
For EACH numbered dot, carefully evaluate:
1. Look at the dot in the image.  What anatomical structure or region \
is the dot actually sitting ON?
2. Where should this label's dot actually be?  Is the dot on the \
CORRECT structure?
3. If the dot is on the WRONG structure or significantly displaced \
(>25 units on the 0-1000 scale from the true centre), mark it as \
INCORRECT and provide corrected coordinates.

COORDINATE SYSTEM (0-1000 scale):
  x = 0 -> left edge    x = 1000 -> right edge
  y = 0 -> top edge     y = 1000 -> bottom edge

OUTPUT — JSON array, one entry per label (same order):

For CORRECT placements:
  {{"label": "...", "correct": true, "sitting_on": "what the dot is on"}}

For INCORRECT placements:
  {{"label": "...", "correct": false, "sitting_on": "what the dot is on", \
"should_be": "correct structure location", "x": <int>, "y": <int>}}

Be STRICT:  if a dot is even slightly on the wrong structure, mark it \
incorrect and provide the correct coordinates.

Respond with ONLY the JSON array.  No markdown, no explanation."""


EXTRACT_LABELS_PROMPT = """\
You are analysing a medical / scientific illustration that has NO text on it.

TASK
Identify every anatomical structure, organ, tissue, or notable region that \
should be labelled in a textbook-quality figure.  For each one provide:
  1. A short, standard anatomical / scientific term.
  2. The precise point where a leader line should connect (centre of structure).

Use a 0-1000 integer coordinate scale:
  x = 0 -> left edge   x = 1000 -> right edge
  y = 0 -> top edge    y = 1000 -> bottom edge

Image dimensions: {width} x {height} pixels.

Return a JSON array (5-30 entries), ordered top-to-bottom then left-to-right:
[{{"label": "Structure name", "x": <int>, "y": <int>}}, ...]

Output ONLY the JSON array."""


# ─────────────────────── Verification-image builder ─────────────────────────

def _build_verification_image(
    image_bytes: bytes,
    positions: list[dict],
) -> bytes:
    """
    Draw numbered red dots at each proposed label position on the original
    image.  Used for Phase 2 (verification).
    Dots are deliberately large and high-contrast for VLM readability.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Larger dots for better VLM visibility
    dot_radius = max(16, int(min(w, h) * 0.022))

    try:
        dot_font = ImageFont.truetype(
            "arialbd.ttf", max(16, int(min(w, h) * 0.022))
        )
    except (OSError, IOError):
        try:
            dot_font = ImageFont.truetype("arial.ttf", max(16, int(min(w, h) * 0.022)))
        except (OSError, IOError):
            dot_font = ImageFont.load_default()

    for i, pos in enumerate(positions):
        px, py = pos["point_x"], pos["point_y"]

        # White halo for contrast on dark backgrounds
        halo_r = dot_radius + 3
        draw.ellipse(
            [px - halo_r, py - halo_r, px + halo_r, py + halo_r],
            fill=(255, 255, 255, 180),
        )

        # Red dot with thick white border
        draw.ellipse(
            [px - dot_radius, py - dot_radius,
             px + dot_radius, py + dot_radius],
            fill=(220, 30, 30, 240),
            outline=(255, 255, 255, 255),
            width=3,
        )

        # Centred number
        num = str(i + 1)
        bbox = dot_font.getbbox(num)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (px - tw // 2, py - th // 2 - 1),
            num,
            fill=(255, 255, 255, 255),
            font=dot_font,
        )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────── Coordinate helpers ─────────────────────────

def _coords_1000_to_pixels(
    x_1000: int, y_1000: int, width: int, height: int,
) -> tuple[int, int]:
    """Convert 0-1000 scale coordinates to pixel coordinates."""
    px = int(x_1000 / 1000.0 * (width - 1))
    py = int(y_1000 / 1000.0 * (height - 1))
    px = max(EDGE_CLAMP_PX, min(px, width - EDGE_CLAMP_PX))
    py = max(EDGE_CLAMP_PX, min(py, height - EDGE_CLAMP_PX))
    return px, py


def _pixels_to_1000(
    px: int, py: int, width: int, height: int,
) -> tuple[int, int]:
    """Convert pixel coordinates to 0-1000 scale."""
    x_1000 = int(px / max(width - 1, 1) * 1000)
    y_1000 = int(py / max(height - 1, 1) * 1000)
    return max(0, min(1000, x_1000)), max(0, min(1000, y_1000))


# ─────────────────────── Extract labels (auto-detect) ─────────────────────────

def extract_labels_from_image(
    image_path: Path,
    provider: str | None = None,
) -> list[dict]:
    """
    Single vision call: identify structures and their positions in the image.
    Returns: [{label, point_x, point_y}, ...] in pixel coordinates.
    """
    provider = provider or LLM_PROVIDER

    with Image.open(image_path) as img:
        width, height = img.size

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = EXTRACT_LABELS_PROMPT.format(width=width, height=height)
    print(f"[vision_placer] Extracting labels from image ({width}x{height}) -> {provider}")

    raw_list = _call_vision(image_bytes, prompt, provider, temperature=0.1)
    positions: list[dict] = []

    for item in raw_list:
        label = item.get("label") or item.get("label_text", "")
        if not label or not isinstance(label, str):
            continue
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        try:
            xi = int(float(x))
            yi = int(float(y))
        except (TypeError, ValueError):
            continue
        xi = max(0, min(1000, xi))
        yi = max(0, min(1000, yi))
        px, py = _coords_1000_to_pixels(xi, yi, width, height)
        positions.append({
            "label": label.strip(),
            "point_x": px,
            "point_y": py,
        })

    print(f"[vision_placer] Extracted {len(positions)} labels from image")
    return positions


# ─────────────────────── Main entry point ─────────────────────────

def locate_labels(
    image_path: Path,
    labels: list[str],
    description: str,
    provider: str | None = None,
) -> list[dict]:
    """
    Multi-phase approach for accurate label placement:

    Phase 1:  Ensemble detection — N chain-of-thought calls, median coords.
    Phase 2:  Iterative verification — overlay dots, verify/correct, repeat.
    Phase 3:  Spatial validation — algorithmic post-processing.

    Returns list of dicts: [{label, point_x, point_y}, ...]
    """
    provider = provider or LLM_PROVIDER

    if not labels:
        return []

    with Image.open(image_path) as img:
        width, height = img.size

    with open(image_path, "rb") as f:
        original_bytes = f.read()

    # ── Phase 1: Ensemble Detection ────────────────────────────
    print(f"[vision_placer] Phase 1: Ensemble detection ({ENSEMBLE_ROUNDS} rounds) "
          f"for {len(labels)} structures -> {provider}")

    detect_prompt = DETECT_PROMPT_COT.format(
        description=description,
        width=width,
        height=height,
        num_labels=len(labels),
        label_list="\n".join(f"  {i+1}. {lbl}" for i, lbl in enumerate(labels)),
    )

    ensemble_results: list[list[dict]] = []
    for round_idx in range(ENSEMBLE_ROUNDS):
        temp = ENSEMBLE_TEMPERATURES[round_idx % len(ENSEMBLE_TEMPERATURES)]
        print(f"[vision_placer]   Round {round_idx+1}/{ENSEMBLE_ROUNDS} (temp={temp})")
        try:
            raw_result = _call_vision(original_bytes, detect_prompt, provider, temperature=temp)
            positions = _parse_detect_result(raw_result, labels, width, height)
            if positions:
                ensemble_results.append(positions)
                print(f"[vision_placer]   -> Got {len(positions)} positions")
        except Exception as e:
            print(f"[vision_placer]   Round {round_idx+1} failed: {e}")

    if not ensemble_results:
        print("[vision_placer] WARNING: All ensemble rounds failed, using fallback")
        return _fallback_positions(labels, width, height)

    # Compute consensus via median
    consensus = _compute_consensus(ensemble_results, labels, width, height)
    print(f"[vision_placer] Ensemble consensus for {len(consensus)} labels:")
    for pos in consensus:
        x1k, y1k = _pixels_to_1000(pos["point_x"], pos["point_y"], width, height)
        print(f"[vision_placer]   {pos['label']} -> ({pos['point_x']}, {pos['point_y']}) "
              f"[{x1k}, {y1k}]")

    # ── Phase 2: Iterative Verification ────────────────────────
    if len(consensus) >= 2:
        consensus = _iterative_verify(
            original_bytes, consensus, labels, description,
            width, height, provider,
        )

    # ── Phase 3: Spatial Post-processing ──────────────────────
    consensus = _spatial_validation(consensus, width, height)

    # Ensure label ordering matches the original list
    label_order = {lbl: i for i, lbl in enumerate(labels)}
    consensus.sort(key=lambda p: label_order.get(p["label"], 999))

    print(f"[vision_placer] Final: {len(consensus)} labels located successfully")
    return consensus


# ─────────────────────── Ensemble consensus ─────────────────────────

def _compute_consensus(
    all_round_results: list[list[dict]],
    labels: list[str],
    width: int,
    height: int,
) -> list[dict]:
    """
    Compute median coordinates from multiple detection rounds.
    For each label, collect all detected (x, y) pairs and take the median
    of x-coords and y-coords independently.  This is robust against
    outliers from any single round.
    """
    consensus: list[dict] = []

    for label in labels:
        xs: list[int] = []
        ys: list[int] = []

        for round_result in all_round_results:
            for pos in round_result:
                if pos["label"] == label:
                    xs.append(pos["point_x"])
                    ys.append(pos["point_y"])
                    break

        if xs and ys:
            # Median is more robust than mean against outliers
            med_x = int(median(xs))
            med_y = int(median(ys))

            # Log spread for debugging
            if len(xs) > 1:
                spread_x = max(xs) - min(xs)
                spread_y = max(ys) - min(ys)
                if spread_x > 80 or spread_y > 80:
                    print(f"[vision_placer]   WARNING: High spread for '{label}': "
                          f"dx={spread_x}, dy={spread_y} "
                          f"(values: x={xs}, y={ys})")

            consensus.append({
                "label": label,
                "point_x": med_x,
                "point_y": med_y,
            })
        else:
            # Label not found in any round — use fallback
            print(f"[vision_placer]   WARNING: '{label}' not found in any round")
            consensus.append({
                "label": label,
                "point_x": width // 2,
                "point_y": height // 2,
            })

    return consensus


# ─────────────────────── Iterative Verification ─────────────────────────

def _iterative_verify(
    original_bytes: bytes,
    positions: list[dict],
    labels: list[str],
    description: str,
    width: int,
    height: int,
    provider: str,
) -> list[dict]:
    """
    Run verification up to VERIFY_MAX_ROUNDS times.
    Each round draws dots, asks VLM to check, applies corrections.
    Stops when no corrections are needed or budget is exhausted.
    """
    current_positions = list(positions)

    for round_num in range(1, VERIFY_MAX_ROUNDS + 1):
        print(f"[vision_placer] Phase 2: Verification round {round_num}/{VERIFY_MAX_ROUNDS}")

        verify_img = _build_verification_image(original_bytes, current_positions)

        # Build assignment descriptions using 0-1000 scale for the VLM
        assignments = []
        for i, pos in enumerate(current_positions):
            x1k, y1k = _pixels_to_1000(pos["point_x"], pos["point_y"], width, height)
            assignments.append(
                f"  Dot {i+1}: \"{pos['label']}\" at ({x1k}, {y1k}) on 0-1000 scale"
            )

        verify_prompt = VERIFY_PROMPT_V2.format(
            description=description,
            assignments="\n".join(assignments),
        )

        try:
            verify_result = _call_vision(verify_img, verify_prompt, provider, temperature=0.1)
        except Exception as e:
            print(f"[vision_placer]   Verification call failed: {e}, keeping current positions")
            break

        # Parse corrections
        corrections = _parse_verification(verify_result, labels, width, height)

        if not corrections:
            print(f"[vision_placer]   All positions verified correct (round {round_num})")
            break

        print(f"[vision_placer]   Round {round_num}: {len(corrections)} corrections")
        for label, (px, py) in corrections.items():
            for pos in current_positions:
                if pos["label"] == label:
                    old_x, old_y = pos["point_x"], pos["point_y"]
                    pos["point_x"] = px
                    pos["point_y"] = py
                    print(f"[vision_placer]     CORRECTED: {label} "
                          f"({old_x},{old_y}) -> ({px},{py})")
                    break

    return current_positions


def _parse_verification(
    verify_result: list[dict],
    labels: list[str],
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    """Parse verification response and extract corrections."""
    corrections: dict[str, tuple[int, int]] = {}
    label_lower_map = {lbl.lower().strip(): lbl for lbl in labels}

    for item in verify_result:
        raw_lbl = str(item.get("label", "")).strip()
        matched = raw_lbl if raw_lbl in labels else label_lower_map.get(raw_lbl.lower())

        correct = item.get("correct", True)
        if not correct and matched:
            x = item.get("x")
            y = item.get("y")
            if x is not None and y is not None:
                try:
                    xi = int(float(x))
                    yi = int(float(y))
                    xi = max(0, min(1000, xi))
                    yi = max(0, min(1000, yi))
                    px, py = _coords_1000_to_pixels(xi, yi, width, height)
                    corrections[matched] = (px, py)
                except (TypeError, ValueError):
                    pass

    return corrections


# ─────────────────────── Spatial Post-processing ─────────────────────────

def _spatial_validation(
    positions: list[dict],
    width: int,
    height: int,
) -> list[dict]:
    """
    Algorithmic post-processing to enforce spatial constraints:
    1. Edge clamping — keep points away from image borders
    2. Deduplication — merge near-identical coordinates
    3. Minimum distance — nudge apart points that are too close
    """
    if not positions:
        return positions

    # 1. Edge clamping
    for pos in positions:
        pos["point_x"] = max(EDGE_CLAMP_PX, min(pos["point_x"], width - EDGE_CLAMP_PX))
        pos["point_y"] = max(EDGE_CLAMP_PX, min(pos["point_y"], height - EDGE_CLAMP_PX))

    # 2. Minimum distance enforcement (on 0-1000 scale)
    min_dist = MIN_POINT_DISTANCE_1000
    max_nudge_iterations = 10

    for _ in range(max_nudge_iterations):
        nudged = False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1k_i, y1k_i = _pixels_to_1000(
                    positions[i]["point_x"], positions[i]["point_y"], width, height
                )
                x1k_j, y1k_j = _pixels_to_1000(
                    positions[j]["point_x"], positions[j]["point_y"], width, height
                )

                dx = x1k_j - x1k_i
                dy = y1k_j - y1k_i
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_dist and dist > 0:
                    # Push apart along the line connecting them
                    push = (min_dist - dist) / 2.0 + 1.0
                    ratio = push / dist
                    nudge_x = dx * ratio
                    nudge_y = dy * ratio

                    # Convert back to pixels and apply
                    nx_i = max(0, min(1000, int(x1k_i - nudge_x)))
                    ny_i = max(0, min(1000, int(y1k_i - nudge_y)))
                    nx_j = max(0, min(1000, int(x1k_j + nudge_x)))
                    ny_j = max(0, min(1000, int(y1k_j + nudge_y)))

                    positions[i]["point_x"], positions[i]["point_y"] = \
                        _coords_1000_to_pixels(nx_i, ny_i, width, height)
                    positions[j]["point_x"], positions[j]["point_y"] = \
                        _coords_1000_to_pixels(nx_j, ny_j, width, height)
                    nudged = True

        if not nudged:
            break

    return positions


def _fallback_positions(
    labels: list[str],
    width: int,
    height: int,
) -> list[dict]:
    """Generate evenly-spaced fallback positions when detection fails entirely."""
    positions = []
    n = len(labels)
    for i, lbl in enumerate(labels):
        frac_y = (i + 1) / (n + 1)
        positions.append({
            "label": lbl,
            "point_x": width // 2,
            "point_y": int(height * frac_y),
        })
    return positions


# ─────────────────────── Pass 1 parsing ─────────────────────────

def _parse_detect_result(
    result: list[dict],
    labels: list[str],
    width: int,
    height: int,
) -> list[dict]:
    """Parse detection result (with optional reasoning field) into pixel positions."""
    label_lower_map = {lbl.lower().strip(): lbl for lbl in labels}
    positions: list[dict] = []
    found_labels: set[str] = set()

    for item in result:
        raw_lbl = str(item.get("label", "")).strip()
        # Try exact match, then case-insensitive, then partial match
        matched_lbl = raw_lbl if raw_lbl in labels else label_lower_map.get(raw_lbl.lower())

        # Fuzzy partial match: if "Proximal phalanx" is in labels and response has "proximal phalanx"
        if not matched_lbl:
            raw_lower = raw_lbl.lower()
            for lbl in labels:
                if lbl not in found_labels and (
                    raw_lower in lbl.lower() or lbl.lower() in raw_lower
                ):
                    matched_lbl = lbl
                    break

        if not matched_lbl or matched_lbl in found_labels:
            continue

        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        try:
            xi = int(float(x))
            yi = int(float(y))
        except (TypeError, ValueError):
            continue

        xi = max(0, min(1000, xi))
        yi = max(0, min(1000, yi))
        px, py = _coords_1000_to_pixels(xi, yi, width, height)
        positions.append({
            "label": matched_lbl,
            "point_x": px,
            "point_y": py,
        })
        found_labels.add(matched_lbl)

    # Fill missing labels with centre-of-image fallback
    missing = [lbl for lbl in labels if lbl not in found_labels]
    if missing:
        print(f"[vision_placer] Warning: {len(missing)} labels not found in "
              f"detection response: {missing}")
        for i, lbl in enumerate(missing):
            frac_y = (i + 1) / (len(missing) + 1)
            positions.append({
                "label": lbl,
                "point_x": width // 2,
                "point_y": int(height * frac_y),
            })

    return positions


# ─────────────────────── VLM API calls ─────────────────────────

def _call_vision(
    image_bytes: bytes,
    prompt: str,
    provider: str,
    temperature: float = 0.1,
) -> list[dict]:
    """Route to the appropriate vision model."""
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
    """Call Gemini Vision with configurable temperature."""
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
    """Call GPT-4o Vision with configurable temperature."""
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


# ─────────────────────── JSON parsing ─────────────────────────

def _parse_json_array(text: str) -> list[dict]:
    """Parse a JSON array from VLM response text, with multiple fallbacks."""
    text = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline > 0:
            text = text[first_newline + 1:]
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

    # Attempt 3: Try to fix common JSON issues (trailing commas, etc.)
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        # Remove trailing commas before ] or }
        import re
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass

    print(f"[vision_placer] Failed to parse JSON: {text[:300]}")
    return []
