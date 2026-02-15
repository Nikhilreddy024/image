"""
Placeholder OCR — detect numbered markers (1, 2, 3, ...) in generated images.

Uses OCR to locate placeholder positions, then maps them to real labels.
Falls back to vision-based placement if OCR fails or detects too few numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
)


def _bbox_center(bbox: list[list[float]]) -> tuple[int, int]:
    """Compute center of a bounding box [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(sum(xs) / 4), int(sum(ys) / 4)


def detect_placeholders_via_ocr(
    image_path: Path,
    labels: list[str],
) -> tuple[list[dict], int]:
    """
    Use OCR to detect numbered placeholders (1, 2, 3, ...) in the image.
    Maps each detected number to the corresponding label by index.

    Returns:
        (positions, num_detected) — positions list and count of successfully detected numbers
    """
    if not labels:
        return [], 0

    try:
        import easyocr
    except ImportError:
        print("[placeholder_ocr] EasyOCR not installed. Run: pip install easyocr")
        return [], 0

    with Image.open(image_path) as img:
        width, height = img.size

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Detect digits only
    results = reader.readtext(
        str(image_path),
        allowlist="0123456789",
        paragraph=False,
    )

    # Parse and collect: number -> (cx, cy, conf, bbox)
    num_to_pos: dict[int, tuple[int, int, float, list]] = {}
    for bbox, text, conf in results:
        text = str(text).strip()
        if not text or not text.isdigit():
            continue
        num = int(text)
        if num < 1 or num > len(labels):
            continue
        cx, cy = _bbox_center(bbox)
        # Keep best confidence if duplicate numbers
        if num not in num_to_pos or conf > num_to_pos[num][2]:
            num_to_pos[num] = (cx, cy, conf, bbox)

    # Build positions in label order (1 -> labels[0], etc.)
    positions: list[dict] = []
    num_detected = 0
    for i, label in enumerate(labels):
        num = i + 1
        if num in num_to_pos:
            cx, cy, _, bbox = num_to_pos[num]
            cx = max(10, min(cx, width - 10))
            cy = max(10, min(cy, height - 10))
            pos = {"label": label, "point_x": cx, "point_y": cy}
            # Store bbox for covering placeholder (expand slightly for circled numbers)
            if bbox:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                pad = 12
                pos["cover_bbox"] = (
                    max(0, int(min(xs)) - pad),
                    max(0, int(min(ys)) - pad),
                    min(width, int(max(xs)) + pad),
                    min(height, int(max(ys)) + pad),
                )
            positions.append(pos)
            num_detected += 1
        else:
            # Fallback: spread vertically; add heuristic cover_bbox in case number is present
            y = int(height * (0.1 + 0.8 * i / max(len(labels), 1)))
            r = 15
            positions.append({
                "label": label,
                "point_x": width // 2,
                "point_y": y,
                "cover_bbox": (
                    max(0, width // 2 - r), max(0, y - r),
                    min(width, width // 2 + r), min(height, y + r),
                ),
            })
            print(f"[placeholder_ocr] Number {num} not detected, using fallback position")

    print(f"[placeholder_ocr] Detected {num_detected}/{len(labels)} placeholders via OCR")
    return positions, num_detected


def _call_vision_fallback(image_bytes: bytes, prompt: str, provider: str) -> list[dict]:
    """Call vision API for fallback label extraction."""
    import base64
    import json

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = response.text or ""
    elif provider == "openai":
        import openai

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode()
        response = client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=0.1,
            max_tokens=4000,
        )
        text = response.choices[0].message.content or ""
    else:
        return []

    # Parse JSON array (with robust extraction)
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        arr = json.loads(text)
        return arr if isinstance(arr, list) else []
    except json.JSONDecodeError:
        # Try to extract array from surrounding text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                arr = json.loads(text[start:end + 1])
                return arr if isinstance(arr, list) else []
            except json.JSONDecodeError:
                pass
        return []


def detect_placeholders_with_fallback(
    image_path: Path,
    labels: list[str],
    provider: str | None = None,
) -> list[dict]:
    """
    Try OCR first; if it detects too few numbers, fall back to vision API.
    """
    provider = provider or LLM_PROVIDER

    positions, num_detected = detect_placeholders_via_ocr(image_path, labels)

    # If OCR found fewer than half, try vision fallback
    min_acceptable = max(1, len(labels) // 2)
    if num_detected < min_acceptable:
        print("[placeholder_ocr] OCR insufficient, falling back to vision API")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        with Image.open(image_path) as img:
            w, h = img.size

        labels_list = "\n".join(f"  {i+1}. {lbl}" for i, lbl in enumerate(labels))
        prompt = (
            f"You are analyzing a medical/scientific illustration ({w}x{h} pixels).\n\n"
            f"Locate EACH of these specific structures in the image and provide the EXACT point "
            f"(as normalized x,y coordinates from 0.0 to 1.0) where a label leader line should "
            f"point — the center of each structure.\n\n"
            f"Structures to locate:\n{labels_list}\n\n"
            f"Return a JSON array with EXACTLY {len(labels)} entries (one per structure, same order):\n"
            f"[{{\"label\": \"exact label text\", \"x\": 0.xx, \"y\": 0.yy}}, ...]\n\n"
            f"x=0 is left edge, x=1 is right edge, y=0 is top, y=1 is bottom.\n"
            f"Be precise — point to the actual anatomical structure, not to empty space.\n"
            f"Respond with ONLY the JSON array."
        )
        raw = _call_vision_fallback(img_bytes, prompt, provider)
        positions = []
        for i, item in enumerate(raw):
            if i >= len(labels):
                break
            lbl = item.get("label") or labels[i]
            try:
                fx = max(0, min(1, float(item.get("x", 0.5))))
                fy = max(0, min(1, float(item.get("y", 0.5))))
            except (TypeError, ValueError):
                fx, fy = 0.5, 0.5
            px = int(fx * (w - 1))
            py = int(fy * (h - 1))
            r = 15
            positions.append({
                "label": lbl,
                "point_x": px,
                "point_y": py,
                "cover_bbox": (
                    max(0, px - r), max(0, py - r),
                    min(w, px + r), min(h, py + r),
                ),
            })
        # Use plan labels if vision returned different labels
        for i, plan_label in enumerate(labels):
            if i < len(positions):
                positions[i]["label"] = plan_label
            else:
                y = int(h * (0.1 + 0.8 * i / max(len(labels), 1)))
                positions.append({
                    "label": plan_label,
                    "point_x": w // 2,
                    "point_y": y,
                    "cover_bbox": (max(0, w//2 - 15), max(0, y - 15), min(w, w//2 + 15), min(h, y + 15)),
                })

    return positions
