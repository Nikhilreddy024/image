"""
Stage 2: Image Generator
Takes structured prompt → generates raster PNG image via API.
Supports Gemini Imagen and OpenAI DALL-E 3.

The image is generated WITH numbered placeholder markers and leader lines/arrows
baked in. The actual label text is replaced in a later post-processing step.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

from PIL import Image

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    IMAGE_GEN_PROVIDER, GEMINI_IMAGE_MODEL, OPENAI_IMAGE_MODEL,
    GENERATED_DIR,
)
from pipeline.utils import retry_on_rate_limit


def _build_placeholder_suffix(labels: list[str]) -> str:
    """
    Build a suffix that reinforces the numbered placeholder annotation instructions.
    The image model renders numbered markers (1, 2, 3…) with leader lines pointing
    to each anatomical structure.  The actual label text is swapped in post-processing.
    """
    if not labels:
        return (
            ". Single illustration in professional medical textbook style. "
            "Include only the anatomical artwork with clean visual hierarchy."
        )

    n = len(labels)
    marker_list = ", ".join(str(i + 1) for i in range(n))
    # Build the marker-to-structure mapping for the image model
    mapping_lines = "\n".join(
        f"  {i + 1} → {lbl}" for i, lbl in enumerate(labels)
    )

    return (
        f"\n\nANNOTATION INSTRUCTIONS (critical):\n"
        f"Include exactly {n} numbered annotation markers ({marker_list}) in the illustration.\n"
        f"Each marker is a clearly visible, bold number placed in the margin areas of the image.\n"
        f"Connect each marker to its corresponding anatomical structure with a clean, thin "
        f"leader line or arrow.\n\n"
        f"Marker-to-structure mapping:\n{mapping_lines}\n\n"
        f"IMPORTANT RULES:\n"
        f"- Render ONLY the number at each marker position. Do NOT write out the label text.\n"
        f"- Place markers in margin columns (left/right sides) with ample space around each.\n"
        f"- Leader lines should be thin, dark, and professional — no crossing.\n"
        f"- Leave generous empty space around each marker number for text to be added later.\n"
        f"- The final result should look like a professional medical textbook figure with "
        f"numbered annotations."
    )


def generate_image(spec: dict, session_id: str, provider: str | None = None) -> Path:
    """
    Generate a raster image WITH numbered placeholder markers and leader lines.
    The model renders both the illustration and the annotation markers/arrows in
    a single pass.  Placeholder numbers are replaced with real label text later
    by the placeholder replacer.

    Args:
        spec: dict from diagram planner with 'drawing_prompt' and 'labels'
        session_id: unique ID for this generation session
        provider: 'gemini' or 'openai' (overrides config default)

    Returns:
        Path to the saved PNG image (with numbered placeholders baked in)
    """
    provider = provider or IMAGE_GEN_PROVIDER

    labels = spec.get("labels") or []
    placeholder_suffix = _build_placeholder_suffix(labels)

    # Use the drawing prompt as-is (it already contains annotation instructions
    # from the planner) and reinforce with the placeholder suffix.
    raw_prompt = spec["drawing_prompt"]
    # Normalise whitespace only — do NOT strip annotation/text instructions
    cleaned = re.sub(r"  +", " ", raw_prompt).strip()
    cleaned = re.sub(r"\.\.+", ".", cleaned)
    prompt = cleaned + placeholder_suffix
    print(f"[image_gen] Prompt ({len(prompt)} chars): {prompt[:120]}...")

    if provider == "gemini":
        img = _generate_with_gemini(prompt)
    elif provider == "openai":
        img = _generate_with_openai(prompt)
    else:
        raise ValueError(f"Unknown image provider: {provider}")

    # Save to disk
    out_dir = GENERATED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raster.png"
    img.save(str(out_path), "PNG")

    return out_path


@retry_on_rate_limit(max_retries=3, initial_wait=15)
def _generate_with_gemini(prompt: str) -> Image.Image:
    """Generate image using Gemini's image generation capability."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["image", "text"],  # type: ignore[call-arg]
        ),
    )

    # Extract image from response parts
    candidates = response.candidates
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        raise RuntimeError("Gemini returned empty response")
    for part in candidates[0].content.parts:
        if part.inline_data is not None and part.inline_data.data is not None:
            img_bytes: bytes = part.inline_data.data
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise RuntimeError("Gemini did not return an image. Response: " + str(response.text))


def _generate_with_openai(prompt: str) -> Image.Image:
    """Generate image using OpenAI (gpt-image-1.5 or DALL-E 3)."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # gpt-image-1.5 uses "low"|"medium"|"high"; DALL-E 3 uses "standard"|"hd"
    quality = "high" if "gpt-image" in OPENAI_IMAGE_MODEL else "hd"

    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        response_format="b64_json",
        n=1,
    )

    if not response.data:
        raise RuntimeError("OpenAI returned empty response")
    img_b64 = response.data[0].b64_json or ""
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
