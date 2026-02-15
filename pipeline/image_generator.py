"""
Stage 2: Image Generator
Takes structured prompt → generates raster PNG image via API.
Supports Gemini Imagen and OpenAI DALL-E 3.
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


# Regex to strip text/label-related negative phrases the planner may have leaked.
# IMPORTANT: This is intentionally narrow — we only strip phrases that prohibit text,
# labels, numbers, or annotations. We keep style negations like "no gradients" since
# those are valid artistic instructions for the image gen model.
_TEXT_NEGATION_RE = re.compile(
    r"(?i)"
    r"(\b(do not|don'?t|never|must not|should not|cannot|absolutely no|strictly no)\b"
    r"[^.;]*(text|label|number|annotation|caption|leader\s*line|letter|word)[^.;]*[.;]?\s*)",
)


def _build_clean_image_suffix(labels: list[str]) -> str:
    """
    Build a suffix that ensures the image is a clean illustration with NO text/numbers.
    Labels are overlaid programmatically later by the label renderer, so the base image
    must be pristine.  We DO hint at which structures should be visually distinct so the
    vision-based label placer can locate them afterward.
    """
    if not labels:
        return (
            ". Single illustration, pure visual artwork. "
            "Absolutely no text, no letters, no numbers, no labels, no annotations anywhere on the image."
        )
    # Build a short hint listing key structures that should be visually distinguishable
    structure_hint = ", ".join(labels[:20])  # cap for prompt-length sanity
    return (
        f". Ensure the following structures are clearly and distinctly drawn so they can "
        f"be individually identified: {structure_hint}. "
        "Each structure must have clear visual boundaries and distinct coloring or shading. "
        "CRITICAL: Do NOT place any text, letters, numbers, labels, annotations, leader lines, "
        "or captions anywhere on the image. The image must be a completely clean illustration "
        "with zero text of any kind."
    )


def generate_image(spec: dict, session_id: str, provider: str | None = None) -> Path:
    """
    Generate a clean raster image from the structured spec.
    The image will contain NO text, numbers, or annotations — those are added
    programmatically later by the label renderer after the vision-based label
    placer locates each structure.

    Args:
        spec: dict from prompt_structurer with 'drawing_prompt' and optionally 'labels'
        session_id: unique ID for this generation session
        provider: 'gemini' or 'openai' (overrides config default)

    Returns:
        Path to the saved PNG image
    """
    provider = provider or IMAGE_GEN_PROVIDER

    labels = spec.get("labels") or []
    clean_suffix = _build_clean_image_suffix(labels)

    # Clean the drawing prompt: strip any negative/prohibition phrases and append suffix
    raw_prompt = spec["drawing_prompt"]
    cleaned = _TEXT_NEGATION_RE.sub("", raw_prompt).strip()
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\.\.+", ".", cleaned)
    prompt = cleaned + clean_suffix
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
