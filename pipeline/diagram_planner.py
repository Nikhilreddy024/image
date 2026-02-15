"""
Stage 1: Diagram Planner
Takes raw user message → returns a structured diagram plan.
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types
import openai

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL, GROQ_LLM_MODEL,
    PROMPT_ENHANCEMENT_PROVIDER,
)
from pipeline.utils import retry_on_rate_limit


ENHANCE_PROMPT_SYSTEM = """You are a creative scientific illustration specialist for medical/biomedical textbook figures. Your role is to transform brief user descriptions into rich, vivid prompts that inspire high-quality annotated image generation.

Given a user's short description, produce a single enhanced, creative prompt suitable for an image generation model. The image model will generate a FULLY ANNOTATED illustration — both the artwork and the annotation markers/arrows in a single pass.

Requirements for the enhanced prompt:
- Be vivid and evocative: describe style (flat vector, textbook, white background), anatomical subject, view (e.g. palmar, lateral, cross-section), and key structures with visual clarity.
- INCLUDE annotation layout instructions: describe numbered placeholder markers (1, 2, 3, …) connected to key structures via clean leader lines or arrows, placed in clear margin columns or annotation areas.
- Describe the annotation aesthetic: thin black or dark-gray leader lines, small circled or bold numbers, professional textbook annotation style matching the reference of high-quality medical atlases.
- Add creative depth: composition, visual hierarchy, spatial relationships, color semantics (nerves as yellow, vessels as red, bone as cream).
- ALWAYS include a legend area: a clearly delineated region (e.g. bottom-right panel, side box) reserved for a color-coded legend/key.
- IMPORTANT: Markers should display ONLY the number (1, 2, 3…) — NOT the descriptive label text. The actual label text will be added in post-processing.
- Vary your language; choose precise anatomical and stylistic terms.
- Keep it concise but comprehensive (roughly 150–300 words).
- Do NOT output a list of labels or any JSON — output only the enhanced prompt text, nothing else."""


PLANNER_SYSTEM_PROMPT = """You are a scientific illustration planner for medical/biomedical textbook figures.

Given a user's description, produce a JSON diagram plan. The image generation model will render BOTH the illustration AND numbered annotation markers with leader lines in a single pass — matching the style of professional medical atlases.

Fields:

1. "drawing_prompt": A comprehensive prompt for generating a fully annotated medical illustration.
   The model must render the illustration AND numbered placeholder annotations together.
   STRUCTURE:
   - Begin with "Style Preface:" describing the visual foundation:
     * Professional medical textbook illustration style
     * Solid white background, clean lines, publication-quality rendering
     * Muted, desaturated color palette: flat beige for skin, muted reddish-brown for muscle, off-white/cream for bone
     * Uniform medium-weight black outlines on anatomical structures

   - Follow with "Main Content:" describing the anatomical subject:
     * Multiple panels if showing different layers/systems (e.g., Skin, Muscle, Bone)
     * Precise anatomical structures with distinct visual features
     * Color semantics: yellow for nerves, red for arteries/vessels, cream for bone

   - CRITICAL — "Annotations:" section describing numbered placeholder markers:
     * Include numbered markers (1, 2, 3, …) with clean, thin leader lines or arrows pointing to each anatomical structure
     * Markers should be bold numbers (optionally circled or in small boxes) placed in clear margin areas or annotation columns
     * Leader lines should be thin, dark, professional — straight or elbow-routed, never crossing each other
     * Markers should be well-spaced and arranged in columns along the left or right margins, or distributed around the illustration
     * Provide the FULL marker-to-structure mapping in the prompt so the model knows which number points to which structure
     * IMPORTANT: Render ONLY the number at each marker position — NOT the descriptive label text. The real label text is added in post-processing.
     * Leave generous space around each marker for text replacement

   - Include "Insets & Legend:" if applicable:
     * Cross-sections, surface landmarks, detail views
     * A clearly bordered legend/key area with color-coded entries

   - End with "Layout Constraints:" balanced composition, ample white space, clear visual hierarchy, no overlapping annotations

   Maximum 400 words.

2. "labels": Array of anatomical/scientific terms. labels[0] maps to marker 1, labels[1] to marker 2, etc.
   - Order logically by anatomical position or system hierarchy
   - 5-30 labels depending on complexity

3. "label_side": "right", "left", "integrated" (markers distributed around the illustration), or "both_sides".

4. "style": One of: "multi_panel_system", "layered_anatomy", "cross_section", "comparative_view", "single_structure".

5. "description": 1-2 sentence summary of the complete figure.

6. "diagram_type": "anatomy" or "relational".

Respond ONLY with valid JSON. No markdown, no code fences.

Example input: "draw a human hand anatomy"
Example output:
{
  "drawing_prompt": "Style Preface: A professional medical textbook illustration in clean vector style, set against a solid white background. Muted, desaturated color palette: flat beige for skin, muted reddish-brown for muscle, off-white/cream for bone. Uniform medium-weight black outlines on all anatomical structures. Main Content: Three separate diagrams of the human hand (palmar view) arranged side-by-side. Left Diagram: surface anatomy with flat color fill showing phalanges, joints, palmar creases, thenar and hypothenar eminences. Center Diagram: muscle layer in muted reddish-brown showing flexor digitorum superficialis and profundus, lumbricals, palmar aponeurosis, thenar and hypothenar muscle groups. Nerves as flat yellow lines, arteries as flat red lines. Right Diagram: skeletal view in off-white showing phalanges, metacarpals I-V, carpals, radius and ulna. Insets & Legend: Top-right inset showing carpal tunnel cross-section. Middle-right inset showing surface landmarks. Bottom-right bordered legend panel with color-coded entries for tissue types. Annotations: Include numbered markers 1 through 17 with thin dark-gray leader lines pointing from the margin columns to each corresponding structure. Markers are bold numbers in small white circles placed along the left and right margins of each panel, well-spaced vertically. Marker mapping: 1→distal phalanx, 2→middle phalanx, 3→proximal phalanx, 4→MCP joint, 5→PIP joint, 6→DIP joint, 7→thenar eminence, 8→hypothenar eminence, 9→flexor digitorum superficialis, 10→flexor digitorum profundus, 11→lumbricals, 12→median nerve, 13→ulnar nerve, 14→metacarpals, 15→carpals, 16→radial artery, 17→ulnar artery. Render ONLY the number at each marker — not the label text. Layout Constraints: Balanced composition, ample white space, annotations must not overlap with anatomy or each other.",
  "labels": ["Distal phalanx", "Middle phalanx", "Proximal phalanx", "MCP joint", "PIP joint", "DIP joint", "Thenar eminence", "Hypothenar eminence", "Flexor digitorum superficialis", "Flexor digitorum profundus", "Lumbricals", "Median nerve", "Ulnar nerve", "Metacarpals", "Carpals", "Radial artery", "Ulnar artery"],
  "label_side": "integrated",
  "style": "multi_panel_system",
  "description": "Comprehensive palmar view of human hand anatomy showing three side-by-side diagrams (Skin, Muscle, Bone) with right column insets for carpal tunnel cross-section, surface landmarks, and color legend, annotated with numbered markers.",
  "diagram_type": "anatomy"
}"""


def enhance_prompt(user_message: str, provider: str | None = None) -> dict:
    """
    Get an enhanced, clarified drawing prompt from the LLM only.
    Uses PROMPT_ENHANCEMENT_PROVIDER from config (groq | gemini | openai).
    Groq is env-only — not exposed in UI; it has no vision API.

    Returns dict with key: drawing_prompt (str). Optional: description (str).
    """
    provider = provider or PROMPT_ENHANCEMENT_PROVIDER
    raw = _generate_enhanced_prompt(user_message, provider)
    drawing_prompt = raw.strip()
    if not drawing_prompt:
        drawing_prompt = user_message
    print(f"[planner] Enhanced prompt length: {len(drawing_prompt)} chars")
    return {
        "drawing_prompt": drawing_prompt,
        "description": drawing_prompt[:200] + ("..." if len(drawing_prompt) > 200 else ""),
    }


def _generate_enhanced_prompt(user_message: str, provider: str) -> str:
    """Call LLM to get enhanced prompt text only."""
    if provider == "gemini":
        return _enhance_with_gemini(user_message)
    elif provider == "openai":
        return _enhance_with_openai(user_message)
    elif provider == "groq":
        return _enhance_with_groq(user_message)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _enhance_with_gemini(user_message: str) -> str:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=ENHANCE_PROMPT_SYSTEM,
            temperature=0.3,
        ),
    )
    return (response.text or "").strip()


def _enhance_with_openai(user_message: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": ENHANCE_PROMPT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def _enhance_with_groq(user_message: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": ENHANCE_PROMPT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def create_plan(user_message: str, provider: str | None = None) -> dict:
    """
    Create a diagram plan from user message.

    Returns dict with keys: drawing_prompt, labels, label_side, style, description, diagram_type
    """
    provider = provider or LLM_PROVIDER
    plan = _generate_plan(user_message, provider)
    print(f"[planner] Plan: {len(plan.get('labels', []))} labels, "
          f"type={plan.get('diagram_type')}, side={plan.get('label_side')}")
    return plan


def _generate_plan(user_message: str, provider: str) -> dict:
    """Generate an initial diagram plan using the LLM."""
    if provider == "gemini":
        return _plan_with_gemini(user_message)
    elif provider == "openai":
        return _plan_with_openai(user_message)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


@retry_on_rate_limit(max_retries=3, initial_wait=10)
def _plan_with_gemini(user_message: str) -> dict:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return _parse_plan(response.text or "")


def _plan_with_openai(user_message: str) -> dict:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return _parse_plan(response.choices[0].message.content or "")


def _parse_plan(text: str) -> dict:
    """Parse LLM response into a plan dict."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {
            "drawing_prompt": text,
            "labels": [],
            "label_side": "right",
            "style": "colored_diagram",
            "description": text[:100],
            "diagram_type": "anatomy",
        }

    # Validate/default required keys
    defaults = {
        "drawing_prompt": "",
        "labels": [],
        "label_side": "right",
        "style": "colored_diagram",
        "description": "",
        "diagram_type": "anatomy",
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    return result
