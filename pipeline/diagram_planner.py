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


ENHANCE_PROMPT_SYSTEM = """You are a creative scientific illustration specialist for medical/biomedical textbook figures. Your role is to transform brief user descriptions into rich, vivid prompts that inspire high-quality image generation.

Given a user's short description, produce a single enhanced, creative prompt suitable for an image generation model. Your output will be used ONLY to generate the base illustration — no labels or text will be requested at this stage.

Requirements for the enhanced prompt:
- Be vivid and evocative: describe style (flat vector, textbook, white background), anatomical subject, view (e.g. palmar, lateral, cross-section), lighting feel, and key structures with visual clarity.
- Use positive descriptions only (what to draw), no negative phrases (e.g. avoid "no text", "no labels").
- Add creative depth: consider composition, visual hierarchy, spatial relationships, and color semantics (e.g. nerves as yellow, vessels as red, bone as cream).
- ALWAYS include a legend placeholder: describe a dedicated, clearly delineated area (e.g. bottom-right corner, bottom strip, side panel) reserved for a legend/key. Use phrases like "reserved legend area", "empty legend panel", or "dedicated legend space" — the legend content will be added programmatically later; you only describe the visual space for it.
- Vary your language: avoid generic phrasing; choose precise anatomical and stylistic terms that bring the diagram to life.
- Keep it concise but comprehensive (roughly 100–250 words).
- Do NOT output a list of labels or any JSON — output only the enhanced prompt text, nothing else."""


PLANNER_SYSTEM_PROMPT = """You are a scientific illustration planner for medical/biomedical textbook figures.

Given a user's description, produce a JSON diagram plan with these fields:

1. "drawing_prompt": A comprehensive, structured prompt for medical illustration generation.
   CRITICAL: The drawing_prompt describes ONLY the illustration itself. Do NOT ask for labels, text, annotations, or leader lines — these are added programmatically later. The base image must be a clean illustration with no text.
   STRUCTURE:
   - Begin with "Style Preface:" describing the visual foundation:
     * Strictly flat, clean vector medical illustration in professional textbook style
     * Solid white background, no gradients, no soft shading, no pseudo-3D effects
     * Uniform medium-weight black outlines throughout
     * Muted, desaturated color palette: flat beige for skin, muted reddish-brown for muscle, off-white/cream for bone
   
   - Follow with "Main Content:" describing the anatomical subject with vivid, precise visual language:
     * Multiple side-by-side diagrams if showing different layers/systems (e.g., Skin, Muscle, Bone)
     * Each diagram showing specific anatomical structures with distinct outlines and clear spatial relationships
     * Creative color semantics: yellow for nerves, red for arteries/vessels, cream for bone, muted tones for soft tissue
     * Describe structures as visual elements (e.g. phalanges, metacarpals, muscle groups) — NOT as labels
   
   - Include "Right Column Insets:" or "Legend & Insets:" if applicable:
     * Cross-sectional views, surface landmarks
     * MANDATORY: A clearly delineated legend/key area — describe its placement (e.g. bottom-right panel, bottom strip, side box) with visual boundaries (border, background panel) so the illustration has a dedicated space for the legend. Do not include legend text or content — only the empty reserved area.
   
   - End with "Crucial Constraints:" balanced composition, ample white space, and creative visual hierarchy
   
   Maximum 300 words. Write ONLY positive visual descriptions. ALWAYS include a dedicated legend area. NEVER include labels, text, annotations, or leader lines in the drawing_prompt.

2. "labels": An array of anatomical/scientific terms that will be rendered as annotations.
   - Order logically by anatomical position or system hierarchy
   - 5-30 labels depending on complexity
   - Group related structures together

3. "label_side": "right" (default), "left", "integrated" (for multi-panel with labels within each), or "legend_based".

4. "style": One of: "multi_panel_system", "layered_anatomy", "cross_section", "comparative_view", "single_structure".

5. "description": 1-2 sentence summary of the complete figure including all panels and insets.

6. "diagram_type": "anatomy" or "relational".

Respond ONLY with valid JSON. No markdown, no code fences.

Example input: "draw a human hand anatomy"
Example output:
{
  "drawing_prompt": "Style Preface: A strictly flat, clean vector medical illustration in a professional textbook style, set against a solid white background. No gradients, no soft shading, no pseudo-3D effects. All elements must have uniform medium-weight black outlines. The color palette must be muted and desaturated suitable for scientific publication: flat beige for skin, muted reddish-brown for muscle, off-white/cream for bone. Main Content: Three separate diagrams of the human hand (palmar view) arranged side-by-side. Left Diagram shows surface anatomy with flat color fill and distinct outlines including phalanges, MCP/PIP/DIP joints, palmar creases, thenar eminence, and hypothenar eminence as visual elements. Center Diagram illustrates muscle layers using flat muted reddish-brown colors: flexor digitorum superficialis and profundus, lumbricals, palmar aponeurosis, thenar and hypothenar muscle groups. Nerves shown as distinct flat yellow lines; arteries as flat red lines. Right Diagram shows skeletal structure in flat off-white: phalanges, metacarpals I-V, carpals, radius and ulna. Right Column Insets: Top right, carpal tunnel cross-section showing tendons, nerve, and ligament. Middle right, surface landmarks as a small flat outline hand. Legend Area: Bottom right, a clearly bordered rectangular panel or box reserved exclusively for a legend/key — empty, with a subtle outline or background to distinguish it from the main illustration. Crucial Constraints: Balanced composition with ample white space and clear visual hierarchy. Illustration only — no text, labels, or annotations.",
  "labels": ["Distal phalanx", "Middle phalanx", "Proximal phalanx", "MCP joint", "PIP joint", "DIP joint", "Thenar eminence", "Hypothenar eminence", "Flexor digitorum superficialis", "Flexor digitorum profundus", "Lumbricals", "Median nerve", "Ulnar nerve", "Metacarpals", "Carpals", "Radial artery", "Ulnar artery"],
  "label_side": "integrated",
  "style": "multi_panel_system",
  "description": "Comprehensive palmar view of human hand anatomy showing three side-by-side diagrams (Skin, Muscle, Bone) with right column insets for carpal tunnel cross-section, surface landmarks, and color legend.",
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
