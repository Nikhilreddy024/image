"""
Stage 5: Refiner — Multi-Agent Intent Router
Handles chat-based refinement of the generated figure.

Architecture:
  User message → Intent Classifier (keyword + LLM fallback)
                     ↓
             ┌───────┴──────────┐
             │   Action Router  │
             └───────┬──────────┘
                     ├─ StyleAgent       (instant re-render)
                     ├─ LabelEditAgent   (text change + re-render)
                     ├─ RemoveLabelAgent (filter + re-render)
                     ├─ RepositionAgent  (re-run vision placer)
                     ├─ AddLabelAgent    (add + re-run vision)
                     └─ RegenerateAgent  (full pipeline re-run)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from google import genai
from google.genai import types
import openai

from config import (
    GOOGLE_API_KEY, OPENAI_API_KEY,
    LLM_PROVIDER, GEMINI_LLM_MODEL, OPENAI_LLM_MODEL,
    GENERATED_DIR,
)
from pipeline.placeholder_replacer import detect_and_replace


# ─── Deterministic Intent Patterns ───────────────────────────────────
# Each pattern → (action, confidence).  Checked in order; first match wins.

_INTENT_PATTERNS: list[tuple[str, str]] = [
    # ── remove_label (check BEFORE reposition since "remove" contains "move") ──
    (r"\b(remove|delete|drop|hide|get\s+rid\s+of)\s+(the\s+)?\w+\s+label", "remove_label"),
    (r"\b(remove|delete|drop|hide)\s+(the\s+)?label", "remove_label"),
    (r"\bdon'?t\s+(label|show|include)\b", "remove_label"),

    # ── reposition_labels ──
    (r"\b(adjust|fix|correct|reposition|update)\b.*\b(label|annotation|position|placement|place)\b", "reposition_labels"),
    (r"\bmove\s+(the\s+)?(label|annotation)", "reposition_labels"),
    (r"(label|annotation)s?\s+(are|were|is)\s+(in\s+)?(wrong|incorrect|bad|misplaced|off|not\s+correct)", "reposition_labels"),
    (r"(place|put|position)\s+(them|labels?|annotations?)\s+(correctly|properly|in\s+(the\s+)?(correct|right|proper)\s+(place|position))", "reposition_labels"),
    (r"labels?\s+(not\s+)?in\s+(correct|right|proper)\s+position", "reposition_labels"),
    (r"(wrong|incorrect|bad)\s+(position|placement|place)", "reposition_labels"),
    (r"(placed?\s+incorrectly|incorrectly\s+placed?)", "reposition_labels"),

    # ── style_change ──
    (r"\b(use|switch\s+to|change\s+to|try)\s+(numbered|textbook|plain.?text|boxed.?text|color.?coded|minimal)\b", "style_change"),
    (r"\b(numbered|textbook|plain.?text|boxed.?text|color.?coded|minimal)\s+(style|labels?|format)\b", "style_change"),
    (r"\bchange\s+(the\s+)?style\b", "style_change"),

    # ── add_label ──
    (r"\b(add|include|also\s+label|mark)\s+(a\s+)?(new\s+)?label\b", "add_label"),
    (r"\balso\s+(label|mark|annotate)\s+(the\s+)?\w+", "add_label"),

    # ── label_edit ──
    (r"\b(rename|change)\s+.+\s+to\s+", "label_edit"),
    (r"\bfix\s+(the\s+)?spelling\b", "label_edit"),

    # ── regenerate (explicit image changes) ──
    (r"\b(regenerate|re-?generate|new\s+image|different\s+image)\b", "regenerate"),
    (r"\b(show|draw|make)\s+(it\s+)?(from|in)\s+(a\s+)?(different|lateral|medial|anterior|posterior)\s+(view|angle|perspective)", "regenerate"),
    (r"\b(single|one)\s+(hand|eye|organ)\b.*\bnot\s+(two|2|multiple)\b", "regenerate"),
    (r"\b(not\s+)?(two|2|multiple)\s+(hand|eye|organ)s?\b", "regenerate"),
    (r"\badd\s+(more\s+)?detail\b", "regenerate"),
    (r"\b(show|include|add)\s+(muscles?|nerves?|vessels?|arteries?|veins?)\b", "regenerate"),
    (r"\b(random\s+lines?|artifacts?|blurry|bad\s+quality|poor\s+quality)\b", "regenerate"),
    (r"\bimage\s+(is|has|looks?)\s+(bad|poor|wrong|blurry)", "regenerate"),
    # Common natural-language regeneration requests
    (r"\b(give|show|draw|create|make)\s+(me\s+)?(a\s+|the\s+)?(detailed|better|new|another|different)\b", "regenerate"),
    (r"\b(i\s+want|i\s+need|please\s+(give|show|draw|create|make))\s+.{0,30}\b(diagram|image|figure|picture|illustration)\b", "regenerate"),
    (r"\b(draw|create|generate|make)\s+.{0,20}\b(anatomy|diagram|figure)\b", "regenerate"),
]


# LLM-based fallback prompt (only used when keywords don't match)
REFINE_CLASSIFY_PROMPT = """Classify this refinement request for a medical diagram into ONE action:

Actions:
- "style_change": change annotation visual style (professional, numbered, textbook, plain, boxed, color_coded, minimal)
- "label_edit": rename/fix specific label text
- "add_label": add a new label
- "remove_label": remove a label
- "reposition_labels": labels are in wrong positions, re-analyze the image
- "regenerate": image itself needs to change (different content, view, quality fix)

Current labels: {labels}
Current style: {style}
Request: "{request}"

Return ONLY JSON: {{"action": "...", "changes": [{{"target": "...", "value": "..."}}], "explanation": "...", "new_style": null}}"""


class RefineSession:
    """Maintains state for iterative refinement of a figure."""

    def __init__(
        self,
        session_id: str,
        plan: dict,
        label_positions: list[dict],
        label_style: str,
    ):
        self.session_id = session_id
        self.plan = plan
        self.label_positions = label_positions
        self.label_style = label_style
        self.history: list[dict] = []
        self.iteration = 0

    def refine(self, request: str, new_style: str | None = None) -> dict:
        """
        Process a refinement request.

        Returns dict with:
            - action: what type of refinement was applied
            - explanation: what was changed
            - needs_regeneration: bool
            - new_plan: if regeneration needed
            - annotated_png: Path if labels were re-rendered
            - annotated_svg: Path if labels were re-rendered
        """
        self.iteration += 1
        self.history.append({"role": "user", "content": request})

        # Classify the request
        classification = _classify_request(
            request,
            self.plan.get("labels", []),
            self.label_style,
        )

        action = classification.get("action", "regenerate")
        explanation = classification.get("explanation", "Processing your request.")

        # Handle style override from UI — only if the user actually changed the style
        if new_style and new_style != self.label_style:
            action = "style_change"
            classification["new_style"] = new_style
            print(f"[refiner] Style changed via dropdown: {self.label_style} -> {new_style}")

        result: dict
        if action == "regenerate":
            new_plan = _build_refined_plan(self.plan, request)
            result = {
                "action": "regenerate",
                "explanation": explanation,
                "needs_regeneration": True,
                "new_plan": new_plan,
            }

        elif action == "style_change":
            style = classification.get("new_style") or self.label_style
            self.label_style = style
            png_path, svg_path = self._re_render(style)
            result = {
                "action": "style_change",
                "explanation": f"Changed annotation style to '{style}'.",
                "needs_regeneration": False,
                "annotated_png": png_path,
                "annotated_svg": svg_path,
            }

        elif action == "label_edit":
            changes = classification.get("changes", [])
            for change in changes:
                old_text = change.get("target", "")
                new_text = change.get("value", "")
                if old_text and new_text:
                    # Update in plan
                    self.plan["labels"] = [
                        new_text if lbl == old_text else lbl
                        for lbl in self.plan.get("labels", [])
                    ]
                    # Update in label_positions
                    for pos in self.label_positions:
                        if pos["label"] == old_text:
                            pos["label"] = new_text

            png_path, svg_path = self._re_render(self.label_style)
            result = {
                "action": "label_edit",
                "explanation": explanation,
                "needs_regeneration": False,
                "annotated_png": png_path,
                "annotated_svg": svg_path,
            }

        elif action == "remove_label":
            # Since annotation arrows are baked into the image, removing a label
            # requires regeneration to update the numbered markers and leader lines.
            changes = classification.get("changes", [])
            for change in changes:
                target = change.get("target", "")
                if target:
                    self.plan["labels"] = [
                        lbl for lbl in self.plan.get("labels", [])
                        if lbl.lower() != target.lower()
                    ]
                    self.label_positions = [
                        pos for pos in self.label_positions
                        if pos["label"].lower() != target.lower()
                    ]

            result = {
                "action": "remove_label",
                "explanation": explanation + " Regenerating image without the removed label marker.",
                "needs_regeneration": True,
                "new_plan": _build_refined_plan(self.plan, request),
            }

        elif action == "add_label":
            # Adding a label requires full regeneration since the AI model
            # needs to place a new numbered marker with a leader line.
            changes = classification.get("changes", [])
            for change in changes:
                new_label = change.get("value", "")
                if new_label and new_label not in self.plan.get("labels", []):
                    self.plan["labels"].append(new_label)

            result = {
                "action": "add_label",
                "explanation": explanation + " Regenerating image with new annotation marker.",
                "needs_regeneration": True,
                "new_plan": _build_refined_plan(self.plan, request),
            }

        elif action == "reposition_labels":
            # Arrows and leader lines are baked into the image, so repositioning
            # requires full regeneration with the AI model re-deciding layout.
            result = {
                "action": "reposition_labels",
                "explanation": "Regenerating image — the AI model will re-decide label positions and arrow layout.",
                "needs_regeneration": True,
                "new_plan": _build_refined_plan(self.plan, request),
            }

        else:
            result = {
                "action": "regenerate",
                "explanation": explanation,
                "needs_regeneration": True,
                "new_plan": _build_refined_plan(self.plan, request),
            }

        self.history.append({"role": "assistant", "content": result["explanation"]})
        return result

    def _re_render(self, style: str) -> tuple[Path, Path]:
        """Re-render labels by re-running placeholder replacement on the raster.

        Since the new pipeline bakes arrows/leader lines into the raster image
        during generation, label-only edits (rename, remove, style change) are
        handled by re-detecting markers and replacing with updated label text.
        """
        raster_path = GENERATED_DIR / self.session_id / "raster.png"

        # Build labels list from current label_positions
        labels = [p["label"] for p in self.label_positions]

        if not labels:
            # Nothing to render
            return raster_path, raster_path

        png_path, svg_path, new_positions = detect_and_replace(
            image_path=raster_path,
            labels=labels,
            session_id=self.session_id,
        )
        self.label_positions = new_positions
        return png_path, svg_path


def _classify_request(request: str, labels: list[str], style: str) -> dict:
    """
    Hybrid intent classifier: deterministic keyword matching first,
    LLM fallback only when keywords don't match.
    """
    # ── Phase 1: Deterministic keyword matching ───────────────
    action = _classify_by_keywords(request)
    if action:
        print(f"[refiner] Intent classified by keywords: {action}")
        return {
            "action": action,
            "changes": [],
            "explanation": f"Detected intent: {action}",
            "new_style": _extract_style_name(request) if action == "style_change" else None,
        }

    # ── Phase 2: LLM fallback for ambiguous requests ─────────
    print(f"[refiner] No keyword match, falling back to LLM classification...")
    prompt = REFINE_CLASSIFY_PROMPT.format(
        request=request,
        labels=json.dumps(labels),
        style=style,
    )

    # Groq is only for prompt enhancement; refiner uses gemini or openai
    provider = "gemini" if LLM_PROVIDER == "groq" else LLM_PROVIDER
    if provider == "gemini":
        text = _llm_gemini(prompt)
    else:
        text = _llm_openai(prompt)

    result = _parse_json(text)
    print(f"[refiner] LLM classified as: {result.get('action', 'unknown')}")
    return result


def _classify_by_keywords(request: str) -> str | None:
    """
    Match user request against known intent patterns.
    Returns the action string if matched, None otherwise.
    """
    text = request.lower().strip()
    for pattern, action in _INTENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return action
    return None


def _extract_style_name(request: str) -> str | None:
    """Extract a style name from the request text."""
    text = request.lower()
    style_map = {
        "numbered": "numbered",
        "textbook": "textbook",
        "plain": "plain_text",
        "plain_text": "plain_text",
        "plain text": "plain_text",
        "boxed": "boxed_text",
        "boxed_text": "boxed_text",
        "boxed text": "boxed_text",
        "color": "color_coded",
        "color_coded": "color_coded",
        "color coded": "color_coded",
        "minimal": "minimal",
        "professional": "professional",
        "pro": "professional",
        "clean": "professional",
    }
    for keyword, style_name in style_map.items():
        if keyword in text:
            return style_name
    return None


def _build_refined_plan(plan: dict, request: str) -> dict:
    """Build a new plan incorporating the user's refinement.
    
    Stores the refinement request so the app layer can re-run the planner
    with the combined context (original + refinement).
    """
    new_plan = dict(plan)
    new_plan["_refinement_request"] = request
    return new_plan


def _llm_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),  # type: ignore[call-arg]
    )
    return response.text or ""


def _llm_openai(prompt: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action": "regenerate", "changes": [], "explanation": text[:200]}
