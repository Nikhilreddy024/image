"""
Configuration loader for medical-figure-gen.
Reads API keys from .env and provides model/path settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Provider Selection ────────────────────────────────────
IMAGE_GEN_PROVIDER = os.getenv("IMAGE_GEN_PROVIDER", "gemini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai (vision + refiner)
# Prompt enhancement only — env-only, not shown in UI. Groq has no vision API.
PROMPT_ENHANCEMENT_PROVIDER = os.getenv("PROMPT_ENHANCEMENT_PROVIDER", "groq")  # groq | gemini | openai

# ── Paths ─────────────────────────────────────────────────
STATIC_DIR = PROJECT_ROOT / "static"
GENERATED_DIR = STATIC_DIR / "generated"

# Ensure output dirs exist
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Settings ────────────────────────────────────────
GEMINI_LLM_MODEL = "gemini-2.0-flash"
# Image generation: nano-banana-pro-preview (fast) | gemini-3-pro-image-preview (best quality)
# Override via GEMINI_IMAGE_MODEL in .env
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "nano-banana-pro-preview")
# gpt-image-1.5 (best) | dall-e-3 — override via OPENAI_IMAGE_MODEL in .env
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
OPENAI_LLM_MODEL = "gpt-4o"
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")  # for prompt enhancement

# ── Label / Annotation Settings ──────────────────────────
DEFAULT_LABEL_STYLE = "professional"  # plain_text | boxed_text | numbered | color_coded | minimal | textbook | professional

# ── Vision Label Placer Settings ─────────────────────────
# Ensemble detection: run N detection calls and take median coordinates
# Higher = more accurate but more API calls.  Set to 1 to disable ensemble.
VISION_ENSEMBLE_ROUNDS = int(os.getenv("VISION_ENSEMBLE_ROUNDS", "3"))
# Iterative verification: max rounds of verify-correct loops (0 to disable)
VISION_VERIFY_ROUNDS = int(os.getenv("VISION_VERIFY_ROUNDS", "2"))

# ── Font Settings ─────────────────────────────────────────
# Pillow font paths (Windows default; override in .env if needed)
FONT_PATH = os.getenv("FONT_PATH", r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD_PATH = os.getenv("FONT_BOLD_PATH", r"C:\Windows\Fonts\arialbd.ttf")
