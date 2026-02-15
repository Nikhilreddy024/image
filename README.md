# Medical Figure Generator

A chat-based tool for generating labeled medical and scientific textbook-style figures. It combines LLM-driven diagram planning, cloud image generation APIs (Gemini Imagen / DALL-E 3) with embedded numbered placeholders, OCR-based placeholder detection, and Pillow/SVG rendering — all orchestrated through a FastAPI web interface with iterative refinement.

---

## Architecture

The system is a **4-stage pipeline** with an optional **refinement loop**, exposed through a chat UI:

```
User Message
    │
    ▼
┌──────────────────────────┐
│ 1. Diagram Planner       │  LLM generates drawing prompt + label list + auditor QA loop
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 2. Image Generator       │  Gemini/DALL-E produces raster PNG with numbered placeholders (1,2,3...)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 3. Placeholder OCR       │  EasyOCR detects placeholder positions; vision API fallback if needed
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 4. Label Renderer        │  Pillow renders annotated PNG + embedded-raster SVG
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ 5. Refiner (chat loop)   │  Hybrid intent classifier routes edits, style changes,
│                          │  label repositioning, or full regeneration
└──────────────────────────┘
```

### Key Design Decisions

- **No local GPU required.** All heavy lifting (image generation, OCR) uses cloud APIs and local EasyOCR.
- **Placeholder-embedded generation.** The image model draws numbered markers (1, 2, 3, ...) directly on the illustration at each structure. OCR detects their positions; placeholders are then covered and replaced with real labels.
- **OCR + vision fallback.** EasyOCR locates placeholder numbers. If detection is insufficient (<50%), the pipeline falls back to a single vision API call for structure positions.
- **Hybrid intent classification.** The refiner uses 23+ deterministic regex patterns before falling back to an LLM, ensuring fast and reliable intent routing for common refinement requests.

### Project Structure

```
medical-figure-gen/
├── .env                          # API keys (not committed)
├── .env.example                  # Template for .env
├── .gitignore
├── config.py                     # Central config: paths, models, settings
├── app.py                        # FastAPI server — orchestrates pipeline + serves UI
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── diagram_planner.py        # Stage 1 — LLM plan generation + auditor loop
│   ├── image_generator.py        # Stage 2 — Cloud image generation (Gemini / OpenAI)
│   ├── placeholder_ocr.py        # Stage 3 — OCR-based placeholder detection (vision fallback)
│   ├── label_renderer.py         # Stage 4 — Pillow PNG + SVG annotation rendering
│   ├── refiner.py                # Stage 5 — Multi-action refinement router
│   └── utils.py                  # Shared utilities (retry decorator, etc.)
├── templates/
│   └── index.html                # Chat UI (split panel: chat + three preview tabs)
└── static/
    └── generated/                # Runtime output (per-session folders)
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 (conda env recommended) |
| **API keys** | At least one of: Google Gemini API key, OpenAI API key |
| **No GPU required** | All computation is done via cloud APIs |

---

## Setup

### 1. Clone & enter the project

```bash
git clone <your-repo-url> medical-figure-gen
cd medical-figure-gen
```

### 2. Create conda environment

```bash
conda create -n svgrender python=3.10 -y
conda activate svgrender
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GOOGLE_GENERATIVE_AI_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key

IMAGE_GEN_PROVIDER=gemini    # or "openai"
LLM_PROVIDER=gemini          # or "openai"
```

---

## Running

```bash
# From the medical-figure-gen directory:
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

> **Windows note:** If `conda activate` doesn't work in VS Code terminal, use the full path:
> ```
> C:\Users\<you>\anaconda3\envs\svgrender\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
> ```

Open **http://127.0.0.1:8000** in your browser.

### Quick Start

1. Type: `"Draw a human hand anatomy diagram"`
2. Wait for the pipeline (15–30 seconds depending on API)
3. The annotated PNG and SVG appear in the preview tabs
4. Refine: `"Move the labels to the left"`, `"Use numbered style"`, `"Add a label for Ulna"`
5. Regenerate: `"Give me a better image"` or `"Regenerate with more detail"`

### UI Controls

| Control | Effect |
|---|---|
| **LLM Provider** dropdown | Gemini or OpenAI for planning & vision tasks |
| **Image Provider** dropdown | Gemini Imagen or DALL-E 3 for image generation |
| **Label Style** dropdown | `boxed_text` (default), `plain_text`, `numbered`, `color_coded`, `minimal`, `textbook` |
| **Skip Annotation** checkbox | Returns only the raw raster image (no labels) |

---

## Configuration Reference

All settings live in [config.py](config.py):

| Setting | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"gemini"` | LLM for planning, vision, & refinement (`"gemini"` or `"openai"`) |
| `IMAGE_GEN_PROVIDER` | `"gemini"` | Image generation API (`"gemini"` or `"openai"`) |
| `GEMINI_LLM_MODEL` | `"gemini-2.0-flash"` | Gemini model for text & vision tasks |
| `GEMINI_IMAGE_MODEL` | `"nano-banana-pro-preview"` | Gemini model for image generation |
| `OPENAI_IMAGE_MODEL` | `"dall-e-3"` | OpenAI image model |
| `OPENAI_LLM_MODEL` | `"gpt-4o"` | OpenAI text & vision model |
| `DEFAULT_LABEL_STYLE` | `"boxed_text"` | Default annotation style |
| `AUDITOR_MAX_ITERATIONS` | `2` | Max planner↔auditor feedback loops (0 = skip) |
| `FONT_PATH` | `arial.ttf` | Font for label rendering (Windows default) |
| `FONT_BOLD_PATH` | `arialbd.ttf` | Bold font for label rendering |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/api/generate` | Full pipeline run |
| `POST` | `/api/refine` | Refine existing figure |
| `GET` | `/api/sessions` | List active sessions |
| `GET` | `/generated/{session_id}/{file}` | Serve generated files (PNG, SVG) |

### Generate request

```json
{
  "message": "Draw a human hand anatomy diagram",
  "label_style": "boxed_text",
  "skip_annotate": false,
  "llm_provider": "gemini",
  "image_provider": "gemini"
}
```

### Generate response

```json
{
  "session_id": "a1b2c3d4",
  "steps": [
    { "stage": "diagram_planner", "result": { "labels": 12, "type": "anatomy" } },
    { "stage": "image_generator", "result": "/generated/a1b2c3d4/raster.png" },
    { "stage": "placeholder_ocr", "result": "Detected 12 label positions via OCR" },
    { "stage": "label_renderer", "result": "boxed_text style, 12 labels" }
  ],
  "raster_url": "/generated/a1b2c3d4/raster.png",
  "annotated_url": "/generated/a1b2c3d4/annotated.png",
  "svg_url": "/generated/a1b2c3d4/annotated.svg",
  "plan": {
    "labels": ["Distal Phalanges", "Middle Phalanges", "..."],
    "description": "Dorsal view of a human right hand showing all 27 skeletal bones.",
    "style": "colored_diagram",
    "diagram_type": "anatomy",
    "label_side": "right"
  }
}
```

### Refine request

```json
{
  "session_id": "a1b2c3d4",
  "message": "Change labels to numbered style",
  "label_style": "numbered",
  "llm_provider": "gemini",
  "image_provider": "gemini"
}
```

---

## How Each Stage Works

### Stage 1 — Diagram Planner (`pipeline/diagram_planner.py`)

Sends the user message to an LLM with a system prompt that produces structured JSON:

```json
{
  "drawing_prompt": "A single scientific medical illustration of a human right hand...",
  "labels": ["Distal Phalanges", "Middle Phalanges", "..."],
  "label_side": "right",
  "style": "colored_diagram",
  "description": "Dorsal view of a human right hand showing all 27 skeletal bones.",
  "diagram_type": "anatomy"
}
```

Key rules enforced by the planner prompt:
- **Drawing prompts are purely positive** — no prohibitions ("do not", "no text", etc.)
- **Max ~60 words** — concise descriptions generate cleaner images
- An **auditor** LLM reviews the plan and can request fixes (up to `AUDITOR_MAX_ITERATIONS` rounds)

### Stage 2 — Image Generator (`pipeline/image_generator.py`)

Calls the configured image API (Gemini Imagen or DALL-E 3) with the drawing prompt.

Before sending, the generator:
1. **Strips negations** via regex — removes any "do not", "no text", etc. the planner may have included
2. **Appends placeholder instruction** — asks for small circled numbers (1, 2, 3, ... N) at each anatomical structure. If no labels in plan, uses "no text, no labels" suffix instead.
3. **Saves** the result as `raster.png` (1024×1024)

### Stage 3 — Placeholder OCR (`pipeline/placeholder_ocr.py`)

Locates label positions by detecting the numbered placeholders embedded in the generated image:

1. **OCR detection** — EasyOCR (with `allowlist='0123456789'`) finds all digits and their bounding boxes in the image
2. **Mapping** — Each detected number N maps to `labels[N-1]`; bbox center becomes the leader-line target point
3. **Cover regions** — Bounding boxes are stored for covering placeholders before drawing real labels
4. **Vision fallback** — If OCR detects fewer than 50% of expected numbers, a single vision API call extracts structure positions instead

The label renderer then covers placeholder regions (with background-colored ellipses) and draws real labels with leader lines.

### Stage 4 — Label Renderer (`pipeline/label_renderer.py`)

Takes the located points and renders annotations in two formats:

- **Annotated PNG** — Pillow draws labels with leader lines directly on the raster image
- **Annotated SVG** — Embeds the raster as a `<image>` element with vector `<text>` and `<line>` overlays

Supports 6 annotation styles: `plain_text`, `boxed_text`, `numbered`, `color_coded`, `minimal`, `textbook`.

Layout features:
- Sorts labels by y-coordinate for clean vertical ordering
- Resolves overlapping labels with automatic spacing
- Configurable label side (right, left, top, bottom, around)

### Stage 5 — Refiner (`pipeline/refiner.py`)

Handles iterative chat-based refinement with a **hybrid intent classifier**:

1. **Deterministic patterns** — 23+ regex rules match common requests instantly:
   - `"change style to numbered"` → `style_change`
   - `"remove the capitate label"` → `remove_label`
   - `"add a label for ulna"` → `add_label`
   - `"move the labels"` → `reposition_labels`
   - `"give me a better image"` → `regenerate`

2. **LLM fallback** — Only used when no regex matches; classifies into one of 6 action types

Supported actions:

| Action | What happens |
|---|---|
| `style_change` | Re-renders labels with new style (no API calls) |
| `label_edit` | Modifies label text, re-renders |
| `remove_label` | Removes a label, re-renders |
| `add_label` | Adds label to plan, re-runs placeholder OCR + renderer |
| `reposition_labels` | Re-runs placeholder OCR on existing image |
| `regenerate` | Re-runs full pipeline (planner → image gen → OCR → render) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `conda activate` doesn't work in VS Code terminal | Use full Python path: `C:\Users\...\anaconda3\envs\svgrender\python.exe` |
| "Gemini did not return an image" | The model may not support your prompt. Try `IMAGE_GEN_PROVIDER=openai` |
| Image has multiple panels / collage | Check that `_NEGATION_RE` in `image_generator.py` is stripping prohibitions. The planner prompt should produce only positive descriptions |
| Labels misplaced or missing | Check that image model drew placeholders; OCR logs show "Detected N/M placeholders". Install EasyOCR: `pip install easyocr` |
| Rate limit errors (429) | Built-in retry with exponential backoff handles this. Reduce request frequency or check API quotas |
| Font not found on Linux/macOS | Set `FONT_PATH` and `FONT_BOLD_PATH` in `.env` to valid TTF paths |

---

## Adding a New Provider

To add a new LLM or image generation provider:

1. Add the API key to `.env` and `config.py`
2. Add a new branch in the relevant pipeline file:
   - **LLM tasks:** `diagram_planner.py`, `placeholder_ocr.py` (vision fallback), `refiner.py`
   - **Image generation:** `image_generator.py`
3. Add the provider name to `IMAGE_GEN_PROVIDER` / `LLM_PROVIDER` options

---

## License

MIT
