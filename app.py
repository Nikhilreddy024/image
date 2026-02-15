"""
Medical Figure Generator - FastAPI Chat Server
Orchestrates the generation flow:
  1. Diagram Planner (LLM) — user prompt → structured plan (drawing_prompt, labels)
  2. Image Generator (API) — enhanced prompt → clean base image (NO text/numbers)
  3. Vision Label Placer — three-pass zone classification locates each structure
  4. Label Renderer (Pillow/SVG) — overlay labels with leader lines
"""

import uuid
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import GENERATED_DIR, PROJECT_ROOT, DEFAULT_LABEL_STYLE
from pipeline.diagram_planner import enhance_prompt, create_plan
from pipeline.image_generator import generate_image
from pipeline.vision_label_placer import locate_labels
from pipeline.label_renderer import (
    compute_label_layout, render_labels_on_png, render_labels_as_svg,
)
from pipeline.refiner import RefineSession

app = FastAPI(title="Medical Figure Generator")

# Static files & templates
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# In-memory session store  {session_id: {...}}
sessions: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate")
async def generate(request: Request):
    """
    Full pipeline: plan → image → vision place → render labels.
    Body: { "message": "...", "session_id": "..." (optional),
            "label_style": "boxed_text", "skip_annotate": false,
            "llm_provider": "gemini", "image_provider": "gemini" }
    """
    body = await request.json()
    message = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())[:8]
    skip_annotate = body.get("skip_annotate", False)
    label_style = body.get("label_style") or DEFAULT_LABEL_STYLE
    llm_provider = body.get("llm_provider")
    image_provider = body.get("image_provider")

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    steps_completed: list[dict] = []

    try:
        # ── Stage 1: Full diagram plan (drawing_prompt, labels, description) for precise label placement ──
        plan = create_plan(message, provider=llm_provider)
        steps_completed.append({
            "stage": "diagram_planner",
            "result": f"Plan ready with {len(plan.get('labels', []))} labels",
        })

        # ── Stage 2: Generate base image (no labels on image) ───────────────────
        raster_path = generate_image(plan, session_id, provider=image_provider)
        raster_url = f"/generated/{session_id}/raster.png"
        steps_completed.append({
            "stage": "image_generator",
            "result": raster_url,
        })

        annotated_png_url = None
        svg_url = None
        label_positions: list[dict] = []

        if not skip_annotate:
            # ── Stage 3: Vision-based label placement ──
            # Uses two-pass approach: direct coordinate detection + visual
            # verification for accurate structure localisation
            labels_from_plan = plan.get("labels") or []
            description = plan.get("description", "")
            if labels_from_plan:
                points = locate_labels(
                    image_path=raster_path,
                    labels=labels_from_plan,
                    description=description,
                    provider=llm_provider,
                )
                steps_completed.append({
                    "stage": "vision_label_placer",
                    "result": f"Located {len(points)} structures via vision analysis",
                })
            else:
                points = []
                steps_completed.append({
                    "stage": "vision_label_placer",
                    "result": "No labels in plan, skipping annotation",
                })

            if points:
                # ── Stage 4: Overlay labels onto image using returned coordinates ──
                from PIL import Image
                with Image.open(raster_path) as img:
                    w, h = img.size

                # Professional style uses balanced left/right distribution
                effective_label_side = plan.get("label_side", "right")
                if label_style == "professional":
                    effective_label_side = "integrated"

                label_positions = compute_label_layout(
                    points=points,
                    image_width=w,
                    image_height=h,
                    label_side=effective_label_side,
                )

                render_labels_on_png(
                    image_path=raster_path,
                    label_positions=label_positions,
                    style=label_style,  # type: ignore[arg-type]
                    session_id=session_id,
                    cover_placeholders=False,  # Clean images: no placeholders
                )
                annotated_png_url = f"/generated/{session_id}/annotated.png"

                render_labels_as_svg(
                    image_path=raster_path,
                    label_positions=label_positions,
                    style=label_style,  # type: ignore[arg-type]
                    session_id=session_id,
                )
                svg_url = f"/generated/{session_id}/annotated.svg"

                steps_completed.append({
                    "stage": "label_renderer",
                    "result": f"{label_style} style, {len(label_positions)} labels",
                })

        # Build plan for session/refiner: labels come from vision response
        plan["labels"] = [p["label"] for p in label_positions]
        plan["label_side"] = plan.get("label_side", "right")
        plan["style"] = plan.get("style", "colored_diagram")
        plan["diagram_type"] = plan.get("diagram_type", "anatomy")

        # ── Store session for refinement ──────────────────────
        sessions[session_id] = {
            "plan": plan,
            "label_positions": label_positions,
            "label_style": label_style,
            "raster_path": raster_path,
            "llm_provider": llm_provider,
            "image_provider": image_provider,
            "refiner": RefineSession(session_id, plan, label_positions, label_style),
        }

        return JSONResponse({
            "session_id": session_id,
            "steps": steps_completed,
            "raster_url": raster_url,
            "annotated_url": annotated_png_url,
            "svg_url": svg_url,
            "enhanced_prompt": plan.get("drawing_prompt", ""),
            "plan": {
                "labels": plan.get("labels", []),
                "description": plan.get("description", ""),
                "style": plan.get("style", ""),
                "diagram_type": plan.get("diagram_type", ""),
                "label_side": plan.get("label_side", ""),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "error": str(e),
            "steps": steps_completed,
            "session_id": session_id,
        }, status_code=500)


@app.post("/api/refine")
async def refine(request: Request):
    """
    Refine an existing figure.
    Body: { "session_id": "...", "message": "...", "label_style": "..." }
    """
    body = await request.json()
    session_id = body.get("session_id", "")
    message = body.get("message", "").strip()
    new_style = body.get("label_style")
    llm_provider = body.get("llm_provider")

    if session_id not in sessions:
        return JSONResponse(
            {"error": "Session not found. Generate a figure first."},
            status_code=404,
        )

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    session = sessions[session_id]
    refiner: RefineSession = session["refiner"]

    try:
        result = refiner.refine(message, new_style=new_style)
        print(f"[app] Refine result: action={result.get('action')}, needs_regen={result.get('needs_regeneration')}, needs_vision={result.get('needs_vision_rerun')}")

        if result.get("needs_regeneration") or result.get("needs_vision_rerun"):
            new_plan = result.get("new_plan", session["plan"])
            img_provider = body.get("image_provider") or session.get("image_provider")
            vision_provider = llm_provider or session.get("llm_provider")

            if result.get("action") == "regenerate":
                # Full regeneration: plan → image → vision locate → render
                refinement_req = new_plan.pop("_refinement_request", message)
                combined_prompt = (
                    session["plan"].get("description", "")
                    + ". " + refinement_req
                )
                print(f"[app] Re-running diagram planner for: {combined_prompt[:80]}...")
                new_plan = create_plan(combined_prompt, provider=vision_provider)

                raster_path = generate_image(
                    new_plan, session_id, provider=img_provider
                )
                raster_url = f"/generated/{session_id}/raster.png"

                labels_from_plan = new_plan.get("labels") or []
                description = new_plan.get("description", "")
                points = locate_labels(
                    image_path=raster_path,
                    labels=labels_from_plan,
                    description=description,
                    provider=vision_provider,
                )
                new_plan["labels"] = [p["label"] for p in points]
                new_plan["label_side"] = new_plan.get("label_side", "right")
            elif result.get("needs_vision_rerun"):
                # Re-run vision label placer on existing image (reposition / add_label)
                raster_path = session["raster_path"]
                raster_url = f"/generated/{session_id}/raster.png"
                description = new_plan.get("description", "")
                points = locate_labels(
                    image_path=raster_path,
                    labels=new_plan.get("labels", []),
                    description=description,
                    provider=vision_provider,
                )
            else:
                raster_path = generate_image(
                    new_plan, session_id, provider=img_provider
                )
                raster_url = f"/generated/{session_id}/raster.png"
                labels_from_plan = new_plan.get("labels") or []
                description = new_plan.get("description", "")
                points = locate_labels(
                    image_path=raster_path,
                    labels=labels_from_plan,
                    description=description,
                    provider=vision_provider,
                )
                new_plan["labels"] = [p["label"] for p in points]
                new_plan["label_side"] = new_plan.get("label_side", "right")

            from PIL import Image
            with Image.open(raster_path) as img:
                w, h = img.size

            style = new_style or session["label_style"]
            # Professional style uses balanced left/right distribution
            effective_label_side = new_plan.get("label_side", "right")
            if style == "professional":
                effective_label_side = "integrated"

            label_positions = compute_label_layout(
                points=points,
                image_width=w,
                image_height=h,
                label_side=effective_label_side,
            )

            render_labels_on_png(
                image_path=raster_path,
                label_positions=label_positions,
                style=style,  # type: ignore[arg-type]
                session_id=session_id,
                cover_placeholders=False,  # Clean images: no placeholders
            )
            render_labels_as_svg(
                image_path=raster_path,
                label_positions=label_positions,
                style=style,  # type: ignore[arg-type]
                session_id=session_id,
            )

            # Update session
            session["plan"] = new_plan
            session["label_positions"] = label_positions
            session["label_style"] = style
            session["raster_path"] = raster_path
            refiner.plan = new_plan
            refiner.label_positions = label_positions
            refiner.label_style = style

            resp = {
                "session_id": session_id,
                "action": "regenerated",
                "explanation": result["explanation"],
                "raster_url": raster_url,
                "annotated_url": f"/generated/{session_id}/annotated.png",
                "svg_url": f"/generated/{session_id}/annotated.svg",
            }
            if result.get("action") == "regenerate" and new_plan.get("drawing_prompt"):
                resp["enhanced_prompt"] = new_plan["drawing_prompt"]
            return JSONResponse(resp)
        else:
            # Label-only or style-only edits — just re-render
            annotated_url = None
            svg_url_out = None
            if result.get("annotated_png"):
                annotated_url = f"/generated/{session_id}/annotated.png"
            if result.get("annotated_svg"):
                svg_url_out = f"/generated/{session_id}/annotated.svg"

            return JSONResponse({
                "session_id": session_id,
                "action": result["action"],
                "explanation": result["explanation"],
                "annotated_url": annotated_url,
                "svg_url": svg_url_out,
            })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return JSONResponse({
        sid: {
            "labels": s["plan"].get("labels", []),
            "description": s["plan"].get("description", ""),
        }
        for sid, s in sessions.items()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
