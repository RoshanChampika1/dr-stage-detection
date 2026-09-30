"""FastAPI service for diabetic retinopathy stage detection.

Endpoints:
    GET  /health    service status and model metadata (also wakes the server)
    POST /predict   multipart upload "file" -> stage, probabilities, Grad-CAM
    GET  /app/      the web page (the same files that are hosted on Vercel)

Run locally from the project root:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/app/

Environment variables:
    MODEL_PATH       checkpoint to load (default api/model/best.pt)
    MODEL_CARD       model card JSON (default api/model_card.json)
    ALLOWED_ORIGINS  comma-separated origins for CORS (default "*")
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.inference import Predictor, decode_image

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


def create_app(model_path: str | Path | None = None, card_path: str | Path | None = None) -> FastAPI:
    """Build the app. The model is loaded once at start-up, not per request."""
    model_path = Path(model_path or os.environ.get("MODEL_PATH", ROOT / "api" / "model" / "best.pt"))
    card_path = Path(card_path or os.environ.get("MODEL_CARD", ROOT / "api" / "model_card.json"))
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["predictor"] = Predictor(model_path, card_path)
        yield
        state.clear()

    app = FastAPI(title="DR Stage Detection API", version="1.0", lifespan=lifespan)

    origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": state["predictor"].info()}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if file.content_type and file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Use JPG or PNG.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than 10 MB.")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            img = decode_image(data)
        except ValueError as err:
            raise HTTPException(400, str(err)) from err
        return state["predictor"].predict(img)

    web_dir = ROOT / "web"
    if web_dir.exists():
        app.mount("/app", StaticFiles(directory=web_dir, html=True), name="web")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/app/")

    return app


app = create_app()
