from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from face_health import (
    DISCLAIMER,
    MODEL_PATH,
    FaceAnalyzer,
    analyze_image,
)


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
UPLOADS_DIR = RESULTS_DIR / "uploads"

RESULTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# File settings
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MAX_FILE_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_available: bool
    version: str


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Face Health Screening API",
    description="Screening-level facial feature analysis API.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Allows the Vite frontend and VS Code Dev Tunnel frontend
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
analyzer: FaceAnalyzer | None = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    global analyzer

    analyzer = FaceAnalyzer()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
@app.on_event("shutdown")
def shutdown_event() -> None:
    if analyzer is not None:
        release_method = getattr(analyzer, "release", None)

        if callable(release_method):
            release_method()


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="running",
        model_available=MODEL_PATH.exists(),
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_available=MODEL_PATH.exists(),
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Analyze image
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:

    # Check analyzer
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Analyzer is not ready. Please try again.",
        )

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    # Check extension
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPG, PNG, WEBP, or BMP.",
        )

    # Read uploaded file
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be less than 10 MB.",
        )

    # Create unique request folder
    request_id = uuid.uuid4().hex

    request_dir = UPLOADS_DIR / request_id

    request_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save original image
    original_path = request_dir / f"original{extension}"

    try:

        original_path.write_bytes(file_bytes)

        # Run face health analysis
        report = analyze_image(
            path=original_path,
            analyzer=analyzer,
            out_dir=request_dir,
            model_path=MODEL_PATH,
            show=False,
        )

        # Create URL for annotated image
        if report.get("annotated_image"):

            annotated_path = Path(
                report["annotated_image"]
            )

            report["annotated_image_url"] = (
                f"/results/{request_id}/{annotated_path.name}"
            )

        # Add request ID
        report["request_id"] = request_id

        # Add disclaimer
        report["disclaimer"] = DISCLAIMER

        return report

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Get result image
# ---------------------------------------------------------------------------
@app.get("/results/{request_id}/{filename}")
def get_result_file(
    request_id: str,
    filename: str,
) -> FileResponse:

    # Prevent path traversal
    safe_filename = Path(filename).name

    result_path = (
        UPLOADS_DIR
        / request_id
        / safe_filename
    )

    if not result_path.exists() or not result_path.is_file():

        raise HTTPException(
            status_code=404,
            detail="Result file was not found.",
        )

    return FileResponse(result_path)


# ---------------------------------------------------------------------------
# Delete result
# ---------------------------------------------------------------------------
@app.delete("/results/{request_id}")
def delete_result(
    request_id: str,
) -> dict[str, str]:

    request_dir = UPLOADS_DIR / request_id

    if not request_dir.exists():

        raise HTTPException(
            status_code=404,
            detail="Result does not exist.",
        )

    shutil.rmtree(request_dir)

    return {
        "status": "deleted",
        "request_id": request_id,
    }