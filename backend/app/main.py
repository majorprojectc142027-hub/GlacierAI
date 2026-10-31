from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.glaciers import router as glaciers_router
from backend.app.api.upload import router as upload_router
from backend.app.api.live import router as live_router
from backend.app.api.historical import router as historical_router
from backend.app.api.classification import (
    router as classification_router
)
from backend.app.api.debris import router as debris_router


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"

DATASETS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(
    title="GlacierAI API",
    description="AI-powered Glacier Monitoring System",
    version="1.0.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# STATIC DATASET FILES
# ==========================================================

app.mount(
    "/datasets",
    StaticFiles(directory=str(DATASETS_DIR)),
    name="datasets",
)


# ==========================================================
# API ROUTES
# ==========================================================

# Upload API
app.include_router(
    upload_router,
)


# Live Glacier Analysis API
app.include_router(
    live_router,
    prefix="/api/live",
    tags=["Live Analysis"],
)


# Glacier Search API
app.include_router(
    glaciers_router,
    prefix="/api/glaciers",
    tags=["Glaciers"],
)


# Historical Analysis API
app.include_router(
    historical_router,
    prefix="/api",
    tags=["Historical Analysis"],
)


# Glacier Type Classification API
app.include_router(
    classification_router,
    prefix="/api/classification",
    tags=["Glacier Classification"],
)


# Debris Analysis API
app.include_router(
    debris_router,
    prefix="/api/debris",
    tags=["Debris Analysis"],
)


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to GlacierAI Backend!",
        "status": "Running Successfully",
    }