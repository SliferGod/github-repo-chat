"""GitHub Chat — FastAPI backend.

Endpoints
---------
GET  /api/health              — liveness probe
POST /api/index-repo          — fetch + index a GitHub repository
POST /api/chat                — RAG query against an indexed repo
GET  /api/repo-info/{key}     — metadata for an indexed repo
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# Boot LLMs and embedding model once at import time.
from llm_setup import get_llm
import repo_store

app = FastAPI(title="GitHub Chat API")

# ── CORS ──────────────────────────────────────────────────────────────────────
_origins = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]
_extra = os.getenv("FRONTEND_ORIGIN", "")
if _extra:
    _origins += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Vercel preview URLs are dynamic; lock down in prod via FRONTEND_ORIGIN
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files (only if the static subdirectory exists)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class IndexRequest(BaseModel):
    github_url: str


class ChatRequest(BaseModel):
    repo_key: str
    question: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/index-repo")
def index_repo(body: IndexRequest):
    """Fetch a GitHub repo and build a VectorStoreIndex. Returns repo metadata + repo_key."""
    if not body.github_url or "github.com" not in body.github_url:
        raise HTTPException(status_code=400, detail="Please provide a valid GitHub URL.")
    try:
        meta = repo_store.fetch_and_index(body.github_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
    return meta


@app.post("/api/chat")
def chat(body: ChatRequest):
    """RAG query against an already-indexed repository."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = repo_store.query_repo(body.repo_key, body.question, get_llm())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    return result


@app.get("/api/repo-info/{repo_key}")
def repo_info(repo_key: str):
    """Check whether a repo_key is currently indexed in /tmp."""
    from pathlib import Path
    index_dir = repo_store.TMP_ROOT / repo_key
    if not index_dir.exists():
        raise HTTPException(status_code=404, detail="Repo not indexed in this instance.")
    return {"repo_key": repo_key, "index_dir": str(index_dir), "indexed": True}


# ── Root — serve frontend ─────────────────────────────────────────────────────

@app.get("/")
def root():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "GitHub Chat API is running. See /docs for API reference."}