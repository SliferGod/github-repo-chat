# repochat

A dark, minimalist GitHub repository chat app. Paste any public GitHub URL, it indexes the codebase using **LlamaIndex** + BGE embeddings, then lets you ask questions via a RAG pipeline powered by **Groq** (Llama 3.3 70B) with **Gemini 2.5 Flash** as fallback.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Vercel serverless) |
| RAG / Indexing | LlamaIndex VectorStoreIndex |
| Embeddings | `BAAI/bge-base-en-v1.5` (local, HuggingFace) |
| LLM (primary) | Groq — `llama-3.3-70b-versatile` |
| LLM (fallback) | Gemini 2.5 Flash |
| Frontend | Vanilla HTML/CSS/JS — dark monospace terminal aesthetic |

---

## Local development

### 1. Clone and install

```bash
git clone https://github.com/yourname/repochat
cd repochat
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and/or GEMINI_API_KEY
```

### 3. Run

```bash
cd api
uvicorn index:app --reload --port 8000
```

Open `frontend/index.html` in a browser, or visit `http://localhost:8000`.

---

## Deploy to Vercel

### 1. Set secrets in Vercel dashboard

```
GROQ_API_KEY      → your Groq API key
GEMINI_API_KEY    → your Gemini API key (fallback)
GITHUB_TOKEN      → GitHub PAT (optional, raises rate limit to 5000/hr)
```

### 2. Deploy

```bash
npm i -g vercel
vercel --prod
```

Vercel will pick up `vercel.json` automatically.

> **Note on cold starts:** The BGE embedding model (~440 MB) is downloaded on first boot. Subsequent requests reuse the warm instance. Consider pre-baking model weights into a Docker image (see `Dockerfile.example`) for production use.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/index-repo` | `{"github_url": "..."}` → fetches + indexes repo |
| `POST` | `/api/chat` | `{"repo_key": "...", "question": "..."}` → RAG answer |
| `GET` | `/api/repo-info/{key}` | Check if a repo key is indexed |

---

## How it works

1. **Index**: The backend calls the GitHub Contents API to fetch all text-based source files (`.py`, `.js`, `.ts`, `.md`, etc.) from the repo tree. Each file becomes a `Document`. LlamaIndex builds a `VectorStoreIndex` using BGE embeddings and persists it to `/tmp/<repo_key>/`.

2. **Chat**: Each question runs through LlamaIndex's `as_query_engine()` with `similarity_top_k=5`. The retrieved nodes are passed to Groq (or Gemini) to generate a grounded answer. Source file paths and snippets are returned alongside the answer.

3. **Caching**: Indexes persist in `/tmp` for the lifetime of the serverless instance. Re-indexing the same repo URL is a no-op if the key already exists.

---

## Limitations

- Public repos only (no auth). Add a `GITHUB_TOKEN` to raise the rate limit.
- Large monorepos are capped at 300 files (configurable via `MAX_FILES` in `repo_store.py`).
- `/tmp` storage is ephemeral on Vercel — indexes are rebuilt on cold start.
- The BGE model requires ~2 GB RAM; ensure your Vercel plan supports it.
