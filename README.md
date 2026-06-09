# repochat

A GitHub repository chat app. Paste any public GitHub URL, index the codebase with a single click, then ask questions about it. It is powered by a RAG pipeline built on LlamaIndex.

![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-orange)

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| RAG / Indexing | LlamaIndex `VectorStoreIndex` |
| Embeddings | Gemini `text-embedding-004` (API-based) |
| LLM (primary) | Groq — `llama-3.3-70b-versatile` |
| LLM (fallback) | Gemini 2.5 Flash |
| Frontend | Vanilla HTML/CSS/JS — dark monospace terminal UI |
| Hosting | Railway (full-stack, single service) |

---

## How it works

1. **Index** — The backend calls the GitHub Contents API to fetch all text-based source files (`.py`, `.js`, `.ts`, `.md`, etc., up to 300 files). Each file becomes a LlamaIndex `Document`, embedded via Gemini and stored as a `VectorStoreIndex` in `/tmp/<repo_key>/`.

2. **Chat** — Each question runs through LlamaIndex's query engine with `similarity_top_k=5`. Retrieved nodes are passed to Groq (or Gemini) to generate a grounded answer. Source file paths and snippets are returned alongside the answer.

3. **Caching** — Indexes live in `/tmp` for the lifetime of the Railway container instance. Re-submitting the same repo URL reuses the existing index instantly.

---

## Local development

### 1. Install dependencies

```bash
git clone https://github.com/SliferGod/github-repo-chat
cd github-repo-chat
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_pat        # optional but recommended
```

### 3. Run

```bash
cd api
uvicorn index:app --reload
```

Then open `http://localhost:8000` in your browser.

---

## Deploy to Railway

Railway runs the full app — backend and frontend — in a single Docker container. No Vercel needed.

### 1. Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **New Project** → **Deploy from GitHub repo**
3. Select your repochat repository
4. Railway auto-detects the `Dockerfile` and starts building

### 2. Add environment variables

In your Railway service dashboard, go to **Variables** and add:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `GEMINI_API_KEY` | Your Gemini API key |
| `GITHUB_TOKEN` | Your GitHub PAT *(optional)* |

### 3. Expose a public URL

1. Click your service → **Settings** tab
2. Scroll to **Networking** → **Public Networking**
3. Click **Generate Domain**

Railway gives you a URL like `github-repo-chat-production.up.railway.app`. That's your live app — open it in a browser and it's ready to use.

---

## API reference

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/api/health` | — | Liveness probe |
| `POST` | `/api/index-repo` | `{"github_url": "..."}` | Fetch and index a repo |
| `POST` | `/api/chat` | `{"repo_key": "...", "question": "..."}` | RAG query |
| `GET` | `/api/repo-info/{key}` | — | Check if a key is indexed |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | (or Gemini) | Primary LLM — Llama 3.3 70B via Groq |
| `GEMINI_API_KEY` | (or Groq) | Fallback LLM + embeddings provider |
| `OPENAI_API_KEY` | Only if no Gemini key | Fallback embeddings via `text-embedding-3-small` |
| `GITHUB_TOKEN` | optional | Raises GitHub API rate limit from 60 → 5000 req/hr |

At least one LLM key and one embedding key are required. If `GEMINI_API_KEY` is set it covers both roles.

---

## Limitations

- **Public repos only** — private repos require passing a GitHub token in the API call (not currently implemented in the UI)
- **300 file cap** — large monorepos are trimmed to the first 300 indexable files (configurable via `MAX_FILES` in `repo_store.py`)
- **Ephemeral indexes** — `/tmp` storage is cleared on Railway container restarts; just re-index the repo after a restart
- **Rate limits** — without a `GITHUB_TOKEN`, the GitHub API allows only 60 unauthenticated requests per hour