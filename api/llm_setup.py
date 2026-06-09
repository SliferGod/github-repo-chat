"""LLM + embedding initialisation for GitHub Chat.

LLM priority:
- Groq llama-3.3-70b-versatile  — primary (higher RPM, fast)
- Gemini 2.5 Flash               — fallback when GROQ_API_KEY is absent

Embedding priority (all API-based — no torch/transformers required):
- Gemini text-embedding-004      — if GEMINI_API_KEY set
- OpenAI text-embedding-3-small  — if OPENAI_API_KEY set
- Groq does not offer an embeddings API, so we fall through to Gemini/OpenAI
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings

# Load .env — check cwd, api/, and project root
for _candidate in (
    Path.cwd() / ".env",
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise RuntimeError(
        "At least one of GROQ_API_KEY or GEMINI_API_KEY must be set. "
        "Create a .env file in the project root (see .env.example)."
    )

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise RuntimeError(
        "An embedding API key is required for Vercel deployment. "
        "Set GEMINI_API_KEY (preferred) or OPENAI_API_KEY in your environment."
    )

# ── LLMs ──────────────────────────────────────────────────────────────────────
groq_llm = None
gemini_llm = None

if GROQ_API_KEY:
    from llama_index.llms.groq import Groq
    groq_llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

if GEMINI_API_KEY:
    from llama_index.llms.google_genai import GoogleGenAI
    gemini_llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

primary_llm = groq_llm or gemini_llm

# ── Embeddings (API-based, no local model weights) ────────────────────────────
if GEMINI_API_KEY:
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-2-preview",
        api_key=GEMINI_API_KEY,
    )
elif OPENAI_API_KEY:
    from llama_index.embeddings.openai import OpenAIEmbedding
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

Settings.llm = primary_llm
Settings.embed_model = embed_model


def get_llm():
    """Return the best available LLM."""
    return groq_llm or gemini_llm