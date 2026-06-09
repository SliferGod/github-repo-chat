"""LLM + embedding initialisation for GitHub Chat.

Provider priority:
- Groq llama-3.3-70b-versatile  — primary (higher RPM, fast)
- Gemini 2.5 Flash               — fallback when GROQ_API_KEY is absent
- BGE base                       — local embeddings (no API quota)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Settings

# Load .env — check the current dir, the api/ parent, and one level above that
# so this works whether uvicorn is launched from the project root or from api/
for _candidate in (
    Path.cwd() / ".env",
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)
        break

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    raise RuntimeError(
        "At least one of GROQ_API_KEY or GEMINI_API_KEY must be set. "
        "Create a .env file in the project root with your keys (see .env.example)."
    )

groq_llm = None
gemini_llm = None

if GROQ_API_KEY:
    from llama_index.llms.groq import Groq
    groq_llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

if GEMINI_API_KEY:
    from llama_index.llms.google_genai import GoogleGenAI
    gemini_llm = GoogleGenAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)

# Prefer Groq; fall back to Gemini
primary_llm = groq_llm or gemini_llm

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

Settings.llm = primary_llm
Settings.embed_model = embed_model


def get_llm():
    """Return the best available LLM."""
    return groq_llm or gemini_llm