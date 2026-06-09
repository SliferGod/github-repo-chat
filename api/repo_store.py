"""GitHub repository fetching and LlamaIndex indexing.

Fetches file contents from a public GitHub repo via the GitHub Contents API,
builds a VectorStoreIndex in /tmp/<repo_key>/, and persists it there for
the lifetime of the serverless instance (warm reuse).

Supported file types for indexing (text-based source files):
.py .js .ts .jsx .tsx .java .go .rs .cpp .c .h .cs .rb .php .swift
.kt .md .txt .yaml .yml .toml .json .html .css .sh .env.example README
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional
import requests

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage

TMP_ROOT = Path("/tmp/github_chat_indexes")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional — raises rate limit to 5000/hr

TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".html",
    ".css", ".scss", ".sh", ".bash", ".env", ".example",
    ".sql", ".graphql", ".proto", ".xml", ".ini", ".cfg",
}
MAX_FILE_SIZE = 150_000   # bytes — skip minified / generated blobs
MAX_FILES = 300           # cap to keep indexing fast on large monorepos


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _repo_key(owner: str, repo: str, ref: str) -> str:
    slug = f"{owner}/{repo}@{ref}"
    return hashlib.md5(slug.encode()).hexdigest()[:16]


def parse_github_url(url: str) -> tuple[str, str, str]:
    """
    Parse a GitHub URL and return (owner, repo, ref).

    Accepted formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/tree/branch
    - github.com/owner/repo
    """
    url = url.strip().rstrip("/")
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^github\.com/", "", url)

    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse GitHub URL: {url!r}")

    owner, repo = parts[0], parts[1]

    # /tree/<ref> or /blob/<ref>/...
    ref = "HEAD"
    if len(parts) >= 4 and parts[2] in ("tree", "blob"):
        ref = parts[3]

    return owner, repo, ref


def _list_tree(owner: str, repo: str, ref: str) -> list[dict]:
    """Return flat list of blob entries from the git tree (recursive)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    resp = requests.get(url, headers=_headers(), timeout=20)
    if resp.status_code == 404:
        raise ValueError(f"Repository not found or is private: {owner}/{repo}")
    resp.raise_for_status()
    data = resp.json()
    return [item for item in data.get("tree", []) if item.get("type") == "blob"]


def _fetch_file(owner: str, repo: str, path: str) -> Optional[str]:
    """Fetch raw file content as text. Returns None on failure or binary."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(), timeout=15)
    if not resp.ok:
        return None
    data = resp.json()
    size = data.get("size", 0)
    if size > MAX_FILE_SIZE:
        return None
    # GitHub returns base64-encoded content
    import base64
    raw = data.get("content", "")
    try:
        text = base64.b64decode(raw).decode("utf-8", errors="replace")
        return text
    except Exception:
        return None


def _should_index(path: str, size: int) -> bool:
    p = Path(path)
    name_lower = p.name.lower()
    # Always include README regardless of extension
    if name_lower.startswith("readme"):
        return True
    return p.suffix.lower() in TEXT_EXTENSIONS and size <= MAX_FILE_SIZE


def fetch_and_index(github_url: str) -> dict:
    """
    Fetch a GitHub repo, build a VectorStoreIndex, persist to /tmp.

    Returns a metadata dict with: owner, repo, ref, repo_key, file_count, index_dir.
    Reuses an existing index if the key already exists in /tmp.
    """
    owner, repo, ref = parse_github_url(github_url)
    repo_key = _repo_key(owner, repo, ref)
    index_dir = TMP_ROOT / repo_key

    # Warm reuse — already indexed this session
    if index_dir.exists() and any(index_dir.iterdir()):
        return {
            "owner": owner, "repo": repo, "ref": ref,
            "repo_key": repo_key, "cached": True,
            "index_dir": str(index_dir),
        }

    # Fetch tree
    tree = _list_tree(owner, repo, ref)

    # Filter to text blobs
    candidates = [
        item for item in tree
        if _should_index(item["path"], item.get("size", 0))
    ][:MAX_FILES]

    if not candidates:
        raise ValueError("No indexable text files found in this repository.")

    # Fetch content and build Documents
    documents: list[Document] = []
    fetched = 0
    for item in candidates:
        content = _fetch_file(owner, repo, item["path"])
        if content and content.strip():
            documents.append(Document(
                text=content,
                metadata={
                    "file_path": item["path"],
                    "repo": f"{owner}/{repo}",
                    "ref": ref,
                    "size": item.get("size", 0),
                },
                doc_id=f"{repo_key}::{item['path']}",
            ))
            fetched += 1
        time.sleep(0.05)   # gentle rate-limit courtesy delay

    if not documents:
        raise ValueError("Could not fetch any file contents from this repository.")

    # Build and persist VectorStoreIndex
    index_dir.mkdir(parents=True, exist_ok=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=str(index_dir))

    return {
        "owner": owner,
        "repo": repo,
        "ref": ref,
        "repo_key": repo_key,
        "cached": False,
        "file_count": fetched,
        "index_dir": str(index_dir),
    }


def load_index(repo_key: str) -> VectorStoreIndex:
    index_dir = TMP_ROOT / repo_key
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(f"No index found for repo_key={repo_key}. Re-index the repository.")
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(index_dir)))


def query_repo(repo_key: str, question: str, llm) -> dict:
    """Run a RAG query against the indexed repository."""
    index = load_index(repo_key)
    engine = index.as_query_engine(similarity_top_k=5, llm=llm)
    response = engine.query(question)

    sources = []
    for node in response.source_nodes:
        meta = node.node.metadata
        sources.append({
            "file_path": meta.get("file_path", "unknown"),
            "score": round(node.score, 4) if node.score is not None else None,
            "snippet": node.node.get_content()[:300],
        })

    return {"answer": str(response), "sources": sources}
