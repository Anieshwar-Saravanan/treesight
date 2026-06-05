"""RAG ask-mode: retrieve the most relevant chunks, then have a *local* LLM
turn them into a cited natural-language answer.

The retrieval half is `search.Searcher` (unchanged). The generation half talks
to a local model server -- Ollama by default (https://ollama.com), reached over
its HTTP API at ``localhost:11434`` -- so no API key, network, or data leaves
the machine. urllib only, matching the dependency-free style of `_APIEmbedder`.

    treesight ask "how are query and index tokens hashed the same way?"

The LLM only ever sees the retrieved chunks, and is told to answer strictly
from them and cite ``path:line`` -- so answers stay grounded in real code.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .search import Hit, Searcher

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

_SYSTEM = (
    "You are a precise code assistant. Answer the question using ONLY the "
    "code excerpts provided. Each excerpt is labelled with its source as "
    "path:start-end. Cite the sources you use inline as `path:line`. If the "
    "excerpts do not contain the answer, say so plainly instead of guessing. "
    "Be concise and concrete; refer to real symbol and file names."
)


@dataclass
class Answer:
    text: str
    model: str
    hits: list[Hit]


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        header = f"# {h.path}:{h.start_line}-{h.end_line}  ({h.kind} {h.symbol})"
        blocks.append(f"{header}\n{h.code}")
    return "\n\n".join(blocks)


def build_prompt(query: str, hits: list[Hit]) -> str:
    return (
        f"Question: {query}\n\n"
        f"Code excerpts retrieved from the repository:\n\n"
        f"{_format_context(hits)}\n\n"
        f"Answer the question using only the excerpts above, citing `path:line`."
    )


def _call_ollama(prompt: str, model: str, host: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # Ollama is reachable but returned an error -- surface its actual message.
        try:
            detail = json.loads(exc.read()).get("error", "")
        except Exception:
            detail = ""
        raise RuntimeError(
            f"Local LLM at {host} returned HTTP {exc.code}"
            + (f": {detail}" if detail else "")
            + f". Check the model `{model}` is pulled (`ollama list`) and that "
            "your Ollama install can actually run it."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach a local LLM at {host} ({exc}). "
            f"Install Ollama (https://ollama.com), then run "
            f"`ollama pull {model}` and make sure `ollama serve` is running."
        ) from exc
    return data.get("message", {}).get("content", "").strip()


def ask(
    query: str,
    index_root: str = ".",
    k: int = 6,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
) -> Answer:
    hits = Searcher(index_root).search(query, k=k)
    if not hits:
        raise RuntimeError("Index returned no chunks to reason over.")
    text = _call_ollama(build_prompt(query, hits), model, host)
    return Answer(text=text, model=model, hits=hits)
