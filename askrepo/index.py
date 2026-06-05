"""Build and persist a searchable index of a repository.

Layout written to ``<repo>/.askrepo_index/``:
  - vectors.npy   float32 (n_chunks, dim), L2-normalised
  - chunks.json   chunk metadata + source text
  - meta.json     provider name, dim, idf table (for the local embedder)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .chunker import chunk_file, language_for, Chunk
from .embeddings import Embedder, LocalHashingEmbedder, get_embedder

INDEX_DIR = ".askrepo_index"

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", "target", ".askrepo_index", ".mypy_cache", ".pytest_cache",
}
_SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".gz",
    ".lock", ".min.js", ".map", ".woff", ".woff2", ".ttf", ".mp4", ".mp3",
}
_MAX_BYTES = 400_000  # skip very large / generated files


def _iter_source_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if any(name.endswith(ext) for ext in _SKIP_EXT):
                continue
            path = Path(dirpath) / name
            if not language_for(str(path)):
                continue  # code search: skip prose/config/binaries
            try:
                if path.stat().st_size > _MAX_BYTES:
                    continue
            except OSError:
                continue
            yield path


def build_index(root: str, provider: str = "local", dim: int = 512) -> dict:
    root_path = Path(root).resolve()
    embedder: Embedder = get_embedder(provider, dim=dim)

    chunks: list[Chunk] = []
    for path in _iter_source_files(root_path):
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(root_path))
        chunks.extend(chunk_file(rel, source))

    if not chunks:
        raise RuntimeError(f"No source chunks found under {root_path}")

    embed_texts = [c.embed_text() for c in chunks]
    embedder.fit(embed_texts)
    vectors = embedder.encode(embed_texts)

    out_dir = root_path / INDEX_DIR
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=0)
    )
    meta = {
        "provider": embedder.name,
        "provider_key": provider,
        "dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "n_files": len({c.path for c in chunks}),
    }
    if isinstance(embedder, LocalHashingEmbedder):
        meta["idf"] = embedder._idf
        meta["default_idf"] = embedder._default_idf
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta
