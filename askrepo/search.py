"""Query a built index with a natural-language string."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import LocalHashingEmbedder, get_embedder
from .index import INDEX_DIR


@dataclass
class Hit:
    score: float
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    code: str


class Searcher:
    def __init__(self, root: str) -> None:
        self.dir = Path(root).resolve() / INDEX_DIR
        if not self.dir.exists():
            raise RuntimeError(
                f"No index at {self.dir}. Run `askrepo index <repo>` first."
            )
        self.vectors = np.load(self.dir / "vectors.npy")
        self.chunks = json.loads((self.dir / "chunks.json").read_text())
        self.meta = json.loads((self.dir / "meta.json").read_text())

        self.embedder = get_embedder(self.meta["provider_key"], dim=self.meta["dim"])
        if isinstance(self.embedder, LocalHashingEmbedder):
            # restore the IDF table learned at index time
            self.embedder._idf = self.meta.get("idf", {})
            self.embedder._default_idf = self.meta.get("default_idf", 1.0)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q  # cosine sim (both L2-normalised)
        top = np.argsort(-scores)[:k]
        hits = []
        for i in top:
            c = self.chunks[int(i)]
            hits.append(
                Hit(float(scores[i]), c["path"], c["symbol"], c["kind"],
                    c["start_line"], c["end_line"], c["code"])
            )
        return hits
