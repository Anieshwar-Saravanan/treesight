"""Embedding providers.

The index/search pipeline is provider-agnostic: anything that turns a list of
strings into an (n, dim) L2-normalised float32 matrix can be plugged in here.

`LocalHashingEmbedder` is the zero-dependency default. It is a hashed TF-IDF
vectoriser with code-aware tokenisation (it splits ``snake_case`` and
``camelCase`` into subwords), so a query like "cosine similarity" matches a
symbol named ``cosine_similarity``. It is *lexical*, not neural -- good enough
to demo and to be useful on most repos out of the box, but for true semantic
matching swap in one of the neural providers below (one config flag, same
pipeline).
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Tokens that carry no signal in code search.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "for", "on", "with",
    "self", "this", "return", "import", "from", "def", "class", "if", "else",
    "var", "let", "const", "function", "func", "fn", "public", "private",
}


def tokenize(text: str) -> list[str]:
    """Split text into code-aware subword tokens.

    ``getUserById`` -> ``get user by id``; ``parse_ast_node`` -> ``parse ast
    node``. The original identifier is kept too, so exact matches still rank.
    """
    out: list[str] = []
    for match in _TOKEN_RE.findall(text):
        ident = match.lower()
        if len(ident) > 2 and ident not in _STOPWORDS:
            out.append(ident)
        # split snake_case and camelCase into subwords
        for piece in _CAMEL_RE.sub(" ", match).replace("_", " ").split():
            piece = piece.lower()
            if len(piece) > 2 and piece not in _STOPWORDS and piece != ident:
                out.append(piece)
    return out


class Embedder(ABC):
    """Turn strings into an (n, dim) L2-normalised matrix."""

    dim: int

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> None:
        """Optional corpus pass (e.g. to learn IDF weights)."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class LocalHashingEmbedder(Embedder):
    """Hashed TF-IDF over code-aware tokens. No network, no model download."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._idf: dict[str, float] = {}
        self._default_idf: float = 1.0

    @property
    def name(self) -> str:
        return f"local-hashing(dim={self.dim})"

    def fit(self, corpus: Sequence[str]) -> None:
        n_docs = max(len(corpus), 1)
        doc_freq: Counter[str] = Counter()
        for doc in corpus:
            for tok in set(tokenize(doc)):
                doc_freq[tok] += 1
        self._idf = {
            tok: math.log((n_docs + 1) / (df + 1)) + 1.0
            for tok, df in doc_freq.items()
        }
        # unseen query terms get the IDF of a term seen once
        self._default_idf = math.log((n_docs + 1) / 1) + 1.0

    @staticmethod
    def _fnv1a(token: str) -> int:
        """Deterministic 32-bit hash (Python's built-in hash() is salted per
        process, which would scatter index- and query-time tokens into
        different buckets)."""
        h = 0x811C9DC5
        for byte in token.encode("utf-8"):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h

    def _hash(self, token: str) -> tuple[int, int]:
        h = self._fnv1a(token)
        index = h % self.dim
        sign = 1 if (h >> 31) & 1 == 0 else -1  # signed hashing reduces collisions
        return index, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            for tok, tf in counts.items():
                idf = self._idf.get(tok, self._default_idf)
                weight = (1.0 + math.log(tf)) * idf  # sublinear tf
                index, sign = self._hash(tok)
                vectors[row, index] += sign * weight
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class _APIEmbedder(Embedder):
    """Shared logic for HTTP embedding providers (OpenAI, Voyage, ...).

    Not exercised in the offline demo, but wired up so you can flip to real
    semantic embeddings with an API key and zero pipeline changes.
    """

    endpoint: str = ""
    model: str = ""
    env_key: str = ""

    def __init__(self, model: str | None = None, dim: int = 1536) -> None:
        self.dim = dim
        if model:
            self.model = model

    def fit(self, corpus: Sequence[str]) -> None:  # neural models need no IDF pass
        return None

    def _payload(self, texts: Sequence[str]) -> dict:
        return {"model": self.model, "input": list(texts)}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import json
        import os
        import urllib.request

        token = os.environ.get(self.env_key)
        if not token:
            raise RuntimeError(
                f"{self.name} needs {self.env_key} set in the environment."
            )
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self._payload(texts)).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        vecs = np.array([row["embedding"] for row in data["data"]], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dim = vecs.shape[1]
        return vecs / norms


class OpenAIEmbedder(_APIEmbedder):
    endpoint = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-small"
    env_key = "OPENAI_API_KEY"

    @property
    def name(self) -> str:
        return f"openai({self.model})"


class VoyageEmbedder(_APIEmbedder):
    endpoint = "https://api.voyageai.com/v1/embeddings"
    model = "voyage-code-3"  # purpose-built for code retrieval
    env_key = "VOYAGE_API_KEY"

    @property
    def name(self) -> str:
        return f"voyage({self.model})"


def get_embedder(provider: str, dim: int = 512) -> Embedder:
    provider = provider.lower()
    if provider == "local":
        return LocalHashingEmbedder(dim=dim)
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"Unknown provider: {provider!r} (use local|openai|voyage)")
