"""Turn source files into semantically meaningful chunks.

Naive code search splits files into fixed-size windows, which slices functions
in half and destroys the unit people actually search for. This chunker keeps a
*symbol* (function / method / class) as one chunk with its real line span.

- Python: parsed with the stdlib ``ast`` module -> exact symbol boundaries,
  qualified names (``Class.method``), docstrings, and kinds.
- Other languages: a structural heuristic that detects declarations with
  per-language regexes and falls back to overlapping line windows so nothing is
  ever lost. (Swap in tree-sitter here to get Python-grade precision for every
  language -- the chunk interface stays identical.)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    path: str
    symbol: str          # qualified name, e.g. "LocalHashingEmbedder.encode"
    kind: str            # function | method | class | block
    start_line: int
    end_line: int
    code: str
    docstring: str = ""
    language: str = ""

    def embed_text(self) -> str:
        """What we actually embed: symbol + docstring weighted, then code."""
        symbol_words = self.symbol.replace(".", " ")
        # repeat the symbol/docstring so identifier intent dominates the vector
        return f"{symbol_words}\n{symbol_words}\n{self.docstring}\n{self.code}"

    def to_dict(self) -> dict:
        return asdict(self)


_LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c",
    ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cs": "csharp",
    ".php": "php", ".swift": "swift", ".kt": "kotlin",
}

# Declaration patterns for the heuristic (non-Python) path.
_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|static\s+|async\s+)*"
    r"(?P<kind>function|func|fn|def|class|interface|struct|type)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+(?P<arrow>[A-Za-z_][A-Za-z0-9_]*)\s*="
    r"\s*(?:async\s+)?\(?[^=]*?=>",
)


def language_for(path: str) -> str:
    for ext, lang in _LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return ""


def chunk_python(path: str, source: str) -> list[Chunk]:
    """Extract every function, method and class as its own chunk."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_generic(path, source, language="python")

    lines = source.splitlines()
    chunks: list[Chunk] = []

    def emit(node: ast.AST, qualname: str, kind: str) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        code = "\n".join(lines[start - 1 : end])
        doc = ast.get_docstring(node) or ""  # type: ignore[arg-type]
        chunks.append(
            Chunk(path, qualname, kind, start, end, code, doc, "python")
        )

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qn = f"{prefix}{child.name}"
                emit(child, qn, "class")
                walk(child, qn + ".")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = f"{prefix}{child.name}"
                emit(child, qn, "method" if prefix else "function")

    walk(tree, "")
    return chunks


def chunk_generic(path: str, source: str, language: str = "") -> list[Chunk]:
    """Heuristic chunker: split on detected declarations, window the rest."""
    lines = source.splitlines()
    if not lines:
        return []

    # find declaration line numbers
    boundaries: list[tuple[int, str, str]] = []  # (line_idx, kind, name)
    for i, line in enumerate(lines):
        m = _DECL_RE.match(line)
        if m:
            name = m.group("name") or m.group("arrow") or "anonymous"
            kind = m.group("kind") or "function"
            boundaries.append((i, kind, name))

    chunks: list[Chunk] = []
    if boundaries:
        for idx, (start_idx, kind, name) in enumerate(boundaries):
            end_idx = (
                boundaries[idx + 1][0] - 1
                if idx + 1 < len(boundaries)
                else len(lines) - 1
            )
            code = "\n".join(lines[start_idx : end_idx + 1])
            mapped = "class" if kind in {"class", "interface", "struct"} else "function"
            chunks.append(
                Chunk(path, name, mapped, start_idx + 1, end_idx + 1, code,
                      language=language or language_for(path))
            )
        return chunks

    # no declarations found -> overlapping windows so content is still searchable
    window, overlap = 40, 10
    step = window - overlap
    for start_idx in range(0, len(lines), step):
        block = lines[start_idx : start_idx + window]
        if not "".join(block).strip():
            continue
        chunks.append(
            Chunk(path, f"block@{start_idx + 1}", "block", start_idx + 1,
                  min(start_idx + window, len(lines)), "\n".join(block),
                  language=language or language_for(path))
        )
    return chunks


def chunk_file(path: str, source: str) -> list[Chunk]:
    if language_for(path) == "python":
        return chunk_python(path, source)
    return chunk_generic(path, source)
