"""Resolve a repo *source* -- a local path or a GitHub URL -- to a local dir.

So every command works the same whether you point it at a folder or a link:

    askrepo index  https://github.com/pallets/flask
    askrepo ask "how is routing done?" --index https://github.com/pallets/flask

GitHub URLs are shallow-cloned once into a cache under ``~/.askrepo/repos/`` and
reused afterwards (pass ``--refresh`` to re-pull). Anything that isn't a GitHub
URL is treated as a plain local path and returned unchanged.

Supported URL forms (with or without a trailing ``.git`` / ``/``):
    https://github.com/owner/repo
    http://github.com/owner/repo
    git@github.com:owner/repo
    github.com/owner/repo
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

CACHE_DIR = Path.home() / ".askrepo" / "repos"

_GITHUB_RE = re.compile(
    r"^(?:https?://|git@)?github\.com[/:]"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


def _match(source: str) -> re.Match[str] | None:
    return _GITHUB_RE.match(source.strip())


def is_github_url(source: str) -> bool:
    return _match(source) is not None


def cache_path_for(source: str) -> Path | None:
    """Local cache directory a GitHub URL maps to, or None if not a URL."""
    m = _match(source)
    if not m:
        return None
    return CACHE_DIR / f"{m['owner']}__{m['repo']}"


def resolve_source(source: str, clone: bool = True, refresh: bool = False) -> str:
    """Return a local directory for ``source``.

    Plain local paths are returned unchanged. GitHub URLs map to a cache dir,
    which is shallow-cloned when ``clone`` is True and missing (and re-pulled
    when ``refresh`` is True). With ``clone=False`` the path is returned without
    touching the network -- so a missing index surfaces the normal error.
    """
    dest = cache_path_for(source)
    if dest is None:
        return source

    m = _match(source)
    url = f"https://github.com/{m['owner']}/{m['repo']}.git"  # type: ignore[index]

    if dest.exists():
        if refresh:
            _run(["git", "-C", str(dest), "pull", "--ff-only"], f"update {url}")
        return str(dest)
    if not clone:
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)], f"clone {url}")
    return str(dest)


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to {what}:\n{detail}") from exc
