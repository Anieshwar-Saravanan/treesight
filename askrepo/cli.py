"""Command-line interface: `python -m askrepo index|search|ask ...`."""

from __future__ import annotations

import argparse
import sys

from .ask import DEFAULT_HOST, DEFAULT_MODEL, ask
from .index import build_index
from .search import Searcher

# ANSI colours (no dependency); disabled automatically when piped.
_C = sys.stdout.isatty()
DIM = "\033[2m" if _C else ""
BOLD = "\033[1m" if _C else ""
CYAN = "\033[36m" if _C else ""
GREEN = "\033[32m" if _C else ""
YELLOW = "\033[33m" if _C else ""
RESET = "\033[0m" if _C else ""


def _cmd_index(args: argparse.Namespace) -> int:
    print(f"{DIM}Indexing {args.path} with provider={args.provider}...{RESET}")
    meta = build_index(args.path, provider=args.provider, dim=args.dim)
    print(
        f"{GREEN}Indexed {meta['n_chunks']} chunks "
        f"from {meta['n_files']} files{RESET} "
        f"{DIM}(provider={meta['provider']}, dim={meta['dim']}){RESET}"
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    searcher = Searcher(args.index)
    hits = searcher.search(args.query, k=args.k)
    print(f"\n{BOLD}Query:{RESET} {args.query}\n")
    for rank, hit in enumerate(hits, 1):
        bar = "#" * round(hit.score * 20)
        print(
            f"{BOLD}{rank}.{RESET} {CYAN}{hit.path}{RESET}"
            f"{DIM}:{hit.start_line}-{hit.end_line}{RESET}  "
            f"{YELLOW}{hit.kind} {hit.symbol}{RESET}  "
            f"{GREEN}{hit.score:.3f}{RESET} {DIM}{bar}{RESET}"
        )
        snippet = hit.code.splitlines()[: args.lines]
        for line in snippet:
            print(f"   {DIM}|{RESET} {line}")
        if len(hit.code.splitlines()) > args.lines:
            print(f"   {DIM}| ...{RESET}")
        print()
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    answer = ask(args.query, index_root=args.index, k=args.k,
                 model=args.model, host=args.host)
    print(f"\n{BOLD}Q:{RESET} {args.query}\n")
    print(answer.text)
    print(f"\n{DIM}Sources ({answer.model}):{RESET}")
    for hit in answer.hits:
        print(
            f"  {CYAN}{hit.path}{RESET}{DIM}:{hit.start_line}-{hit.end_line}{RESET}"
            f"  {YELLOW}{hit.kind} {hit.symbol}{RESET}  {GREEN}{hit.score:.3f}{RESET}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="askrepo",
        description="Semantic search over a codebase (AST-aware chunking + vectors).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build a search index for a repo")
    p_index.add_argument("path", help="path to the repository root")
    p_index.add_argument("--provider", default="local",
                         choices=["local", "openai", "voyage"])
    p_index.add_argument("--dim", type=int, default=512,
                        help="vector dimension for the local embedder")
    p_index.set_defaults(func=_cmd_index)

    p_search = sub.add_parser("search", help="query an existing index")
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("--index", default=".", help="repo root holding the index")
    p_search.add_argument("-k", type=int, default=5, help="number of results")
    p_search.add_argument("--lines", type=int, default=6,
                         help="snippet lines to show per hit")
    p_search.set_defaults(func=_cmd_search)

    p_ask = sub.add_parser("ask", help="answer a question with a local LLM (RAG)")
    p_ask.add_argument("query", help="natural-language question")
    p_ask.add_argument("--index", default=".", help="repo root holding the index")
    p_ask.add_argument("-k", type=int, default=6,
                       help="number of chunks to feed the LLM as context")
    p_ask.add_argument("--model", default=DEFAULT_MODEL,
                       help="local Ollama model (or set OLLAMA_MODEL)")
    p_ask.add_argument("--host", default=DEFAULT_HOST,
                       help="Ollama host URL (or set OLLAMA_HOST)")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"\033[31merror:\033[0m {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
