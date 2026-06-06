# askRepo — semantic code search

Ask your codebase questions in plain English and get back the exact functions
that answer them — not keyword matches, not whole files, but the right *symbol*
with precise line numbers.

```
$ askrepo index .
Indexed 190 chunks from 42 files (provider=local-hashing(dim=512), dim=512)

$ askrepo search "verify a signature and reject tampered data"

1. src/itsdangerous/signer.py:227-242   method Signer.verify_signature   0.188
   | def verify_signature(self, value, sig) -> bool:
   |     """Verifies the signature for the given value."""
   |     ...
2. src/itsdangerous/signer.py:24-28      method SigningAlgorithm.verify_signature
3. src/itsdangerous/exc.py:22-33         class BadSignature
```

No keyword in that query appears in `verify_signature`'s body — it's matched by
meaning.

## Why this is harder than it looks

The naive version ("embed every 50 lines, do a vector search") fails for two
reasons this project addresses head-on:

1. **Chunking destroys meaning.** Fixed-size windows slice functions in half,
   so a hit points you at a fragment with no boundaries. `askrepo` parses
   **Python with the stdlib `ast` module** to extract each function, method, and
   class as one chunk, with its qualified name (`Class.method`), docstring, and
   exact line span. Other languages use a structural-declaration heuristic with
   an overlapping-window fallback so nothing is ever lost. (Drop in tree-sitter
   to get Python-grade precision everywhere — the `Chunk` interface doesn't
   change.)

2. **Embeddings are a swappable layer, not the product.** The pipeline is
   provider-agnostic. The default `local` provider is a **zero-dependency hashed
   TF-IDF** vectoriser with code-aware tokenisation: it splits `getUserById` into
   `get user by id` and `parse_ast_node` into `parse ast node`, so natural
   language lines up with identifiers. It uses a deterministic FNV-1a hash
   (Python's built-in `hash()` is salted per process and would scatter
   index- and query-time tokens into different buckets — a bug worth knowing
   about). For true semantic quality, flip one flag to a neural provider.

## Architecture

```
askrepo/
  chunker.py     AST-aware chunking (ast for Python, heuristics elsewhere)
  embeddings.py  Embedder interface + Local / OpenAI / Voyage providers
  index.py       walk repo -> chunk -> embed -> persist vectors + metadata
  search.py      embed query -> cosine similarity -> ranked hits
  cli.py         `index` and `search` subcommands
```

The index lives in `<repo>/.askrepo_index/` as a numpy vector matrix plus
JSON metadata — inspectable, diff-able, no database to run.

## Install & use

```bash
pip install -r requirements.txt          # just numpy for the local provider

python -m askrepo index /path/to/repo
python -m askrepo search "where do we validate webhook payloads" --index /path/to/repo -k 5
```

### Point it at a GitHub repo

`index`, `search`, and `ask` accept a GitHub URL anywhere they take a path. The
repo is shallow-cloned once into `~/.askrepo/repos/` and reused; pass `--refresh`
to re-pull.

```bash
python -m askrepo index  https://github.com/pallets/flask
python -m askrepo ask "how does routing map a URL to a view?" --index https://github.com/pallets/flask
```

## Real semantic embeddings

The `local` provider is lexical (great out of the box, no setup). For genuine
semantic retrieval, switch provider — same pipeline, same commands:

```bash
export VOYAGE_API_KEY=...     # voyage-code-3 is purpose-built for code
python -m askrepo index /path/to/repo --provider voyage
```

`OPENAI_API_KEY` + `--provider openai` works identically. Adding a provider is
~15 lines: subclass `Embedder`, implement `encode()`.

## Honest limitations

- The `local` embedder is lexical + subword, not neural — it won't connect
  truly unrelated wording to concepts the way `voyage`/`openai` do. It's a
  strong baseline, not a replacement for real embeddings.
- Non-Python chunking is heuristic. Wire up tree-sitter for parser-grade spans.
- Brute-force cosine over all vectors is fine to tens of thousands of chunks;
  past that, add an ANN index (FAISS / hnswlib) behind the same `search()`.

## Where this goes as a product

The retrieval core here is the hard, defensible part. Natural extensions: an
incremental indexer that only re-embeds changed files on git commit, an
"ask"-mode that feeds the top hits to an LLM for a cited answer, an editor
extension, and a hosted tier for teams onboarding to large/legacy repos — the
niche where keyword search hurts most and people pay to fix it.
