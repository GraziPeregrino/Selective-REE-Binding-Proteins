"""Text reader for the Gemini-curated literature corpus (Week 1 Block 3.3).

The CrewAI agent in Block 4 will iterate over the corpus produced by this
module, extracting ProteinVariant and BindingMeasurement annotations from
each paper's relevant passages. The curation step (Gemini 2.5 Pro,
performed manually) lives upstream; this module only handles file I/O.

Run as a script for a quick smoke test:
    python -m agentic_ai.loaders.text_reader
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Filenames in inputs/processed/ that are NOT scientific papers and should
# be skipped (documentation, build artifacts, OS metadata).
_NON_PAPER_FILES = {
    ".gitkeep",
    ".DS_Store",
    "README.md",
    "PROMPT.md",
}

# Default location of the curated text corpus inside the project.
_DEFAULT_CORPUS_DIR = Path("agentic_ai/inputs/processed")


def load_curated_corpus(
    corpus_dir: Path = None,
) -> Dict[str, str]:
    """
    Loads every curated .txt file in the corpus directory into a dict keyed
    by filename (without extension). Skips non-paper files like .gitkeep
    and README.md.
    @param corpus_dir: Path to the directory holding curated .txt files.
                       Defaults to agentic_ai/inputs/processed.
    return : Dict mapping filename (no extension) to file contents (str).
    raises : FileNotFoundError if the corpus directory does not exist.
    raises : ValueError if the directory contains no .txt files at all.
    """
    if corpus_dir is None:
        corpus_dir = _DEFAULT_CORPUS_DIR

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Curated corpus directory not found at {corpus_dir}. "
            f"Run the Gemini curation step on PDFs in agentic_ai/inputs/ "
            f"and save outputs to {corpus_dir}/"
        )

    corpus: Dict[str, str] = {}

    for path in sorted(corpus_dir.iterdir()):
        if path.name in _NON_PAPER_FILES or path.name.startswith("."):
            continue
        if path.suffix.lower() != ".txt":
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        corpus[path.stem] = text

    if not corpus:
        raise ValueError(
            f"No usable .txt files found in {corpus_dir}. "
            f"Expected Gemini-curated paper extracts."
        )

    return corpus


def list_curated_files(corpus_dir: Path = None) -> List[str]:
    """
    Lists the filenames (without extension) of every curated paper in the
    corpus directory. Useful for previewing the corpus without reading
    every file into memory.
    @param corpus_dir: Path to the directory holding curated .txt files.
                       Defaults to agentic_ai/inputs/processed.
    return : Sorted list of filenames without the .txt extension.
    """
    if corpus_dir is None:
        corpus_dir = _DEFAULT_CORPUS_DIR

    if not corpus_dir.exists():
        return []

    return sorted(
        path.stem
        for path in corpus_dir.iterdir()
        if path.suffix.lower() == ".txt"
        and path.name not in _NON_PAPER_FILES
        and not path.name.startswith(".")
    )


def main() -> int:
    """
    Loads the curated corpus and prints a summary table. Used as a smoke
    test.
    return : Shell exit code (0 on success, 1 on error).
    """
    try:
        corpus = load_curated_corpus()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    total_chars = sum(len(text) for text in corpus.values())

    print(
        f"Loaded curated corpus: "
        f"{len(corpus)} papers, "
        f"{total_chars:,} total characters"
    )

    print()
    print(f"  {'Paper':<45} {'Chars':>8}")
    print(f"  {'-' * 45} {'-' * 8}")
    for filename in sorted(corpus.keys()):
        char_count = len(corpus[filename])
        print(f"  {filename:<45} {char_count:>8,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
