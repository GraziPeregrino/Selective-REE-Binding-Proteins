"""Tests for the curated literature corpus reader (Week 1 Block 3.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentic_ai.loaders.text_reader import (
    list_curated_files,
    load_curated_corpus,
)

# Path to the real corpus. Tests skip when it isn't present so a fresh
# clone can still pass without manual setup.
_CORPUS_DIR = Path("agentic_ai/inputs/processed")

# Skip everything in this module when the corpus is missing.
pytestmark = pytest.mark.skipif(
    not _CORPUS_DIR.exists()
    or not any(_CORPUS_DIR.glob("*.txt")),
    reason=f"No curated corpus at {_CORPUS_DIR}. Run Gemini curation first.",
)


def test_load_curated_corpus_returns_dict_of_strings():
    """
    Verifies that the loader returns a dict mapping filenames to text
    contents, both as strings.
    """
    corpus = load_curated_corpus()

    assert isinstance(corpus, dict)
    assert all(isinstance(key, str) for key in corpus.keys())
    assert all(isinstance(value, str) for value in corpus.values())


def test_load_curated_corpus_holds_expected_paper_count():
    """
    Verifies the corpus loads the full set of 15 Gemini-curated papers.
    """
    corpus = load_curated_corpus()

    assert len(corpus) == 15


def test_load_curated_corpus_skips_non_paper_files(tmp_path):
    """
    Verifies that loader ignores .gitkeep, README.md, and dotfiles even
    when they exist alongside the .txt extracts.
    """
    (tmp_path / "paper_one.txt").write_text("real paper content")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "README.md").write_text("project notes")
    (tmp_path / ".DS_Store").write_text("os metadata")

    corpus = load_curated_corpus(corpus_dir=tmp_path)

    assert list(corpus.keys()) == ["paper_one"]


def test_load_curated_corpus_skips_empty_txt_files(tmp_path):
    """
    Verifies that empty .txt files are excluded so they don't trigger
    no-op CrewAI calls in Block 4.
    """
    (tmp_path / "real_paper.txt").write_text("content")
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "whitespace_only.txt").write_text("   \n\n  ")

    corpus = load_curated_corpus(corpus_dir=tmp_path)

    assert "real_paper" in corpus
    assert "empty" not in corpus
    assert "whitespace_only" not in corpus


def test_load_curated_corpus_raises_when_directory_missing(tmp_path):
    """
    Verifies the loader gives a clear error when the corpus directory
    doesn't exist, pointing the user to the curation step.
    """
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Gemini curation"):
        load_curated_corpus(corpus_dir=missing_dir)


def test_load_curated_corpus_raises_when_no_txt_files(tmp_path):
    """
    Verifies that an empty directory raises ValueError rather than
    silently returning an empty corpus.
    """
    (tmp_path / "image.png").write_text("not a paper")
    (tmp_path / "data.csv").write_text("col1,col2")

    with pytest.raises(ValueError, match="No usable .txt files"):
        load_curated_corpus(corpus_dir=tmp_path)


def test_list_curated_files_matches_loader_keys():
    """
    Verifies that list_curated_files() returns the same keyset that
    load_curated_corpus() would, allowing cheap previews.
    """
    corpus_keys = set(load_curated_corpus().keys())
    listed = set(list_curated_files())

    assert corpus_keys == listed


def test_list_curated_files_returns_empty_for_missing_dir(tmp_path):
    """
    Verifies the listing function returns an empty list (not an error)
    when the corpus dir is missing — convenient for status displays.
    """
    missing = tmp_path / "absent"

    assert list_curated_files(corpus_dir=missing) == []
