"""Batch orchestrator for running the literature researcher on the full
curated corpus (Week 1 Block 4.3).

Per-paper failures are caught and logged so a single bad paper does not
crash the whole run. The runner returns a CorpusRunResult containing:
  - successful extractions keyed by paper_id
  - failures keyed by paper_id with the error message
  - cumulative cost and timing estimates

CLI:
    python -m agentic_ai.agents.corpus_runner --dry-run
    python -m agentic_ai.agents.corpus_runner --paper elsevier_2001-0370_2025
    python -m agentic_ai.agents.corpus_runner
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.agents.extractor import extract_from_paper
from agentic_ai.loaders.text_reader import load_curated_corpus

# Rough cost estimate constants tuned for gpt-4o-mini in May 2026.
# Input: $0.15 / 1M tokens. Output: $0.60 / 1M tokens.
# Average chars-per-token across our corpus is ~3.5.
_USD_PER_INPUT_TOKEN = 0.15 / 1_000_000
_USD_PER_OUTPUT_TOKEN = 0.60 / 1_000_000
_CHARS_PER_TOKEN = 3.5

# Empirical estimate from Block 4.2 single-paper run.
_ESTIMATED_OUTPUT_TOKENS_PER_PAPER = 1500


@dataclass
class CorpusRunResult:
    """
    Bundles the outcome of one corpus-wide extraction run.
    @param successful: PaperExtraction objects keyed by paper_id.
    @param failures: Error messages keyed by paper_id (one entry per
                     paper that raised during extraction).
    @param elapsed_seconds: Wall-clock time of the full run.
    @param estimated_cost_usd: Rough USD estimate based on character
                               counts; actual billing may differ by 2x.
    """

    successful: Dict[str, PaperExtraction] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0


def estimate_corpus_cost(
    corpus: Dict[str, str] = None,
) -> float:
    """
    Estimates the USD cost of extracting from every paper in the corpus,
    based on input character counts plus a fixed estimate of output
    tokens per paper.
    @param corpus: Dict mapping paper_id to paper text.
    return : Estimated USD cost as a float.
    """
    if corpus is None:
        return 0.0

    total_input_chars = sum(len(text) for text in corpus.values())
    total_input_tokens = total_input_chars / _CHARS_PER_TOKEN
    total_output_tokens = (
        _ESTIMATED_OUTPUT_TOKENS_PER_PAPER * len(corpus)
    )

    input_cost = total_input_tokens * _USD_PER_INPUT_TOKEN
    output_cost = total_output_tokens * _USD_PER_OUTPUT_TOKEN

    return round(input_cost + output_cost, 4)


def run_corpus(
    paper_ids: Optional[List[str]] = None,
    corpus: Optional[Dict[str, str]] = None,
) -> CorpusRunResult:
    """
    Runs the literature researcher against every paper in the corpus
    (or a filtered subset), catching per-paper errors so one failure
    does not abort the rest.
    @param paper_ids: Optional list of paper_ids to restrict the run.
                      Useful for retrying a single paper after a fix.
                      None means run every paper in the corpus.
    @param corpus: Optional pre-loaded corpus dict. If None, the curated
                   corpus is loaded from disk.
    return : A CorpusRunResult bundling successes, failures, timing,
             and a cost estimate.
    """
    if corpus is None:
        corpus = load_curated_corpus()

    if paper_ids is not None:
        corpus = {
            pid: text
            for pid, text in corpus.items()
            if pid in paper_ids
        }

    result = CorpusRunResult(
        estimated_cost_usd=estimate_corpus_cost(corpus),
    )

    start = time.time()

    for paper_id, paper_text in corpus.items():
        print(
            f"[{paper_id}] running ({len(paper_text):,} chars)... ",
            end="",
            flush=True,
        )

        per_paper_start = time.time()
        try:
            extraction = extract_from_paper(
                paper_text=paper_text,
                paper_id=paper_id,
            )
            result.successful[paper_id] = extraction
            elapsed = time.time() - per_paper_start
            print(
                f"ok ({len(extraction.variants)} variants, "
                f"{len(extraction.measurements)} measurements, "
                f"{elapsed:.1f}s)"
            )
        except Exception as exc:
            elapsed = time.time() - per_paper_start
            error_message = f"{type(exc).__name__}: {exc}"
            result.failures[paper_id] = error_message
            print(f"FAILED after {elapsed:.1f}s")
            print(f"   {error_message}")

    result.elapsed_seconds = time.time() - start

    return result


def _print_summary(result: CorpusRunResult) -> None:
    """
    Prints a compact summary of a corpus run to stdout.
    @param result: The CorpusRunResult returned by run_corpus().
    """
    total = len(result.successful) + len(result.failures)
    total_variants = sum(
        len(extraction.variants)
        for extraction in result.successful.values()
    )
    total_measurements = sum(
        len(extraction.measurements)
        for extraction in result.successful.values()
    )

    print()
    print("=" * 60)
    print(f"Corpus run summary")
    print("=" * 60)
    print(f"Papers processed:    {total}")
    print(f"  Successful:        {len(result.successful)}")
    print(f"  Failed:            {len(result.failures)}")
    print(f"Records extracted:   "
          f"{total_variants} variants, {total_measurements} measurements")
    print(f"Wall-clock time:     {result.elapsed_seconds:.1f}s")
    print(f"Estimated cost:      ${result.estimated_cost_usd:.4f}")

    if result.failures:
        print()
        print("Failed papers:")
        for paper_id, error in result.failures.items():
            print(f"  - {paper_id}: {error}")


def main() -> int:
    """
    CLI entry point. Supports dry-run cost estimation, single-paper
    retry, and full corpus runs.
    return : Shell exit code (0 if all papers succeeded, 1 otherwise).
    """
    parser = argparse.ArgumentParser(
        description="Run the literature researcher on the curated corpus."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cost estimate without making API calls.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist each PaperExtraction as JSON in "
             "data/processed/extractions/ after the run completes.",
    )
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Run only this paper_id (filename stem) instead of all papers.",
    )
    args = parser.parse_args()

    try:
        corpus = load_curated_corpus()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.paper:
        if args.paper not in corpus:
            print(
                f"[error] paper_id {args.paper!r} not found. "
                f"Available: {sorted(corpus.keys())[:5]}...",
                file=sys.stderr,
            )
            return 1
        corpus = {args.paper: corpus[args.paper]}

    if args.dry_run:
        cost = estimate_corpus_cost(corpus)
        total_chars = sum(len(t) for t in corpus.values())
        print(
            f"Dry run: {len(corpus)} papers, "
            f"{total_chars:,} chars total, "
            f"~${cost:.4f} estimated cost. "
            f"No API calls made."
        )
        return 0

    result = run_corpus(corpus=corpus)
    _print_summary(result)

    if args.save and result.successful:
        # Import here to avoid loading json plumbing on dry runs
        from agentic_ai.loaders.extraction_io import save_extractions
        written = save_extractions(result.successful)
        print(f"\nPersisted {written} extraction(s) to "
              f"data/processed/extractions/")

    return 0 if not result.failures else 1


if __name__ == "__main__":
    sys.exit(main())
