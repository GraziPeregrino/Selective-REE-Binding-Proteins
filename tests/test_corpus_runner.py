"""Tests for the corpus orchestrator (Week 1 Block 4.5).

Covers the surface area that doesn't require a live LLM call:
  - cost estimation
  - paper filtering
  - empty corpus handling
  - summary printing for success and failure cases

run_corpus() with a non-empty real corpus would invoke the OpenAI
API; that path is covered by manual end-to-end runs documented in the
project log. These tests pin the orchestration logic only.
"""
from __future__ import annotations

from agentic_ai.agents.corpus_runner import (
    CorpusRunResult,
    _print_summary,
    estimate_corpus_cost,
    run_corpus,
)
from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.schemas import BindingMeasurement, ProteinVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_extraction_with_counts(
    n_variants: int = 0,
    n_measurements: int = 0,
    source_paper: str = "fake_paper",
) -> PaperExtraction:
    """
    Builds a PaperExtraction with the requested number of variants and
    measurements. Used to populate fake CorpusRunResults for summary
    testing.
    """
    variants = [
        ProteinVariant(
            variant_id=f"o-{i}",
            source_organism="Test organism",
            source_paper=source_paper,
        )
        for i in range(n_variants)
    ]
    measurements = [
        BindingMeasurement(
            variant_id=f"o-{i % max(n_variants, 1)}",
            target_element="Neodymium",
            measurement_type="Kd",
            value=1.0e-9,
            units="M",
            source_paper=source_paper,
        )
        for i in range(n_measurements)
    ]
    return PaperExtraction(variants=variants, measurements=measurements)


# ---------------------------------------------------------------------------
# estimate_corpus_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_returns_zero_for_empty_corpus():
    """
    Verifies that an empty corpus has zero estimated cost. Prevents
    division-by-zero and other edge-case surprises.
    """
    assert estimate_corpus_cost({}) == 0.0


def test_estimate_cost_returns_zero_for_none_input():
    """
    Verifies that None input is handled gracefully (returns 0.0).
    """
    assert estimate_corpus_cost(None) == 0.0


def test_estimate_cost_scales_with_text_length():
    """
    Verifies that the cost estimate increases when the corpus text
    grows. Concrete number not pinned (the multipliers may change with
    pricing), just the directional monotonicity.
    """
    small = {"paper_a": "x" * 1000}
    large = {"paper_a": "x" * 100_000}
    assert estimate_corpus_cost(large) > estimate_corpus_cost(small)


def test_estimate_cost_scales_with_paper_count():
    """
    Verifies that doubling the number of papers (with same text per
    paper) at least increases cost, because output-token cost is per
    paper.
    """
    one_paper = {"paper_a": "x" * 5000}
    two_papers = {"paper_a": "x" * 5000, "paper_b": "x" * 5000}
    assert estimate_corpus_cost(two_papers) > estimate_corpus_cost(one_paper)


def test_estimate_cost_is_in_realistic_range_for_full_corpus_scale():
    """
    Sanity-check: a 15-paper corpus of ~8000 chars each should land in
    a few cents, not a few dollars. Catches accidentally swapping cost
    factors (e.g. confusing $0.15 / 1M with $0.15 / 1K).
    """
    corpus = {f"paper_{i}": "x" * 8000 for i in range(15)}
    cost = estimate_corpus_cost(corpus)
    assert 0.001 < cost < 0.50


# ---------------------------------------------------------------------------
# run_corpus
# ---------------------------------------------------------------------------

def test_run_corpus_with_empty_corpus_returns_empty_result():
    """
    Verifies that run_corpus on an empty dict short-circuits cleanly
    without invoking any LLM call. Exercises the orchestration logic
    in isolation.
    """
    result = run_corpus(corpus={})

    assert isinstance(result, CorpusRunResult)
    assert result.successful == {}
    assert result.failures == {}
    assert result.estimated_cost_usd == 0.0
    assert result.elapsed_seconds >= 0.0


def test_run_corpus_filters_to_specified_paper_ids():
    """
    Verifies that paper_ids filtering correctly narrows an input
    corpus before iteration. Combined with empty-iteration semantics,
    this lets us test the filter without an LLM call.
    """
    corpus = {
        "paper_a": "text a",
        "paper_b": "text b",
        "paper_c": "text c",
    }

    # Pick zero matching IDs — corpus becomes empty after filtering,
    # so run_corpus completes without calling the agent.
    result = run_corpus(paper_ids=["nonexistent"], corpus=corpus)

    assert result.successful == {}
    assert result.failures == {}


def test_run_corpus_filter_preserves_cost_estimate_for_filtered_subset():
    """
    Verifies that the cost estimate reflects the filtered corpus, not
    the input corpus. Critical because users running --paper expect a
    single-paper cost, not the full-corpus cost.
    """
    full_corpus = {
        "paper_a": "x" * 10000,
        "paper_b": "x" * 10000,
        "paper_c": "x" * 10000,
    }
    full_cost = estimate_corpus_cost(full_corpus)

    # Filter to an empty subset; cost should be near zero.
    result = run_corpus(paper_ids=["does_not_exist"], corpus=full_corpus)

    assert result.estimated_cost_usd < full_cost
    assert result.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# _print_summary
# ---------------------------------------------------------------------------

def test_print_summary_handles_empty_result(capsys):
    """
    Verifies that _print_summary doesn't crash on an empty result and
    prints the expected zero-count framing.
    """
    result = CorpusRunResult()
    _print_summary(result)

    captured = capsys.readouterr()
    assert "Corpus run summary" in captured.out
    assert "Papers processed:    0" in captured.out
    assert "Successful:        0" in captured.out
    assert "Failed:            0" in captured.out


def test_print_summary_reports_success_counts(capsys):
    """
    Verifies that the summary correctly aggregates variant and
    measurement counts across multiple successful papers.
    """
    result = CorpusRunResult(
        successful={
            "paper_a": _make_extraction_with_counts(
                n_variants=3, n_measurements=10, source_paper="paper_a",
            ),
            "paper_b": _make_extraction_with_counts(
                n_variants=2, n_measurements=5, source_paper="paper_b",
            ),
        },
        elapsed_seconds=42.5,
        estimated_cost_usd=0.0123,
    )

    _print_summary(result)
    captured = capsys.readouterr()

    assert "Papers processed:    2" in captured.out
    assert "Successful:        2" in captured.out
    assert "5 variants" in captured.out  # 3 + 2
    assert "15 measurements" in captured.out  # 10 + 5
    assert "42.5s" in captured.out
    assert "$0.0123" in captured.out


def test_print_summary_reports_failures(capsys):
    """
    Verifies that the summary lists failed papers with their error
    messages so users can diagnose retry-able failures.
    """
    result = CorpusRunResult(
        successful={"good_paper": _make_extraction_with_counts(
            n_variants=1, n_measurements=2, source_paper="good_paper",
        )},
        failures={
            "bad_paper": "ValidationError: invalid target_element",
        },
        elapsed_seconds=10.0,
    )

    _print_summary(result)
    captured = capsys.readouterr()

    assert "Successful:        1" in captured.out
    assert "Failed:            1" in captured.out
    assert "Failed papers:" in captured.out
    assert "bad_paper" in captured.out
    assert "ValidationError" in captured.out


# ---------------------------------------------------------------------------
# CorpusRunResult
# ---------------------------------------------------------------------------

def test_corpus_run_result_default_construction():
    """
    Verifies that CorpusRunResult constructs cleanly with no args and
    has the documented field defaults.
    """
    result = CorpusRunResult()

    assert result.successful == {}
    assert result.failures == {}
    assert result.elapsed_seconds == 0.0
    assert result.estimated_cost_usd == 0.0
