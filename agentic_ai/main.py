"""Entry point for the REE-binding agentic pipeline.

Runs the full Block 1-3 smoke test: verifies OpenAI connectivity, loads
the MOESM3 master dataset, loads the curated literature corpus, and
prints a unified status summary including a sample variant from the
Diep et al. 2026 dataset.

Block 4 will extend this entry point to invoke the CrewAI annotation
agent against the loaded text corpus.
"""
from __future__ import annotations

import sys
from typing import Optional

from agentic_ai.loaders.text_reader import load_curated_corpus
from agentic_ai.loaders.xlsx_loader import load_moesm3
from agentic_ai.schemas import CorpusRecords
from agentic_ai.utils.env_check import load_api_key, ping_openai


def _check_openai_connectivity() -> Optional[str]:
    """
    Verifies that the OpenAI API key is configured and reachable.
    @return : None on success, error message string on failure.
    """
    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        return f"config error: {exc}"

    try:
        reply = ping_openai(api_key)
    except Exception as exc:
        return f"api error: {type(exc).__name__}: {exc}"

    return None if reply else "api error: empty response"


def _print_dataset_summary(corpus: CorpusRecords) -> None:
    """
    Prints a compact summary of the loaded MOESM3 corpus and one
    canonical sample variant to confirm the loader is producing the
    right records.
    @param corpus: The CorpusRecords object returned by load_moesm3.
    """
    print(
        f"MOESM3:              "
        f"{len(corpus.variants)} variants, "
        f"{len(corpus.measurements)} measurements"
    )

    sample = next(
        (v for v in corpus.variants if v.variant_id == "o-621"),
        None,
    )
    if sample is None:
        return

    print()
    print(f"Sample variant: {sample.variant_id} (Mex-LanM)")
    print(f"  organism: {sample.source_organism}")
    print(f"  cluster:  {sample.selectivity_cluster}")
    print(f"  sequence: {sample.sequence[:40]}... "
          f"({len(sample.sequence)} residues)")
    print(f"  sample measurements:")

    interesting_elements = {"Lanthanum", "Praseodymium", "Neodymium", "Lutetium"}
    sample_measurements = [
        m for m in corpus.measurements
        if m.variant_id == sample.variant_id
        and m.target_element in interesting_elements
    ]
    for measurement in sample_measurements:
        print(
            f"    {measurement.target_element:<15} "
            f"{measurement.measurement_type} = {measurement.value:.3f}"
        )


def main() -> int:
    """
    Runs the Block 1-3 pipeline smoke test end-to-end.
    return : Shell exit code (0 on full success, 1 on any failure).
    """
    print("=== REE-Binding Pipeline — Block 3 status ===")
    print()

    # --- OpenAI connectivity ----------------------------------------------
    openai_error = _check_openai_connectivity()
    if openai_error:
        print(f"OpenAI connectivity: FAIL ({openai_error})", file=sys.stderr)
        return 1
    print("OpenAI connectivity: ok")

    # --- MOESM3 master dataset --------------------------------------------
    try:
        corpus = load_moesm3()
    except (FileNotFoundError, ValueError) as exc:
        print(f"MOESM3:              FAIL ({exc})", file=sys.stderr)
        return 1
    _print_dataset_summary(corpus)

    # --- Curated literature corpus ----------------------------------------
    print()
    try:
        text_corpus = load_curated_corpus()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Literature corpus:   FAIL ({exc})", file=sys.stderr)
        return 1

    total_chars = sum(len(text) for text in text_corpus.values())
    print(
        f"Literature corpus:   "
        f"{len(text_corpus)} papers, "
        f"{total_chars:,} chars total"
    )

    print()
    print("All Block 3 components healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())