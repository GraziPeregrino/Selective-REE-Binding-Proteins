"""Orchestrator for running the literature researcher across the corpus.

This module exposes one function -- extract_from_paper() -- that takes a
single paper's text and returns a PaperExtraction. Block 4.3 will add a
batch function that loops over the full corpus.

After the agent emits raw measurements, this module deterministically
populates each measurement's value_in_molar field using
agentic_ai.loaders.unit_conversion. The agent itself is forbidden
(via the task prompt) from touching that field, which eliminated a
class of 1000x conversion errors we observed in Block 4.2.

Costs ~$0.005 per paper on gpt-4o-mini for typical corpus sizes (5-13KB
of curated text).
"""
from __future__ import annotations

from agentic_ai.agents.extraction_models import PaperExtraction
from agentic_ai.agents.literature_researcher import build_extraction_crew
from agentic_ai.loaders.unit_conversion import to_molar
from agentic_ai.schemas import BindingMeasurement


def extract_from_paper(
    paper_text: str = None,
    paper_id: str = None,
) -> PaperExtraction:
    """
    Runs the Literature Researcher agent against a single paper, then
    deterministically normalizes every measurement's value_in_molar.
    @param paper_text: The full curated text of one paper.
    @param paper_id: Stable identifier used in the source_paper field of
                     produced records (e.g. the .txt filename stem).
    return : A PaperExtraction containing variants and measurements,
             with value_in_molar computed from value + units in Python
             rather than by the LLM.
    raises : ValueError if paper_text or paper_id is empty.
    raises : RuntimeError if the Crew returns no Pydantic output.
    """
    crew = build_extraction_crew(
        paper_text=paper_text,
        paper_id=paper_id,
    )

    result = crew.kickoff()

    extraction = getattr(result, "pydantic", None)
    if extraction is None:
        raise RuntimeError(
            f"Crew returned no Pydantic output. Raw result: {result!r}"
        )

    extraction.measurements = [
        _normalize_measurement(measurement)
        for measurement in extraction.measurements
    ]

    return extraction


def _normalize_measurement(
    measurement: BindingMeasurement = None,
) -> BindingMeasurement:
    """
    Returns a new BindingMeasurement with value_in_molar computed from
    value + units via the deterministic converter. The original
    measurement is not mutated.
    @param measurement: The agent-produced BindingMeasurement.
    return : A new BindingMeasurement instance with normalized
             value_in_molar.
    """
    normalized_molar = to_molar(
        value=measurement.value,
        units=measurement.units,
    )

    return measurement.model_copy(update={"value_in_molar": normalized_molar})