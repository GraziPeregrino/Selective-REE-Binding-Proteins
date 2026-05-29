"""Persistence layer for PaperExtraction records (Week 1 Block 4.3.5).

Saves each PaperExtraction as a single JSON file under
data/processed/extractions/, named by paper_id. Loading reads the
directory back into memory as a dict[paper_id, PaperExtraction].

This persistence layer is what turns the in-memory CrewAI output into a
reproducible dataset: once persisted, downstream code (Block 5 merge,
Week 2 feature engineering, Week 3 ML training) can iterate without
re-running the agent and re-paying for the API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from agentic_ai.agents.extraction_models import PaperExtraction

# Canonical location for persisted agent extractions. Committed to git
# since each file is small and reproducibility matters more than disk.
_DEFAULT_EXTRACTIONS_DIR = Path("data/processed/extractions")


def save_extractions(
    extractions: Dict[str, PaperExtraction] = None,
    output_dir: Path = None,
) -> int:
    """
    Writes each PaperExtraction to a JSON file named '<paper_id>.json'.
    Creates the output directory if it does not exist.
    @param extractions: Dict mapping paper_id to PaperExtraction (as
                        returned by CorpusRunResult.successful).
    @param output_dir: Directory to write JSON files to. Defaults to
                       data/processed/extractions.
    return : Number of files written.
    """
    if extractions is None or not extractions:
        return 0

    if output_dir is None:
        output_dir = _DEFAULT_EXTRACTIONS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for paper_id, extraction in extractions.items():
        path = output_dir / f"{paper_id}.json"
        path.write_text(
            extraction.model_dump_json(indent=2),
            encoding="utf-8",
        )
        written += 1

    return written


def load_extractions(
    input_dir: Path = None,
) -> Dict[str, PaperExtraction]:
    """
    Reads every JSON file in the extractions directory back into a dict
    of validated PaperExtraction objects. Schema validation on load is
    a free safety net: if any persisted file violates the current
    schema, Pydantic will raise before any downstream code touches it.
    @param input_dir: Directory to read from. Defaults to
                      data/processed/extractions.
    return : Dict mapping paper_id (filename stem) to PaperExtraction.
    """
    if input_dir is None:
        input_dir = _DEFAULT_EXTRACTIONS_DIR

    if not input_dir.exists():
        return {}

    extractions: Dict[str, PaperExtraction] = {}

    for path in sorted(input_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        extractions[path.stem] = PaperExtraction.model_validate_json(text)

    return extractions
