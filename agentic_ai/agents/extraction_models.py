"""Pydantic models that wrap the literature-extraction agent's output.

CrewAI's `output_pydantic` expects a single BaseModel as the return type,
not a list. These wrapper models package collections of ProteinVariant and
BindingMeasurement records in a single object the agent can fill in.

Records produced here flow into the unified CorpusRecords assembled in
Block 5, where they are merged with the MOESM3 records loaded by
agentic_ai.loaders.xlsx_loader.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from agentic_ai.schemas import BindingMeasurement, ProteinVariant


class PaperExtraction(BaseModel):
    """
    Output contract for the literature researcher agent. One instance
    represents everything the agent found in a single paper.
    @param variants: Protein variants mentioned in the paper with enough
                     detail to construct a ProteinVariant record.
    @param measurements: Binding measurements (Kd, Kd_app, EC50, etc.)
                         extracted from the paper, each tied to a variant.
    @param notes: Free-text notes the agent wants to log about the
                  extraction (e.g. "paper is a news commentary, no
                  primary measurements"). Used for downstream auditing.
    """

    variants: List[ProteinVariant] = Field(default_factory=list)
    measurements: List[BindingMeasurement] = Field(default_factory=list)
    notes: str = Field(default="")
