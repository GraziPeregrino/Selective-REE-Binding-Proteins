"""CrewAI agent + task for literature annotation extraction (Week 1 Block 4.1).

This module defines the 'Literature Researcher' agent and the per-paper
'Extract Variants and Measurements' task. The agent reads one curated
paper at a time and emits a PaperExtraction record listing every LanM
variant it found along with any published binding constants.

The orchestrator (Block 4.3) is responsible for iterating over the
15-paper corpus and calling this agent once per paper.
"""
from __future__ import annotations
from crewai import Agent, Crew, Process, Task
from agentic_ai.agents.extraction_models import PaperExtraction

# LLM identifier passed to CrewAI. The crewai package reads OPENAI_API_KEY
# from the environment automatically (we already load it via python-dotenv
# in agentic_ai.utils.env_check).
_LITERATURE_LLM = "gpt-4o-mini"

# Reuseable strings to keep the agent persona consistent across calls.
_AGENT_ROLE = "Computational biology literature researcher"

_AGENT_GOAL = (
    "Extract verifiable LanM protein variant annotations and published "
    "binding constants from curated scientific papers, with strict "
    "fidelity to the source text and zero fabrication."
)

_AGENT_BACKSTORY = (
    "You have a decade of experience analyzing lanthanide-binding "
    "proteins, with deep familiarity with the Lanmodulin family "
    "(Mex-LanM, Hans-LanM, Melba-LanM, and engineered variants such as "
    "R100K, 4P2A, 4D9X). You read papers slowly and only report "
    "values that are explicitly stated in the text. You refuse to "
    "interpolate, guess, or convert units when the original measurement "
    "is ambiguous. When the same variant appears under multiple names "
    "(e.g. 'o-621' and 'Mex-LanM'), you report both."
)

# Task description fed to the agent for each paper. The {paper_text}
# placeholder is filled in by the orchestrator at call time.
_TASK_DESCRIPTION_TEMPLATE = """
Read the following scientific paper excerpt and extract every LanM
protein variant and published binding measurement you can find.

PAPER EXCERPT
=============
{paper_text}
=============

EXTRACTION RULES
1. One variant per record. When a paper uses shorthand notation for a
   group of variants (e.g. '4D9X' = '4D9N', '4D9A', '4D9M', '4D9H'),
   emit one ProteinVariant record per concrete variant, NOT one
   collective record. Every variant_id that appears in any measurement
   must also exist as a variant record.
2. variant_id must be the most specific concrete identifier. Use
   '4D9H', not '4D9X'. Use 'Hans-LanM(R100K)', not 'Hans-LanM
   variants'. Use 'o-36' or 'Melba-LanM' (whichever the paper uses
   more), not 'C5 cluster members'.
3. Measurements: include every numerical binding constant (Kd, Kd_app,
   EC50, Kd_dimer, etc.) with its element and units. Preserve scientific
   notation exactly as printed; a value reported as '2.4e-12 M' must be
   stored as the float 2.4e-12, NOT 2.4.
4. IMPORTANT: Do NOT perform any unit conversion. Leave the
   `value_in_molar` field set to null for every measurement. A
   deterministic Python converter will compute it post hoc.
5. source_organism should be a plain scientific name, no FASTA-style
   headers, accession numbers, or '>' prefixes. Example: write
   'Methylorubrum extorquens', not '>Methylorubrum extorquens
   [WP_...]'.
6. Use 'cited_from_earlier_work' as the value_source_type whenever the
   paper attributes the number to a different publication; use
   'primary' only for measurements the paper itself reports.
7. If the paper does not state a full continuous amino acid sequence
   for a variant, leave sequence as null rather than fabricating one.
8. Use source_paper = '{paper_id}' for every record.
9. If the paper has no extractable variants or measurements (e.g. a
   news commentary), return empty lists and explain in notes.

CONSISTENCY CHECK before returning:
   Every variant_id that appears in measurements MUST also appear in
   the variants list. If you cited a measurement for variant '4D9H',
   you must include a ProteinVariant record for '4D9H'.

Return a single PaperExtraction record.
"""

_TASK_EXPECTED_OUTPUT = (
    "A PaperExtraction object containing two parallel lists: 'variants' "
    "with one ProteinVariant per named ortholog mentioned in the paper, "
    "and 'measurements' with one BindingMeasurement per published "
    "numerical binding constant. Include a brief 'notes' string if any "
    "ambiguity or context is worth recording."
)

def build_literature_agent() -> Agent:
    """
    Constructs the Literature Researcher agent with project-standard
    persona and configuration. Stateless: returns a fresh Agent on every
    call so the orchestrator can safely run multiple papers in sequence.
    return : A configured CrewAI Agent ready to be assigned to a Task.
    """
    return Agent(
        role=_AGENT_ROLE,
        goal=_AGENT_GOAL,
        backstory=_AGENT_BACKSTORY,
        llm=_LITERATURE_LLM,
        verbose=False,
        allow_delegation=False,
    )

def build_extraction_task(
    paper_text: str = None,
    paper_id: str = None,
    agent: Agent = None,
) -> Task:
    """
    Constructs the per-paper extraction Task, with the paper's text
    interpolated into the description. The task is bound to a Pydantic
    output contract so any extraction violating the schema is rejected.
    @param paper_text: The full curated text of one paper. Required.
    @param paper_id: A short identifier used in the source_paper field
                     of every produced record (typically the filename
                     stem from the curated corpus). Required.
    @param agent: The CrewAI Agent that will execute the task. If None,
                  a fresh Literature Researcher is built.
    return : A configured CrewAI Task ready to be wrapped in a Crew.
    """
    if paper_text is None or not paper_text.strip():
        raise ValueError("paper_text must be a non-empty string")
    if paper_id is None or not paper_id.strip():
        raise ValueError("paper_id must be a non-empty string")

    if agent is None:
        agent = build_literature_agent()

    description = _TASK_DESCRIPTION_TEMPLATE.format(
        paper_text=paper_text,
        paper_id=paper_id,
    )

    return Task(
        description=description,
        expected_output=_TASK_EXPECTED_OUTPUT,
        agent=agent,
        output_pydantic=PaperExtraction,
    )

def build_extraction_crew(
    paper_text: str = None,
    paper_id: str = None,
) -> Crew:
    """
    Constructs a complete single-task Crew ready to extract from one
    paper. The orchestrator calls .kickoff() on the returned Crew.
    @param paper_text: The full curated text of one paper. Required.
    @param paper_id: A short identifier used in the source_paper field
                     of every produced record. Required.
    return : A configured CrewAI Crew with one agent and one task.
    """
    agent = build_literature_agent()
    task = build_extraction_task(
        paper_text=paper_text,
        paper_id=paper_id,
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
