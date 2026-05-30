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


def refresh_persisted_molar_values(
    input_dir: Path = None,
    output_dir: Path = None,
) -> int:
    """
    Recomputes value_in_molar for every persisted measurement using the
    deterministic unit converter and writes the refreshed extractions.
    Useful after adding support for a newly observed unit spelling.
    @return : Number of measurements whose normalized value changed.
    """
    from agentic_ai.loaders.unit_conversion import to_molar

    if input_dir is None:
        input_dir = _DEFAULT_EXTRACTIONS_DIR
    if output_dir is None:
        output_dir = input_dir

    extractions = load_extractions(input_dir=input_dir)
    changed = 0

    for extraction in extractions.values():
        refreshed = []
        for measurement in extraction.measurements:
            value_in_molar = to_molar(
                measurement.value,
                measurement.units,
            )
            if value_in_molar != measurement.value_in_molar:
                changed += 1
            refreshed.append(
                measurement.model_copy(
                    update={"value_in_molar": value_in_molar}
                )
            )
        extraction.measurements = refreshed

    save_extractions(extractions, output_dir=output_dir)
    return changed


def enrich_persisted_extractions(
    input_dir: Path = None,
    output_dir: Path = None,
    dry_run: bool = False,
) -> dict:
    """
    Loads every persisted PaperExtraction, applies variant enrichment
    via agentic_ai.loaders.variant_aliases.enrich_variant, and re-saves
    the result. The enrichment:
      - canonicalizes variant_ids (e.g. 'mex-LanM' -> 'o-621')
      - populates construct_type and parent_scaffold on each variant
      - drops variants whose names are in the Tier 3 drop list
      - propagates canonicalized variant_ids to every BindingMeasurement
        so foreign keys still resolve after the canonicalization
      - drops measurements whose canonicalized variant_id no longer
        appears in the (post-drop) variants list
    @param input_dir: Directory holding the JSON extractions to enrich.
                      Defaults to data/processed/extractions.
    @param output_dir: Directory to write enriched JSONs to. Defaults
                       to input_dir (in-place rewrite).
    @param dry_run: When True, computes the changes but does not write
                    to disk. Useful for previewing impact.
    return : A dict summarizing the run with keys: papers_processed,
             variants_kept, variants_dropped, variants_canonicalized,
             measurements_kept, measurements_dropped.
    """
    from agentic_ai.loaders.variant_aliases import enrich_variant
    from agentic_ai.schemas import ProteinVariant

    if input_dir is None:
        input_dir = _DEFAULT_EXTRACTIONS_DIR
    if output_dir is None:
        output_dir = input_dir

    extractions = load_extractions(input_dir=input_dir)

    summary = {
        "papers_processed": 0,
        "variants_kept": 0,
        "variants_dropped": 0,
        "variants_canonicalized": 0,
        "variants_auto_created": 0,
        "measurements_kept": 0,
        "measurements_dropped": 0,
    }

    enriched_by_paper = {}

    for paper_id, extraction in extractions.items():
        # Pass 1: enrich every explicitly-listed ProteinVariant.
        # rename_map covers any variant the paper bothered to list.
        rename_map = {}
        new_variants = []

        for variant in extraction.variants:
            enriched = enrich_variant(variant.variant_id)

            if enriched is None:
                summary["variants_dropped"] += 1
                continue

            canonical_id = enriched["canonical_id"]
            if canonical_id != variant.variant_id:
                summary["variants_canonicalized"] += 1

            rename_map[variant.variant_id] = canonical_id

            new_variants.append(
                variant.model_copy(update={
                    "variant_id": canonical_id,
                    "construct_type": enriched["construct_type"],
                    "parent_scaffold": enriched["parent_scaffold"],
                })
            )
            summary["variants_kept"] += 1

        # Pass 2: walk measurements. Resolve each measurement's
        # variant_id through the same enrich chain so the foreign key
        # always points at the canonical name, even when the agent
        # used inconsistent names within its own output.
        valid_variant_ids = {v.variant_id for v in new_variants}
        new_measurements = []
        auto_created_ids = set()
        sample_paper_for_auto = paper_id

        for measurement in extraction.measurements:
            # First check if the measurement's existing variant_id was
            # already renamed in Pass 1 (the common case).
            new_variant_id = rename_map.get(measurement.variant_id)

            # If not, the measurement's variant_id wasn't listed under
            # variants in this paper. Try enriching it directly — this
            # handles two cases:
            #   (a) measurement uses a different canonical form of an
            #       existing variant (e.g. measurement says
            #       'Hans-LanM(R100K)' while the variant entry says
            #       'R100K'); enriching the measurement's id reaches
            #       the same canonical form.
            #   (b) the agent recorded a measurement for a variant it
            #       never bothered to list (LBT case). We auto-create
            #       a minimal ProteinVariant for those.
            if new_variant_id is None:
                enriched = enrich_variant(measurement.variant_id)
                if enriched is None:
                    summary["measurements_dropped"] += 1
                    continue
                new_variant_id = enriched["canonical_id"]

                # Case (b): variant doesn't exist yet in this paper.
                if new_variant_id not in valid_variant_ids:
                    new_variants.append(ProteinVariant(
                        variant_id=new_variant_id,
                        source_organism="(auto-created from measurement)",
                        construct_type=enriched["construct_type"],
                        parent_scaffold=enriched["parent_scaffold"],
                        notes="Auto-created during Block 4.4c enrichment "
                              "from an orphan measurement whose variant "
                              "record was missing in the agent extraction.",
                        source_paper=sample_paper_for_auto,
                    ))
                    valid_variant_ids.add(new_variant_id)
                    auto_created_ids.add(new_variant_id)

            if new_variant_id not in valid_variant_ids:
                summary["measurements_dropped"] += 1
                continue

            new_measurements.append(
                measurement.model_copy(update={"variant_id": new_variant_id})
            )
            summary["measurements_kept"] += 1

        summary["variants_auto_created"] += len(auto_created_ids)

        # Dedupe variants by canonical_id. The agent sometimes lists the
        # same protein under both its literature name and its MOESM3 ID
        # within the same paper; after canonicalization those collapse
        # to the same variant_id, so we merge them with field-merging
        # semantics (longer/richer non-overlapping fields win).
        new_variants, merged_count = _dedupe_variants_by_id(new_variants)
        summary.setdefault("variants_merged_duplicates", 0)
        summary["variants_merged_duplicates"] += merged_count
        summary["variants_kept"] -= merged_count

        new_extraction = extraction.model_copy(update={
            "variants": new_variants,
            "measurements": new_measurements,
        })
        enriched_by_paper[paper_id] = new_extraction
        summary["papers_processed"] += 1

    if not dry_run:
        save_extractions(enriched_by_paper, output_dir=output_dir)

    return summary


def _dedupe_variants_by_id(
    variants: list,
) -> tuple:
    """
    Collapses duplicate ProteinVariant records that share the same
    variant_id within a paper. The agent occasionally lists the same
    protein under both its literature name and its MOESM3 index;
    canonicalization collapses both to the same variant_id and the
    duplicate records arrive at this step.
    Merge policy: combine non-overlapping fields across duplicates so
    no signal is lost. When two duplicates both populate the same
    field, the longer/richer value wins (longest sequence, longest
    notes, deduplicated union of mutations).
    @param variants: List of ProteinVariant records (possibly with
                     intra-list variant_id collisions).
    return : Tuple of (deduplicated list, count of merges performed).
    """
    by_id = {}
    merges = 0

    for v in variants:
        existing = by_id.get(v.variant_id)
        if existing is None:
            by_id[v.variant_id] = v
            continue

        merges += 1
        by_id[v.variant_id] = _merge_variants(existing, v)

    return list(by_id.values()), merges


def _merge_variants(a, b):
    """
    Returns a new ProteinVariant with non-overlapping fields combined
    from a and b, and overlapping fields resolved by 'longer wins'.
    @param a: One of two ProteinVariants sharing a variant_id.
    @param b: The other ProteinVariant.
    return : A merged ProteinVariant.
    """
    def longer_string(x, y):
        """Pick the longer non-empty string, or whichever is non-None."""
        if not x:
            return y
        if not y:
            return x
        return x if len(x) >= len(y) else y

    def union_mutations(x, y):
        """Deduplicate mutations across both lists, preserving order."""
        seen = set()
        combined = []
        for item in list(x or []) + list(y or []):
            if item not in seen:
                seen.add(item)
                combined.append(item)
        return combined

    def first_non_none(x, y):
        return x if x is not None else y

    return a.model_copy(update={
        "source_organism": longer_string(
            a.source_organism, b.source_organism
        ),
        "sequence":            longer_string(a.sequence, b.sequence),
        "parent_variant_id": first_non_none(
            a.parent_variant_id, b.parent_variant_id
        ),
        "mutations":           union_mutations(a.mutations, b.mutations),
        "mutation_notation": first_non_none(
            a.mutation_notation, b.mutation_notation
        ),
        "taxonomy":            longer_string(a.taxonomy, b.taxonomy),
        "ef_hand_count": first_non_none(
            a.ef_hand_count, b.ef_hand_count
        ),
        "selectivity_cluster": first_non_none(
            a.selectivity_cluster, b.selectivity_cluster
        ),
        "notes":               longer_string(a.notes, b.notes),
        # construct_type, parent_scaffold, variant_id, source_paper are
        # all canonical at this point — keep `a`'s (they should match).
    })
