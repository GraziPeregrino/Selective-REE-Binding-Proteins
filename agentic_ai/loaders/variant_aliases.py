"""Variant name normalization and classifier for the literature corpus.

This module handles two distinct concerns kept separate by design:

  1. Normalization (`normalize_variant_name`): collapses surface-level
     variations of the same name (case, whitespace, missing tags,
     orphan mutations) into a canonical token. Pure string conditioning,
     no biology.

  2. Classification (added in subsequent commits): maps the canonical
     token to its construct_type, parent_scaffold, and (where relevant)
     a MOESM3 ortholog index for joining. This is the biology layer.

The normalizer runs first. The classifier consumes its output.

Run as a script for a quick smoke test:
    python -m agentic_ai.loaders.variant_aliases
"""
from __future__ import annotations

import re
from typing import Optional

# Organism prefixes that should be capitalized in the canonical form.
# The literature uses '<Organism>-LanM' with a capitalized prefix
# (Mex-LanM, Hans-LanM, Melba-LanM). Lowercase forms like 'mex-LanM'
# are typos to be corrected.
_KNOWN_ORGANISM_PREFIXES = {"mex", "hans", "melba"}

# ---------------------------------------------------------------------------
# TIER 1: Canonical aliases for wild-type orthologs with MOESM3 IDs
# ---------------------------------------------------------------------------
# Maps the common literature names of wild-type LanM orthologs to their
# MOESM3 row index. These are the high-value joins: every literature
# measurement of one of these proteins gets paired with the full
# 15-element selectivity profile from Diep et al. 2026.
#
# Confidence: HIGH. Each alias is stated explicitly in Diep et al. 2026
# (the paper that defined the o-N indexing system).
#
# To add a new alias, verify against MOESM3 and cite the source paper
# in a comment so the rationale is visible at the call site.
_CANONICAL_ALIASES = {
    "Mex-LanM":   "o-621",   # M. extorquens LanM (Cotruvo 2018, Diep 2026)
    "WT-LanM":    "o-621",   # 'Wild-type LanM' in Elsevier MD paper = Mex-LanM
    "WT":         "o-621",   # Bare 'WT' resolves to Mex-LanM based on its
                             # canonical Ca2+ Kd (710 µM, nihms-1790336)
    "Hans-LanM":  "o-180",   # H. quercus LanM (Mattocks 2023, Diep 2026)
    "Melba-LanM": "o-36",    # M. sp. 13MFTsu3.1M2 LanM (named in Diep 2026)
}

# ---------------------------------------------------------------------------
# TIER 2: Construct classifications for literature variants
# ---------------------------------------------------------------------------
# Maps the canonical form of each literature variant to its
# (construct_type, parent_scaffold) tuple. Drives population of the
# ProteinVariant.construct_type and ProteinVariant.parent_scaffold
# fields when persisting agent extractions.
#
# Confidence varies. Comments cite the source paper for each
# classification decision so the rationale is visible at the call site.
#
# Notes:
#   - The 6 MOESM3-native 'o-N' identifiers are NOT in this map; they
#     are populated by the MOESM3 loader directly.
#   - The 4 Tier 1 canonical aliases (Mex-LanM, etc.) are also handled
#     upstream: by the time classification runs, they have already been
#     resolved to 'o-N' form by resolve_to_canonical_id().
#   - Construct types come from the controlled vocabulary in
#     agentic_ai.schemas._KNOWN_CONSTRUCT_TYPES.
#   - parent_scaffold values come from agentic_ai.schemas._KNOWN_SCAFFOLDS,
#     with 'unknown' for variants whose scaffold lineage is not confirmed.
_CONSTRUCT_CLASSIFICATIONS = {
    # -----------------------------------------------------------------
    # Point mutants of Mex-LanM (Elsevier MD paper, 2025)
    # 4D9X = D->X substitution at position 9 of each EF hand (4 hands)
    # 4P2A = P->A substitution at position 2 of each EF hand
    # -----------------------------------------------------------------
    "4D9A":  ("point_mutant",  "Lanmodulin"),
    "4D9H":  ("point_mutant",  "Lanmodulin"),
    "4D9M":  ("point_mutant",  "Lanmodulin"),
    "4D9N":  ("point_mutant",  "Lanmodulin"),
    "4P2A":  ("point_mutant",  "Lanmodulin"),

    # -----------------------------------------------------------------
    # Point mutant of Hans-LanM (Mattocks 2023, Nature)
    # R100K = engineered monomeric variant of the dimeric Hans-LanM
    # -----------------------------------------------------------------
    "Hans-LanM(R100K)":  ("point_mutant",  "Lanmodulin"),

    # -----------------------------------------------------------------
    # Fluorescent fusion sensors (Park 2022 PNAS et al.)
    # EF hands of LanM grafted into GFP-derivative scaffolds for
    # ratiometric REE-binding readout in cells
    # -----------------------------------------------------------------
    "LanTERN":     ("fusion_sensor",  "Lanmodulin+GFP"),
    "LanM-GCaMP":  ("fusion_sensor",  "Lanmodulin+GFP"),

    # -----------------------------------------------------------------
    # Engineered chelators (biorxiv 2025.12.14.694215 and related)
    # -----------------------------------------------------------------
    "LanND-Gd":   ("engineered_chelator",  "Lanmodulin"),
    "ProCA32":    ("engineered_chelator",  "Lanmodulin"),
    "CaBM":       ("engineered_chelator",  "Calmodulin"),

    # -----------------------------------------------------------------
    # RF series — designed chelators (biorxiv 2025_10_19_682977)
    # Lineage not fully confirmed in our curated excerpts; flag for
    # follow-up while keeping the records in the dataset.
    # -----------------------------------------------------------------
    "RF1":       ("engineered_chelator",  "unknown"),
    "RF2":       ("engineered_chelator",  "unknown"),
    "RF2 6AW":   ("engineered_chelator",  "unknown"),
    "RF3":       ("engineered_chelator",  "unknown"),

    # -----------------------------------------------------------------
    # MIF — Multimetal Ion-stacking metalloprotein Framework
    # (biorxiv 2025.10.21.683075)
    # Natural lanpepsy PepSY-domain protein from M. flagellatus.
    # NOT a Lanmodulin derivative; the engineering is in immobilization
    # and chromatographic use, not in the protein scaffold itself.
    # -----------------------------------------------------------------
    "MIF":       ("ortholog",  "lanpepsy"),
    # -----------------------------------------------------------------
    # Isolated EF-hand peptides — synthetic constructs containing the
    # 12-residue LanM EF-hand loop sequence in isolation. Gutenthaler
    # 2022 (Inorganic Chemistry Frontiers) measured each EF's REE
    # binding stripped from the structural corset; biorxiv_2025.12.14
    # uses similar peptides as Gd contrast agents (relaxivity data).
    # The micromolar Kds and L/(s·mmol) relaxivities are real and
    # scientifically valuable distinctions from the picomolar Kd of
    # the full LanM scaffold.
    # -----------------------------------------------------------------
    "EF1":  ("engineered_chelator",  "Lanmodulin"),
    "EF2":  ("engineered_chelator",  "Lanmodulin"),
    "EF3":  ("engineered_chelator",  "Lanmodulin"),
    "EF4":  ("engineered_chelator",  "Lanmodulin"),
    # -----------------------------------------------------------------
    # Lanthanide Binding Tag — de novo designed peptide (Imperiali lab)
    # ~20 residues with a tryptophan antenna for Tb-luminescence.
    # Surface-displayed on E. coli/Caulobacter in some biorxiv papers.
    # NOT derived from any natural protein; classified as de_novo.
    # -----------------------------------------------------------------
    "LBT":  ("engineered_chelator",  "de_novo"),
    # -----------------------------------------------------------------
    # Non-Lanmodulin or unconfirmed orthologs
    # -----------------------------------------------------------------
    "wild-type S. oneidensis":  ("ortholog",  "non_LanM_protein"),
    "CDS J19_31570":            ("ortholog",  "unknown"),
}
# ---------------------------------------------------------------------------
# TIER 3: Drop rules — names too ambiguous to retain
# ---------------------------------------------------------------------------
# The agent occasionally extracts names that are not specific proteins,
# either because the source paper uses generic shorthand or because the
# agent confused a sub-protein motif name for a variant. Records with
# these names are dropped from the enriched dataset entirely.
#
# Confidence: HIGH. Each name is a known agent failure mode caught in
# the Block 4.3 corpus run. Adding new entries should be conservative —
# anything that COULD be a real protein under some interpretation
# stays.
_DROP_RULES = {
    "LanM",   # Generic name with no specific organism — could be any LanM,
              # and the one measurement attached (5.0 unitless Dy) is too
              # ambiguous to retain
    # NOTE: 'EF1'/'EF2'/'EF3'/'EF4' were previously in this list but
    # were promoted to Tier 2 classification when downstream inspection
    # showed they carry real Kd / relaxivity data for isolated synthetic
    # EF-hand peptides (Gutenthaler 2022 et al.).
    # 'WT' was promoted to a Tier 1 alias (-> 'o-621') after inspection
    # showed the single 'WT' measurement is Mex-LanM's canonical Ca2+ Kd.
}
# Bare mutation codes that appear without their parent name in some
# papers. Map them to the canonical <parent>(<mutation>) form.
# Currently just R100K (the Mattocks 2023 Hans-LanM monomeric variant);
# future entries should be added here with a comment citing the paper.
_BARE_MUTATIONS_TO_PARENTED = {
    "R100K": "Hans-LanM(R100K)",  # Mattocks et al. 2023, Nature
}

# Tags appended to the C- or N-terminus for purification or matrix
# conjugation. These do not change the binding behavior of the core
# protein, so we strip them in normalization. If a downstream user
# cares about tagged-vs-untagged distinctions, they can add a
# `notes` field on the variant rather than diverging the variant_id.
_STRIPPABLE_TAGS_SUFFIXES = (
    "-Cys",
    "-GSGC",
    "(GSGC)",
    "-SpyTag",
    "-SpyTag3",
    "-His6",
    "-His10",
)

# Pre-compiled regex for ortholog-ID detection: `o-36`, `O36`, `o36`,
# `O-36` etc. all normalize to the lowercase, hyphenated form.
_ORTHOLOG_ID_PATTERN = re.compile(r"^[Oo]-?(\d+)$")


def normalize_variant_name(raw: str = None) -> Optional[str]:
    """
    Collapses surface-level variations of a variant identifier into a
    canonical string token that the classifier can look up.
    Normalization order matters: rules are applied in the sequence
    below so later rules can assume earlier ones have run.
    @param raw: The variant identifier as it appears in the source
                text (e.g. from a CrewAI agent extraction).
    return : The canonical form as a string, or None if the input is
             empty, whitespace-only, or otherwise unusable.
    """
    if raw is None:
        return None

    # Rule 6: Whitespace and empty handling (first, so later rules can
    # assume non-empty input).
    cleaned = raw.strip()
    if not cleaned:
        return None

    # Rule 4: Ortholog ID canonicalization.
    # 'o36', 'O-36', 'O36', 'o-36' all -> 'o-36'.
    ortholog_match = _ORTHOLOG_ID_PATTERN.match(cleaned)
    if ortholog_match:
        return f"o-{ortholog_match.group(1)}"

    # Rule 5: Strip purification/conjugation tags. Done before other
    # transformations so 'mex-LanM-Cys' becomes 'Mex-LanM' cleanly.
    cleaned = _strip_known_tags(cleaned)

    # Rule 3: Parent bare mutation codes.
    # 'R100K' -> 'Hans-LanM(R100K)'.
    if cleaned in _BARE_MUTATIONS_TO_PARENTED:
        return _BARE_MUTATIONS_TO_PARENTED[cleaned]

    # Rule 2: Internal whitespace handling for known shorthand.
    # '4 P2A' -> '4P2A', 'WT LanM' -> 'WT-LanM'.
    cleaned = _collapse_known_internal_whitespace(cleaned)

    # Rule 1: Organism prefix capitalization.
    # 'mex-LanM' -> 'Mex-LanM'.
    cleaned = _capitalize_organism_prefix(cleaned)

    return cleaned


def _strip_known_tags(text: str) -> str:
    """
    Removes any of the known purification/conjugation tag suffixes
    from the end of a variant name. Applied iteratively in case a
    name has multiple stacked tags.
    @param text: The variant string after initial whitespace strip.
    return : The string with any trailing known tags removed.
    """
    changed = True
    while changed:
        changed = False
        for suffix in _STRIPPABLE_TAGS_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip("-")
                changed = True
                break
    return text


def _collapse_known_internal_whitespace(text: str) -> str:
    """
    Removes internal whitespace in cases known to be typos. We do NOT
    blindly strip all internal whitespace because some real names
    contain spaces meaningfully (none in the current corpus, but we
    leave the door open).
    @param text: The variant string.
    return : The string with known typo-spaces fixed.
    """
    # '4 P2A' -> '4P2A' (and similar shorthand)
    text = re.sub(r"^(\d)\s+([A-Z]\d+[A-Z])$", r"\1\2", text)

    # 'WT LanM' -> 'WT-LanM'
    text = re.sub(r"^WT\s+LanM$", "WT-LanM", text)

    return text


def _capitalize_organism_prefix(text: str) -> str:
    """
    Capitalizes the organism prefix of '<organism>-LanM' style names
    when the prefix is in our known set.
    Example: 'mex-LanM' -> 'Mex-LanM'.
    @param text: The variant string.
    return : The string with the organism prefix capitalized if it
             matched a known organism, otherwise unchanged.
    """
    parts = text.split("-", 1)
    if len(parts) != 2:
        return text

    prefix, rest = parts
    if prefix.lower() in _KNOWN_ORGANISM_PREFIXES:
        return f"{prefix.capitalize()}-{rest}"

    return text


def resolve_to_canonical_id(raw: str = None) -> Optional[str]:
    """
    Normalizes a raw variant identifier and then resolves it to its
    canonical MOESM3 ortholog ID if a known alias exists.
    This is the integration point between Tier 1 alias resolution and
    the rest of the pipeline. Block 5 will use this to join literature
    measurements to MOESM3 records.
    @param raw: A variant identifier as it appears in source text.
    return : The canonical ID (e.g. 'o-621'), or the normalized form
             when no alias is known, or None when the input is
             unusable. Callers should treat 'o-N' format identifiers
             as MOESM3-joinable and everything else as standalone.
    """
    normalized = normalize_variant_name(raw)
    if normalized is None:
        return None

    return _CANONICAL_ALIASES.get(normalized, normalized)


def classify_construct(
    canonical_name: str = None,
) -> Optional[tuple]:
    """
    Returns the (construct_type, parent_scaffold) classification for a
    canonical variant name. Caller must pass already-normalized input
    (i.e. the output of normalize_variant_name() or the matching
    canonical form).
    @param canonical_name: A normalized variant identifier.
    return : A (construct_type, parent_scaffold) tuple when the variant
             is classified, or None when it is unknown. 'o-N' format
             identifiers (MOESM3-native or aliased) are NOT classified
             here; they receive 'ortholog' + 'Lanmodulin' from the
             MOESM3 loader directly. Returning None for them indicates
             'use the MOESM3 defaults, do not override.'
    """
    if canonical_name is None:
        return None

    return _CONSTRUCT_CLASSIFICATIONS.get(canonical_name)


def should_drop(canonical_name: str = None) -> bool:
    """
    Returns True if the canonical name is in the drop list and should
    not appear in the enriched dataset. Caller passes the already-
    normalized form.
    @param canonical_name: A normalized variant identifier, or None.
    return : True if the name should be dropped; False otherwise
             (including for None and unknown names — these are not the
             drop list's responsibility).
    """
    if canonical_name is None:
        return False

    return canonical_name in _DROP_RULES


def enrich_variant(
    raw_name: str = None,
) -> Optional[dict]:
    """
    End-to-end variant enrichment: takes a raw variant identifier from
    a literature extraction and returns its full canonical metadata, or
    None if the variant should be dropped.
    This is the integration point Block 4.4c will use to populate the
    construct_type, parent_scaffold, and (where applicable) MOESM3
    join-key fields on persisted PaperExtraction records.
    @param raw_name: A variant identifier as it appears in source text.
    return : Either a dict with keys:
               - 'canonical_id': str (possibly an 'o-N' MOESM3 index)
               - 'construct_type': str
               - 'parent_scaffold': Optional[str]
             or None if the variant should be dropped or is unparseable.
    """
    normalized = normalize_variant_name(raw_name)
    if normalized is None:
        return None

    if should_drop(normalized):
        return None

    canonical_id = _CANONICAL_ALIASES.get(normalized, normalized)

    classification = _CONSTRUCT_CLASSIFICATIONS.get(normalized)
    if classification is not None:
        construct_type, parent_scaffold = classification
    elif canonical_id.startswith("o-"):
        # MOESM3-native or aliased to MOESM3: assume ortholog/Lanmodulin
        construct_type = "ortholog"
        parent_scaffold = "Lanmodulin"
    else:
        # Unknown variant outside both the classification map and MOESM3
        construct_type = "unknown"
        parent_scaffold = None

    return {
        "canonical_id": canonical_id,
        "construct_type": construct_type,
        "parent_scaffold": parent_scaffold,
    }
