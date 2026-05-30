"""Tests for the variant name normalizer (Week 1 Block 4.4b)."""
from __future__ import annotations

import pytest

from agentic_ai.loaders.variant_aliases import normalize_variant_name


# ---------------------------------------------------------------------------
# Rule 6: empty / whitespace / None handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_normalize_returns_none_for_empty_input(raw):
    """
    Verifies that any falsy or whitespace-only input normalizes to
    None, signaling 'no usable name' to the classifier.
    """
    assert normalize_variant_name(raw) is None


def test_normalize_strips_outer_whitespace_from_real_names():
    """
    Verifies that real names with surrounding whitespace are stripped
    but otherwise preserved.
    """
    assert normalize_variant_name("  Mex-LanM  ") == "Mex-LanM"


# ---------------------------------------------------------------------------
# Rule 1: organism prefix capitalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("mex-LanM", "Mex-LanM"),
    ("hans-LanM", "Hans-LanM"),
    ("melba-LanM", "Melba-LanM"),
])
def test_normalize_capitalizes_lowercase_organism_prefix(raw, expected):
    """
    Verifies that lowercase organism prefixes in <organism>-LanM names
    are capitalized to canonical form.
    """
    assert normalize_variant_name(raw) == expected


@pytest.mark.parametrize("name", ["Mex-LanM", "Hans-LanM", "Melba-LanM"])
def test_normalize_preserves_already_canonical_names(name):
    """
    Verifies that canonical names pass through unchanged.
    """
    assert normalize_variant_name(name) == name


def test_normalize_preserves_wt_lanm_distinction():
    """
    Verifies that 'WT-LanM' (a legitimate distinct name, not a typo)
    is preserved rather than collapsed.
    """
    assert normalize_variant_name("WT-LanM") == "WT-LanM"


def test_normalize_leaves_unknown_organism_prefixes_unchanged():
    """
    Verifies that the capitalization rule only fires for organisms in
    our known set. 'foo-LanM' should NOT become 'Foo-LanM' (we want
    novel names to surface to the classifier as-is for inspection).
    """
    assert normalize_variant_name("foo-LanM") == "foo-LanM"


# ---------------------------------------------------------------------------
# Rule 2: internal whitespace in shorthand
# ---------------------------------------------------------------------------

def test_normalize_collapses_internal_space_in_mutation_shorthand():
    """
    Verifies that '4 P2A' (typo with internal space) becomes '4P2A'.
    """
    assert normalize_variant_name("4 P2A") == "4P2A"


def test_normalize_converts_wt_space_lanm_to_canonical_form():
    """
    Verifies that 'WT LanM' becomes 'WT-LanM'.
    """
    assert normalize_variant_name("WT LanM") == "WT-LanM"


# ---------------------------------------------------------------------------
# Rule 3: bare mutation parenting
# ---------------------------------------------------------------------------

def test_normalize_parents_bare_r100k_to_hans_lanm():
    """
    Verifies that the bare 'R100K' mutation code is parented to
    'Hans-LanM(R100K)' since R100K is the canonical Hans-LanM
    monomeric variant in the literature.
    """
    assert normalize_variant_name("R100K") == "Hans-LanM(R100K)"


def test_normalize_preserves_already_parented_mutations():
    """
    Verifies that 'Hans-LanM(R100K)' (already canonical) is unchanged.
    """
    assert normalize_variant_name("Hans-LanM(R100K)") == "Hans-LanM(R100K)"


@pytest.mark.parametrize("name", ["4D9A", "4D9H", "4D9M", "4D9N", "4P2A"])
def test_normalize_preserves_4d9x_shorthand(name):
    """
    Verifies that the 4D9X EF-hand shorthand notation (LanM-specific
    mutation convention from the Elsevier MD paper) passes through
    unchanged. These don't fit the <parent>(<mutation>) pattern.
    """
    assert normalize_variant_name(name) == name


# ---------------------------------------------------------------------------
# Rule 4: ortholog ID canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("o-36", "o-36"),
    ("o36", "o-36"),
    ("O-36", "o-36"),
    ("O36", "o-36"),
    ("o-621", "o-621"),
    ("O-180", "o-180"),
])
def test_normalize_canonicalizes_ortholog_id_capitalization_and_hyphen(raw, expected):
    """
    Verifies that ortholog IDs in any capitalization or hyphen variant
    collapse to the canonical lowercase-hyphenated form matching
    MOESM3.
    """
    assert normalize_variant_name(raw) == expected


def test_normalize_does_not_match_ortholog_pattern_on_engineered_names():
    """
    Verifies that the ortholog pattern is strict enough not to false-
    positive on engineered constructs that happen to start with 'O'.
    """
    assert normalize_variant_name("Ortho-construct") == "Ortho-construct"


# ---------------------------------------------------------------------------
# Rule 5: tag stripping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Mex-LanM-Cys", "Mex-LanM"),
    ("Hans-LanM-Cys", "Hans-LanM"),
    ("o-36(GSGC)", "o-36"),
    ("Mex-LanM-SpyTag", "Mex-LanM"),
    ("Hans-LanM-SpyTag3", "Hans-LanM"),
    ("Mex-LanM-His6", "Mex-LanM"),
    ("Mex-LanM-His10", "Mex-LanM"),
])
def test_normalize_strips_purification_and_conjugation_tags(raw, expected):
    """
    Verifies that purification/conjugation tag suffixes are stripped
    so tagged constructs join cleanly to their untagged equivalents.
    """
    assert normalize_variant_name(raw) == expected


def test_normalize_strips_multiple_stacked_tags():
    """
    Verifies that names with multiple stacked tags (e.g. SpyTag plus
    His10) get all tags stripped.
    """
    assert normalize_variant_name("Mex-LanM-His10-SpyTag") == "Mex-LanM"


def test_normalize_combines_tag_strip_with_capitalization():
    """
    Verifies that tag stripping composes correctly with organism
    prefix capitalization: 'mex-LanM-Cys' -> 'Mex-LanM'.
    """
    assert normalize_variant_name("mex-LanM-Cys") == "Mex-LanM"

# ---------------------------------------------------------------------------
# TIER 1: alias resolution
# ---------------------------------------------------------------------------

from agentic_ai.loaders.variant_aliases import resolve_to_canonical_id


@pytest.mark.parametrize("raw,expected", [
    ("Mex-LanM",   "o-621"),
    ("mex-LanM",   "o-621"),   # capitalization typo, normalizer fixes first
    ("WT-LanM",    "o-621"),
    ("WT LanM",    "o-621"),   # whitespace typo, normalizer fixes first
    ("Hans-LanM",  "o-180"),
    ("hans-LanM",  "o-180"),
    ("Melba-LanM", "o-36"),
    ("melba-LanM", "o-36"),
])
def test_resolve_canonical_aliases_to_moesm3_ids(raw, expected):
    """
    Verifies that the four wild-type orthologs with known MOESM3
    indices resolve correctly through both normalization and alias
    lookup. These are the high-value joins powering the Block 5 merge.
    """
    assert resolve_to_canonical_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "o-36", "o-127", "o-412", "o-543", "o-585", "o-621",
])
def test_resolve_passes_through_native_ortholog_ids(raw):
    """
    Verifies that 'o-N' identifiers already in MOESM3 form pass through
    the resolver unchanged. These are the variants the agent extracted
    using the same notation MOESM3 uses, so no aliasing is needed.
    """
    assert resolve_to_canonical_id(raw) == raw


def test_resolve_returns_normalized_form_for_unknown_aliases():
    """
    Verifies that variants without a known canonical alias get the
    normalized form returned, not None. The Block 5 merge will treat
    these as standalone records (no MOESM3 join, but the record is
    kept).
    """
    assert resolve_to_canonical_id("4D9H") == "4D9H"
    assert resolve_to_canonical_id("LanTERN") == "LanTERN"
    assert resolve_to_canonical_id("MIF") == "MIF"


def test_resolve_returns_none_for_empty_input():
    """
    Verifies that empty input flows through both the normalizer and
    the alias step, yielding None at the end.
    """
    assert resolve_to_canonical_id(None) is None
    assert resolve_to_canonical_id("") is None
    assert resolve_to_canonical_id("   ") is None


def test_resolve_chains_tag_strip_with_alias_lookup():
    """
    Verifies that a tagged variant gets the tag stripped by the
    normalizer AND then resolved through the alias map. This is the
    end-to-end test for the most surface-variation-heavy input shape.
    """
    assert resolve_to_canonical_id("mex-LanM-Cys") == "o-621"
    assert resolve_to_canonical_id("Hans-LanM-SpyTag3") == "o-180"


def test_resolve_handles_bare_mutation_through_alias_chain():
    """
    Verifies that bare 'R100K' gets parented to 'Hans-LanM(R100K)' by
    the normalizer, then resolves through alias lookup (returning the
    normalized form since R100K is a mutant, not a wild-type ortholog
    in MOESM3).
    """
    assert resolve_to_canonical_id("R100K") == "Hans-LanM(R100K)"

    # ---------------------------------------------------------------------------
# TIER 2: construct classification
# ---------------------------------------------------------------------------

from agentic_ai.loaders.variant_aliases import classify_construct


@pytest.mark.parametrize("variant_id", ["4D9A", "4D9H", "4D9M", "4D9N", "4P2A"])
def test_classify_mex_lanm_point_mutants_correctly(variant_id):
    """
    Verifies that the five Mex-LanM EF-hand point mutants from the
    Elsevier MD paper are classified as point mutants of Lanmodulin.
    """
    construct_type, scaffold = classify_construct(variant_id)
    assert construct_type == "point_mutant"
    assert scaffold == "Lanmodulin"


def test_classify_hans_lanm_r100k_as_point_mutant():
    """
    Verifies that the Hans-LanM(R100K) monomeric variant is classified
    as a point mutant of Lanmodulin.
    """
    assert classify_construct("Hans-LanM(R100K)") == ("point_mutant", "Lanmodulin")


@pytest.mark.parametrize("variant_id", ["LanTERN", "LanM-GCaMP"])
def test_classify_fluorescent_sensors_as_fusion_constructs(variant_id):
    """
    Verifies that LanM-derived fluorescent sensors are classified as
    fusion sensors on the Lanmodulin+GFP scaffold.
    """
    construct_type, scaffold = classify_construct(variant_id)
    assert construct_type == "fusion_sensor"
    assert scaffold == "Lanmodulin+GFP"


def test_classify_lanmodulin_chelators_correctly():
    """
    Verifies that engineered chelators derived from Lanmodulin are
    classified with the right parent scaffold.
    """
    assert classify_construct("LanND-Gd") == ("engineered_chelator", "Lanmodulin")
    assert classify_construct("ProCA32") == ("engineered_chelator", "Lanmodulin")


def test_classify_calmodulin_chelator_as_distinct_scaffold():
    """
    Verifies that CaBM is annotated as a Calmodulin-derived chelator,
    not silently lumped with Lanmodulin constructs.
    """
    assert classify_construct("CaBM") == ("engineered_chelator", "Calmodulin")


@pytest.mark.parametrize("variant_id", ["RF1", "RF2", "RF2 6AW", "RF3"])
def test_classify_rf_series_with_unknown_scaffold(variant_id):
    """
    Verifies that the RF-series engineered chelators are kept in the
    dataset with explicit 'unknown' scaffold annotation rather than
    being dropped or silently labeled.
    """
    construct_type, scaffold = classify_construct(variant_id)
    assert construct_type == "engineered_chelator"
    assert scaffold == "unknown"


def test_classify_mif_as_lanpepsy_ortholog():
    """
    Verifies that MIF is classified as an ortholog with parent_scaffold
    'lanpepsy', preserving the architectural distinction between LanM
    (EF-hand) and lanpepsy (PepSY-domain) lineages. This is the key
    Block 4.4 finding for the Week 3 ML model.
    """
    assert classify_construct("MIF") == ("ortholog", "lanpepsy")


def test_classify_non_lanm_protein_preserves_distinction():
    """
    Verifies that the S. oneidensis wild-type is classified as an
    ortholog with explicit non_LanM_protein scaffold marker.
    """
    assert classify_construct("wild-type S. oneidensis") == (
        "ortholog",
        "non_LanM_protein",
    )


def test_classify_returns_none_for_unknown_variant():
    """
    Verifies that variants outside the classification map return None,
    signaling to the caller 'fall through to defaults or drop'.
    'o-N' MOESM3-native IDs fall into this bucket since the loader
    has already set their fields.
    """
    assert classify_construct("o-621") is None
    assert classify_construct("o-36") is None
    assert classify_construct("CompletelyMadeUpVariant") is None


def test_classify_returns_none_for_none_input():
    """
    Verifies that None input flows through cleanly rather than raising.
    """
    assert classify_construct(None) is None


def test_classify_does_not_short_circuit_normalization_responsibility():
    """
    Verifies that classify_construct() expects already-normalized input.
    Surface-variant forms like 'mex-LanM' should return None because
    they are not in the canonical map; the caller is expected to
    normalize first via resolve_to_canonical_id().
    """
    assert classify_construct("mex-lanm") is None
    assert classify_construct("Mex-LanM-Cys") is None