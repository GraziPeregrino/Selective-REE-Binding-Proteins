"""Tests for REE-specific physicochemical features (Week 2 Block 2.3)."""
from __future__ import annotations

from agentic_ai.features.ree_features import (
    get_ree_features,
    get_ree_features_batch,
    known_elements,
)


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

def test_known_elements_returns_17_entries():
    """
    Verifies that the lookup table covers all 14 lanthanides (Pm
    excluded) plus Ca, Sc, Y. Pins the count so accidental
    additions/removals break the test.
    """
    elements = known_elements()
    assert len(elements) == 17


def test_known_elements_includes_all_moesm3_elements():
    """
    Verifies that every element actually used in the MOESM3 dataset
    has a lookup entry. This is the most important coverage test:
    a missing element would silently produce NaN features at
    assembly time.
    """
    moesm3_elements = {
        "Cerium", "Dysprosium", "Erbium", "Europium", "Gadolinium",
        "Holmium", "Lanthanum", "Lutetium", "Neodymium", "Praseodymium",
        "Samarium", "Terbium", "Thulium", "Ytterbium", "Yttrium",
    }
    table_elements = set(known_elements())
    assert moesm3_elements.issubset(table_elements)


def test_known_elements_includes_calcium_and_scandium():
    """
    Verifies that the non-lanthanide REE-adjacent elements (Ca, Sc)
    are present even though MOESM3 doesn't test them. Useful for
    Week 3+ literature joins where Ca²⁺ measurements appear.
    """
    elements = set(known_elements())
    assert "Calcium" in elements
    assert "Scandium" in elements


def test_known_elements_excludes_promethium():
    """
    Verifies that Promethium is correctly absent. Pm has no stable
    isotope and is never used in REE-binding experiments.
    """
    assert "Promethium" not in known_elements()


# ---------------------------------------------------------------------------
# Feature shape and key contract
# ---------------------------------------------------------------------------

def test_get_features_returns_five_keys():
    """
    Verifies that every lookup returns exactly 5 features.
    """
    features = get_ree_features("Neodymium")
    assert len(features) == 5


def test_get_features_returns_documented_keys():
    """
    Verifies that the feature dict has the documented keys.
    """
    features = get_ree_features("Neodymium")
    expected_keys = {
        "atomic_number", "oxidation_state", "ionic_radius_pm_cn8",
        "charge_density_z_per_pm3", "is_lanthanide",
    }
    assert set(features.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Specific element values
# ---------------------------------------------------------------------------

def test_calcium_has_oxidation_state_2():
    """
    Verifies that Calcium has +2 oxidation state, distinguishing it
    from the +3 lanthanides. This is the dominant chemical signal
    differentiating Ca from REE in LanM binding.
    """
    features = get_ree_features("Calcium")
    assert features["oxidation_state"] == 2
    assert features["is_lanthanide"] == 0


def test_lanthanum_is_lanthanide_with_oxidation_3():
    """
    Verifies Lanthanum (the prototype lanthanide) has +3 ox state
    and the is_lanthanide flag.
    """
    features = get_ree_features("Lanthanum")
    assert features["atomic_number"] == 57
    assert features["oxidation_state"] == 3
    assert features["is_lanthanide"] == 1


def test_yttrium_is_not_lanthanide_despite_being_ree_adjacent():
    """
    Verifies that Yttrium has is_lanthanide=0 even though chemically
    REE-adjacent. The flag must reflect 4f-block membership, not
    chemical similarity, so the model can learn the two signals
    separately.
    """
    features = get_ree_features("Yttrium")
    assert features["atomic_number"] == 39
    assert features["oxidation_state"] == 3
    assert features["is_lanthanide"] == 0


def test_all_lanthanides_have_is_lanthanide_one():
    """
    Verifies that every 4f-block element gets is_lanthanide=1.
    """
    lanthanides = [
        "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
        "Samarium", "Europium", "Gadolinium", "Terbium",
        "Dysprosium", "Holmium", "Erbium", "Thulium",
        "Ytterbium", "Lutetium",
    ]
    for elem in lanthanides:
        assert get_ree_features(elem)["is_lanthanide"] == 1


def test_non_lanthanides_have_is_lanthanide_zero():
    """
    Verifies that Ca, Sc, Y all have is_lanthanide=0.
    """
    for elem in ["Calcium", "Scandium", "Yttrium"]:
        assert get_ree_features(elem)["is_lanthanide"] == 0


# ---------------------------------------------------------------------------
# Lanthanide contraction
# ---------------------------------------------------------------------------

def test_lanthanide_contraction_is_monotonic():
    """
    Verifies the textbook lanthanide contraction: ionic radius
    strictly decreases from La (Z=57) through Lu (Z=71). This is
    the foundational chemistry signal the ML model will leverage.
    """
    lanthanides_in_order = [
        "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
        "Samarium", "Europium", "Gadolinium", "Terbium",
        "Dysprosium", "Holmium", "Erbium", "Thulium",
        "Ytterbium", "Lutetium",
    ]
    radii = [
        get_ree_features(elem)["ionic_radius_pm_cn8"]
        for elem in lanthanides_in_order
    ]
    for i in range(len(radii) - 1):
        assert radii[i] > radii[i + 1], (
            f"Radius did not decrease from {lanthanides_in_order[i]} "
            f"to {lanthanides_in_order[i+1]}: {radii[i]} -> {radii[i+1]}"
        )


def test_la_lu_total_contraction_in_expected_range():
    """
    Verifies that the total contraction from La to Lu is approximately
    18 pm, consistent with the literature.
    """
    la = get_ree_features("Lanthanum")["ionic_radius_pm_cn8"]
    lu = get_ree_features("Lutetium")["ionic_radius_pm_cn8"]
    contraction = la - lu
    assert 16 <= contraction <= 20


# ---------------------------------------------------------------------------
# Charge density derivation
# ---------------------------------------------------------------------------

def test_charge_density_is_computed_from_ox_state_and_radius():
    """
    Verifies that charge density equals oxidation_state divided by
    radius cubed. This is the derived value contract — the table
    stores the inputs, charge_density is computed at lookup time
    to prevent transcription errors.
    """
    features = get_ree_features("Neodymium")
    expected = features["oxidation_state"] / features["ionic_radius_pm_cn8"] ** 3
    assert abs(features["charge_density_z_per_pm3"] - expected) < 1e-15


def test_calcium_has_lower_charge_density_than_lanthanides():
    """
    Verifies that Ca²⁺ has lower charge density than even the largest
    Ln³⁺ (Lanthanum) because of the +2 vs +3 charge difference.
    This is the key separator the ML model needs.
    """
    ca = get_ree_features("Calcium")["charge_density_z_per_pm3"]
    la = get_ree_features("Lanthanum")["charge_density_z_per_pm3"]
    assert ca < la


def test_lu_has_highest_charge_density_among_lanthanides():
    """
    Verifies that Lutetium (smallest Ln) has the highest charge
    density among lanthanides. Reflects both the lanthanide
    contraction and the constant +3 charge.
    """
    lanthanides = [
        "Lanthanum", "Cerium", "Neodymium", "Samarium",
        "Gadolinium", "Dysprosium", "Erbium", "Ytterbium", "Lutetium",
    ]
    charge_densities = {
        e: get_ree_features(e)["charge_density_z_per_pm3"]
        for e in lanthanides
    }
    max_elem = max(charge_densities, key=charge_densities.get)
    assert max_elem == "Lutetium"


# ---------------------------------------------------------------------------
# Unknown element handling
# ---------------------------------------------------------------------------

def test_unknown_element_returns_none():
    """
    Verifies that an element name not in the table returns None
    rather than raising. Callers decide whether to skip, fill NaN,
    or raise.
    """
    assert get_ree_features("Plutonium") is None
    assert get_ree_features("MadeUpElement") is None


def test_none_input_returns_none():
    """
    Verifies that None input returns None rather than raising.
    """
    assert get_ree_features(None) is None


def test_lookup_is_case_sensitive():
    """
    Verifies that lowercase or mixed-case element names are not
    matched. MOESM3 uses Title-Case names exclusively; we enforce
    that convention strictly to surface upstream data inconsistencies.
    """
    assert get_ree_features("neodymium") is None
    assert get_ree_features("NEODYMIUM") is None


# ---------------------------------------------------------------------------
# Batch lookup
# ---------------------------------------------------------------------------

def test_batch_returns_one_result_per_input():
    """
    Verifies batch shape contract: N inputs -> N outputs.
    """
    results = get_ree_features_batch(["Neodymium", "Calcium", "Lutetium"])
    assert len(results) == 3


def test_batch_preserves_input_order():
    """
    Verifies that batch output order matches input order, important
    for joining onto a DataFrame.
    """
    elements = ["Lutetium", "Calcium", "Lanthanum"]
    results = get_ree_features_batch(elements)
    assert results[0]["atomic_number"] == 71  # Lu
    assert results[1]["atomic_number"] == 20  # Ca
    assert results[2]["atomic_number"] == 57  # La


def test_batch_returns_none_for_unknown_without_aborting():
    """
    Verifies that batch tolerates unknown elements by placing None
    at their position. Important for processing large datasets where
    a single bad entry should not block the rest.
    """
    results = get_ree_features_batch([
        "Neodymium", "MadeUp", "Lutetium",
    ])
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


def test_batch_returns_empty_list_for_empty_input():
    """
    Verifies that empty/None input does not crash.
    """
    assert get_ree_features_batch([]) == []
    assert get_ree_features_batch(None) == []
