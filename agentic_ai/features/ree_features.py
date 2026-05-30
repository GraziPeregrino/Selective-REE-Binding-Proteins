"""REE-specific physicochemical features (Week 2 Block 2.3).

Given a target element name (e.g. 'Neodymium'), returns a dict of
properties used as ML features. At Block 2.5 assembly time, these get
joined onto every MOESM3 row by target_element.

Data sources:
  - Atomic number: standard periodic table
  - Oxidation state: predominant state in aqueous LanM-binding
    conditions (Ln(III) for lanthanides; Sc(III), Y(III); Ca(II))
  - Ionic radius: Shannon (1976), Acta Cryst A32:751, at
    coordination number 8. For coordination-number-dependent radii,
    CN=8 was chosen because it is the typical coordination of REE
    ions in LanM EF-hand loops (Cotruvo 2018).
    https://doi.org/10.1107/S0567739476001551
  - is_lanthanide: 1 for the 4f-block elements (atomic numbers 57-71,
    excluding Promethium which has no stable isotope), 0 otherwise.

Properties held for ablation but NOT in baseline:
  - hydration_enthalpy_kj_mol: strongly correlates with ionic radius
    across Ln(III); add as a separate experiment.
  - hydrated_radius_pm: similar correlation, inconsistent literature
    values for non-Ln elements.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Hand-curated table of REE-relevant element properties.
# Each entry maps element_name -> (atomic_number, oxidation_state,
#                                  ionic_radius_pm_cn8, is_lanthanide).
# Ionic radii are from Shannon 1976 at coordination number 8.
# Values verified against the original Shannon (1976) table; charge
# density is computed at lookup time, not stored, to avoid the risk
# of inconsistency between stored and derived values.
_ELEMENT_PROPERTIES = {
    # Non-lanthanide REE-relevant elements
    "Calcium":      (20, 2, 112.0, 0),  # Ca²⁺, CN=8, natural LanM competitor
    "Scandium":     (21, 3,  87.0, 0),  # Sc³⁺, CN=8, smallest REE
    "Yttrium":      (39, 3, 101.9, 0),  # Y³⁺, CN=8, REE-adjacent

    # Lanthanides (4f block, atomic numbers 57-71)
    # Promethium (61) is omitted: no stable isotope, never used
    # in LanM binding experiments
    "Lanthanum":    (57, 3, 116.0, 1),  # La³⁺, largest lanthanide
    "Cerium":       (58, 3, 114.3, 1),
    "Praseodymium": (59, 3, 112.6, 1),
    "Neodymium":    (60, 3, 110.9, 1),
    "Samarium":     (62, 3, 107.9, 1),
    "Europium":     (63, 3, 106.6, 1),
    "Gadolinium":   (64, 3, 105.3, 1),
    "Terbium":      (65, 3, 104.0, 1),
    "Dysprosium":   (66, 3, 102.7, 1),
    "Holmium":      (67, 3, 101.5, 1),
    "Erbium":       (68, 3, 100.4, 1),
    "Thulium":      (69, 3,  99.4, 1),
    "Ytterbium":    (70, 3,  98.5, 1),
    "Lutetium":     (71, 3,  97.7, 1),  # Lu³⁺, smallest lanthanide
}


def get_ree_features(element_name: str = None) -> Optional[Dict]:
    """
    Returns the property feature dict for one REE-relevant element.
    @param element_name: An element name as it appears in MOESM3
                         (e.g. 'Neodymium', 'Terbium'). Case-sensitive
                         to match the dataset convention.
    return : Dict with 5 features (atomic_number, oxidation_state,
             ionic_radius_pm_cn8, charge_density_z_per_pm3,
             is_lanthanide), or None if the element is not in the
             table. Returning None lets the caller decide whether to
             skip the row, raise, or fill with NaN.
    """
    if element_name is None:
        return None

    entry = _ELEMENT_PROPERTIES.get(element_name)
    if entry is None:
        return None

    atomic_number, ox_state, radius_pm, is_ln = entry
    charge_density = ox_state / (radius_pm ** 3)

    return {
        "atomic_number":            atomic_number,
        "oxidation_state":          ox_state,
        "ionic_radius_pm_cn8":      radius_pm,
        "charge_density_z_per_pm3": charge_density,
        "is_lanthanide":            is_ln,
    }


def get_ree_features_batch(
    element_names: List[str] = None,
) -> List[Optional[Dict]]:
    """
    Vectorized lookup for multiple elements. Returns one dict per
    input in order; unknown elements yield None at that position.
    @param element_names: List of element name strings.
    return : List of feature dicts (or None for unknown elements).
    """
    if not element_names:
        return []
    return [get_ree_features(name) for name in element_names]


def known_elements() -> List[str]:
    """
    Returns the list of all element names known to this module,
    sorted alphabetically. Useful for completeness checks against the
    MOESM3 corpus and for documentation.
    """
    return sorted(_ELEMENT_PROPERTIES.keys())
