"""Tests for deterministic literature-unit conversion."""
from __future__ import annotations

import pytest

from agentic_ai.loaders.unit_conversion import to_molar


@pytest.mark.parametrize("units", ["uM", "μM", "µM"])
def test_to_molar_accepts_common_micromolar_spellings(units):
    """
    Verifies that ASCII and both Unicode micro symbols convert equally.
    """
    assert to_molar(3.69, units) == pytest.approx(3.69e-6)


def test_to_molar_accepts_mol_per_liter_notation():
    """
    Verifies conversion of the notation used by one persisted paper.
    """
    assert to_molar(1.25e-7, "Mol L^{-1}") == pytest.approx(1.25e-7)


def test_to_molar_leaves_non_molar_units_unconverted():
    """
    Verifies that process metrics remain outside the molar target field.
    """
    assert to_molar(94.7, "mol%") is None
