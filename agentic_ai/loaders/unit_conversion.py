"""Deterministic unit conversion for BindingMeasurement values.

LLM agents are unreliable at numeric conversion (Block 4.2 surfaced a
1000x systematic error when the agent was asked to populate
value_in_molar itself). This module performs the conversion in plain
Python instead.

Conversions are limited to the small set of units actually used in the
LanM literature: molar prefixes (M, mM, uM, nM, pM, fM) and unitless
ratios (logD, separation factors). Anything else returns None and the
caller decides whether to keep the record.
"""
from __future__ import annotations
from typing import Optional

# Multiplicative factor to convert a value in <key> into molar units.
# 'M' is 1.0 (no conversion). Micro and 'u' accepted as a common ASCII
# stand-in for the Greek mu. Capitalization is normalized before lookup.
_TO_MOLAR = {
    "m":   1e-3,
    "um":  1e-6,
    "μm":  1e-6,
    "nm":  1e-9,
    "pm":  1e-12,
    "fm":  1e-15,
}

# Unit strings that are intentionally not in molar units. Returning None
# for these signals to the caller that value_in_molar is not meaningful.
_UNITLESS_UNITS = {
    "unitless",
    "logd",
    "log_d",
    "sf",
    "separation_factor",
    "fold",
    "ratio",
}


def to_molar(value: float = None, units: str = None) -> Optional[float]:
    """
    Converts a numeric value with units into molar units.
    @param value: Raw numeric value.
    @param units: Unit string as reported in the source (e.g. 'M', 'pM',
                  'uM', 'unitless'). Case- and whitespace-insensitive.
    return : Value in molar units, or None when the input is unitless or
             the unit is not recognized.
    """
    if value is None or units is None:
        return None

    normalized = units.strip().lower()

    if normalized in _UNITLESS_UNITS:
        return None

    if normalized == "m":
        return float(value)

    factor = _TO_MOLAR.get(normalized)
    if factor is None:
        return None

    return float(value) * factor
