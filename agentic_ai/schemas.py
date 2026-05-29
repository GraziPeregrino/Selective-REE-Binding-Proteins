"""Pydantic schemas for the REE-binding pipeline (Path C, Week 1 Block 3).

Two-tier relational design:
  - ProteinVariant: one record per protein (sequence, organism, mutations, cluster).
  - BindingMeasurement: one record per (variant x REE x measurement-type) triple.

A single XLSX row (Diep et al. 2026, MOESM3) yields one ProteinVariant and
fifteen BindingMeasurement records. A single literature passage extracted by
the CrewAI agent in Block 4 may yield one variant and one or more measurements.

Both record types carry `source_paper` for provenance and downstream dedup.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

# The 20 canonical amino acid one-letter codes
_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Measurement types we currently model. Kept as strings (not Enum) so the
# CrewAI agent can introduce new types in Block 4 without schema changes.
# Documented here for the agent's prompt context.
_KNOWN_MEASUREMENT_TYPES = (
    "normalized_logD",
    "logD",          # Diep et al. 2026 normalized log distribution coefficient
    "Kd",            # Dissociation constant (M)
    "Kd_app",        # Apparent dissociation constant from titration (M)
    "Kd_dimer",      # Protein dimerization Kd (M)
    "EC50",          # Half-maximal effective concentration (M)
    "Ka",            # Association constant (M^-1)
    "separation_factor",  # SF, unitless ratio
)

# 15 rare earth elements measured in the Diep et al. dataset, plus Ca for refs
# Canonical full element names accepted by the schema.
# LREs, HREs, plus reference ions (Ca, Sc) and the actinides that occasionally
# appear in LanM literature for comparison studies.
_VALID_REES = {
    # Lanthanides
    "Yttrium", "Lanthanum", "Cerium", "Praseodymium", "Neodymium",
    "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium",
    "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium",
    # Reference ions
    "Calcium", "Scandium",
    # Actinides (Mattocks 2022, Deblonde 2020 use these as comparisons)
    "Uranium", "Plutonium", "Americium", "Curium", "Thorium",
}

# Map common scientific notation aliases to canonical element names.
# Covers: symbols (La, Eu), Roman-numeral oxidation states (LaIII, Eu(III)),
# Unicode superscripts (La³⁺), ASCII charge notation (Ca2+).
# Keys are stored lowercased; the validator normalizes input before lookup.
_ELEMENT_ALIASES = {
    # Lanthanides — symbol + III variations
    "y": "Yttrium", "y3+": "Yttrium", "yiii": "Yttrium", "y(iii)": "Yttrium", "y³⁺": "Yttrium",
    "la": "Lanthanum", "la3+": "Lanthanum", "laiii": "Lanthanum", "la(iii)": "Lanthanum", "la³⁺": "Lanthanum",
    "ce": "Cerium", "ce3+": "Cerium", "ceiii": "Cerium", "ce(iii)": "Cerium", "ce³⁺": "Cerium",
    "pr": "Praseodymium", "pr3+": "Praseodymium", "priii": "Praseodymium", "pr(iii)": "Praseodymium", "pr³⁺": "Praseodymium",
    "nd": "Neodymium", "nd3+": "Neodymium", "ndiii": "Neodymium", "nd(iii)": "Neodymium", "nd³⁺": "Neodymium",
    "sm": "Samarium", "sm3+": "Samarium", "smiii": "Samarium", "sm(iii)": "Samarium", "sm³⁺": "Samarium",
    "eu": "Europium", "eu3+": "Europium", "euiii": "Europium", "eu(iii)": "Europium", "eu³⁺": "Europium",
    "gd": "Gadolinium", "gd3+": "Gadolinium", "gdiii": "Gadolinium", "gd(iii)": "Gadolinium", "gd³⁺": "Gadolinium",
    "tb": "Terbium", "tb3+": "Terbium", "tbiii": "Terbium", "tb(iii)": "Terbium", "tb³⁺": "Terbium",
    "dy": "Dysprosium", "dy3+": "Dysprosium", "dyiii": "Dysprosium", "dy(iii)": "Dysprosium", "dy³⁺": "Dysprosium",
    "ho": "Holmium", "ho3+": "Holmium", "hoiii": "Holmium", "ho(iii)": "Holmium", "ho³⁺": "Holmium",
    "er": "Erbium", "er3+": "Erbium", "eriii": "Erbium", "er(iii)": "Erbium", "er³⁺": "Erbium",
    "tm": "Thulium", "tm3+": "Thulium", "tmiii": "Thulium", "tm(iii)": "Thulium", "tm³⁺": "Thulium",
    "yb": "Ytterbium", "yb3+": "Ytterbium", "ybiii": "Ytterbium", "yb(iii)": "Ytterbium", "yb³⁺": "Ytterbium",
    "lu": "Lutetium", "lu3+": "Lutetium", "luiii": "Lutetium", "lu(iii)": "Lutetium", "lu³⁺": "Lutetium",
    # Reference ions
    "ca": "Calcium", "ca2+": "Calcium", "caii": "Calcium", "ca(ii)": "Calcium", "ca²⁺": "Calcium",
    "sc": "Scandium", "sc3+": "Scandium", "sciii": "Scandium", "sc(iii)": "Scandium", "sc³⁺": "Scandium",
    # Actinides (used in some LanM comparison studies)
    "u": "Uranium", "uvi": "Uranium", "u(vi)": "Uranium", "uo2": "Uranium",
    "pu": "Plutonium", "puiv": "Plutonium",
    "am": "Americium", "amiii": "Americium", "am(iii)": "Americium",
    "cm": "Curium", "cmiii": "Curium", "cm(iii)": "Curium",
    "th": "Thorium", "thiv": "Thorium",
}

# Group-level element labels that are NOT valid for per-element measurements.
# Records with these labels are dropped rather than failed, since the data is
# real but not at the right granularity for our schema.
_GROUP_LEVEL_LABELS = {
    "light rees", "heavy rees", "lres", "hres",
    "rare earths", "lanthanides", "actinides",
    "rare earth elements", "rees",
}

class ProteinVariant(BaseModel):
    """
    Represents a single protein construct: wild-type ortholog or engineered
    mutant. One record per unique sequence in the corpus.
    @param variant_id: Stable, human-readable identifier. Examples:
                       'o-621', 'Mex-LanM', 'Hans-LanM(R100K)', 'Mex-LanM(4D9H)'.
    @param source_organism: Full scientific name with strain when available.
    @param sequence: Mature amino acid sequence (signal peptide removed) using
                     the 20 standard one-letter codes. None when unreported.
    @param parent_variant_id: For mutants, the variant_id of the wild-type
                              parent (e.g. R100K's parent is 'Hans-LanM').
                              None for wild-type proteins.
    @param mutations: List of substitution notations like ['R100K', 'T57S'].
                      Empty list for wild-type.
    @param mutation_notation: Original paper's notation when unparseable as
                              individual substitutions (e.g. '4D9H' = D-to-H
                              at position 9 of all four EF hands). None when
                              not applicable.
    @param taxonomy: Optional taxonomic metadata (family, genus, etc.) as
                     free-form dict-style string. None when unreported.
    @param ef_hand_count: Number of EF-hand motifs (typically 3 or 4 for LanMs).
    @param selectivity_cluster: Agglomerative cluster (0-7) from Diep et al.
                                2026. None when not assigned.
    @param source_paper: Filename or DOI of the source. Used for dedup and
                         provenance.
    """

    variant_id: str = Field(..., min_length=1)
    source_organism: str = Field(..., min_length=1)
    sequence: Optional[str] = Field(default=None, min_length=10)
    parent_variant_id: Optional[str] = None
    mutations: List[str] = Field(default_factory=list)
    mutation_notation: Optional[str] = None
    taxonomy: Optional[str] = None
    ef_hand_count: Optional[int] = Field(default=None, ge=0, le=10)
    selectivity_cluster: Optional[int] = Field(default=None, ge=0, le=10)
    source_paper: str = Field(..., min_length=1)

    @field_validator("sequence")
    @classmethod
    def _sequence_uses_standard_aas(cls, value: Optional[str]) -> Optional[str]:
        """
        Validates that the sequence contains only the 20 standard amino acid
        one-letter codes after normalization. Permits None.
        @param value: The raw sequence string or None.
        return : The cleaned (stripped, uppercased) sequence, or None.
        raises : ValueError if any non-standard characters are found.
        """
        if value is None:
            return None

        cleaned = value.strip().upper()

        invalid = set(cleaned) - _VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(
                f"Sequence contains invalid characters: {sorted(invalid)}"
            )

        return cleaned


class BindingMeasurement(BaseModel):
    """
    Represents a single experimental measurement of a protein variant
    interacting with one element. One row per (variant_id, target_element,
    measurement_type) triple. Joining BindingMeasurement to ProteinVariant
    on variant_id produces the ML-ready training table.
    @param variant_id: Foreign key to ProteinVariant.variant_id.
    @param target_element: Full element name (e.g. 'Neodymium').
    @param measurement_type: Type of measurement, e.g. 'logD', 'Kd', 'Kd_app'.
                             See _KNOWN_MEASUREMENT_TYPES for the canonical set.
    @param value: Numeric measurement value in the raw paper units.
    @param units: Unit string as reported (e.g. 'M', 'pM', 'unitless').
    @param value_in_molar: Normalized value in molar units when applicable.
                           None for unitless measurements (logD, SF).
    @param conditions_pH: Reported buffer pH. None when unreported.
    @param conditions_notes: Free-text additional conditions (technique,
                             temperature, replicates). None when unreported.
    @param value_source_type: 'primary' for measurements published in the
                              cited paper itself, 'cited_from_earlier_work'
                              when the paper reproduces values from another
                              source.
    @param source_paper: Filename or DOI of the source.
    """

    variant_id: str = Field(..., min_length=1)
    target_element: str = Field(..., min_length=1)
    measurement_type: str = Field(..., min_length=1)
    value: float
    units: str = Field(..., min_length=1)
    value_in_molar: Optional[float] = Field(default=None, gt=0, lt=1e-2)
    conditions_pH: Optional[float] = Field(default=None, gt=0, lt=14)
    conditions_notes: Optional[str] = None
    value_source_type: str = Field(default="primary")
    source_paper: str = Field(..., min_length=1)

    @field_validator("target_element")
    @classmethod
    def _element_is_known(cls, value: str) -> str:
        """
        Normalizes element notation to canonical full name. Accepts:
          - Full names: 'Neodymium', 'neodymium'
          - Symbols: 'Nd'
          - Oxidation-state forms: 'NdIII', 'Nd(III)', 'Nd3+', 'Nd³⁺'
        Rejects group-level labels ('Light REEs') with a clear message
        so the caller can drop those records without crashing the run.
        @param value: The element label as supplied.
        return : The canonicalized element name.
        raises : ValueError when the element is not recognized.
        """
        raw = value.strip()
        lookup_key = raw.lower()

        # First, try canonical name match (already in target form)
        title_cased = raw.title()
        if title_cased in _VALID_REES:
            return title_cased

        # Group-level labels: explicit failure with clear cause
        if lookup_key in _GROUP_LEVEL_LABELS:
            raise ValueError(
                f"target_element {value!r} is a group-level label, "
                f"not a specific element. Skip this measurement or "
                f"split it into per-element records."
            )

        # Alias lookup
        canonical = _ELEMENT_ALIASES.get(lookup_key)
        if canonical is not None:
            return canonical

        raise ValueError(
            f"Unknown target_element {value!r}. "
            f"Expected an element name (e.g. 'Neodymium'), symbol "
            f"('Nd'), or oxidation-state notation ('NdIII', 'Nd³⁺')."
        )
    @field_validator("value_source_type")
    @classmethod
    def _source_type_is_known(cls, value: str) -> str:
        """
        Validates that the value_source_type uses the canonical vocabulary.
        @param value: The source-type string as supplied.
        return : The validated source-type string.
        raises : ValueError if the value is not one of the known types.
        """
        allowed = {"primary", "cited_from_earlier_work"}
        if value not in allowed:
            raise ValueError(
                f"value_source_type must be one of {sorted(allowed)}, got {value!r}"
            )

        return value


class CorpusRecords(BaseModel):
    """
    Top-level container for the full extracted corpus. The XLSX loader (Diep
    et al. 2026) and the CrewAI agent (literature annotation, Block 4) both
    produce CorpusRecords objects that can be concatenated by the assembly
    step in Block 5.
    @param variants: List of ProteinVariant records.
    @param measurements: List of BindingMeasurement records.
    """

    variants: List[ProteinVariant] = Field(default_factory=list)
    measurements: List[BindingMeasurement] = Field(default_factory=list)

    def __len__(self) -> int:
        """
        return : Total record count across both tiers.
        """
        return len(self.variants) + len(self.measurements)
