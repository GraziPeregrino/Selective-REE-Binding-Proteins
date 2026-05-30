# REE-Binding Protein Selectivity Predictor

End-to-end machine learning project that predicts rare earth element (REE)
selectivity of Lanmodulin (LanM) protein orthologs from amino acid sequence
features. Combines a published high-throughput selectivity dataset (616 orthologs
× 15 REEs) with literature-derived binding-constant annotations extracted by a
CrewAI agent from peer-reviewed papers.

## Project Goal

Build a sequence-to-selectivity predictor that can:
1. Take a protein amino acid sequence as input
2. Predict its selectivity profile across 15 rare earth elements
3. Help R&D teams in biomining/separation prioritize variants for wet-lab testing

## Tech Stack

- **Data pipeline:** pandas, Pydantic, CrewAI, OpenAI API
- **Bioinformatics:** Biopython (sequence feature engineering — planned for Week 2)
- **ML:** scikit-learn, XGBoost (planned for Week 3)
- **App / MLOps:** Streamlit, Docker (planned for Weeks 4–5)
- **Dev:** pytest (210 tests passing), flake8, pylint

## Data Sources

### Primary dataset

**Diep et al. 2026 — A family portrait of lanmodulin selectivity for enhanced
rare-earth separations.** *Nature Chemical Biology* 22, 829–839.
DOI: [10.1038/s41589-026-02176-3](https://doi.org/10.1038/s41589-026-02176-3)

Supplementary Data 1 (MOESM3) is a high-throughput selectivity screen of 616
LanM orthologs across 15 rare earth elements, generated via the SpyCI-LAMBS
ICP-MS assay. The dataset reports **normalized logD selectivity scores**
(unitless, 0–1) — the per-element log distribution coefficient divided by the
maximum per-replicate. This is the project's primary ML training target
(9,240 variant × element records).

### Literature annotation corpus

15 peer-reviewed papers on Lanmodulin engineering and characterization,
processed by a CrewAI agent into a structured corpus of 89 binding
measurements across 27 unique protein variants. Papers were pre-processed
with Gemini 2.5 Pro to extract relevant passages prior to LLM extraction.
The literature dataset reports **actual binding constants** (Kd, Kd_app,
EC50, etc.) in mass-action units, which makes 32 of the 89 records directly
comparable to molar Kd values. See `data/processed/extractions/`.

### Crucial semantic distinction

The two datasets measure different quantities:

| Source | Measures | Units | Use |
|---|---|---|---|
| MOESM3 | Normalized log distribution coefficient | unitless (0–1) | Primary training target |
| Literature | Thermodynamic dissociation constants | M, nM, etc. | Orthogonal validation |

These cannot be merged into a single target column without corrupting the
science. Each lives in its own CSV with its own column schema.

### Data not redistributed

Source PDFs and XLSX supplementary files are copyrighted and excluded from
git. They are downloadable from the cited publications.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OpenAI API key
```

## Reproducing the project

1. Download `41589_2026_2176_MOESM3_ESM.xlsx` from Diep et al. 2026
   (Nature Chemical Biology) → place in `data/raw/supplementary/`
2. Verify environment: `python -c "from agentic_ai.utils.env_check import load_api_key; print('API key loaded:', load_api_key()[:8] + '...')"`
3. Run the test suite: `pytest -v` (expect 210 passing)
4. (Optional, ~$0.02 OpenAI cost) Re-run the literature extraction:
   `python -m agentic_ai.agents.corpus_runner --save`
5. Re-assemble the CSV datasets: `python -m agentic_ai.loaders.dataset_assembly`

## Dataset Summary

After Block 5, the project produces two source-specific datasets:

### `moesm3_selectivity_data.csv` — Primary training data
- 9,240 rows, 11 columns
- 616 unique LanM orthologs × 15 REEs
- Target column: `value` (normalized_logD, 0–1 unitless)
- Sequences inline for direct ML tokenization

### `literature_binding_data.csv` — Validation cohort
- 89 rows, 13 columns
- 27 unique variants across 13 REEs from 15 papers
- Mixed `value_type`: Kd (45), Kd_app (12), EC50 (10), fold_change (4), logD (4), other
- 32 records have parseable `value_in_molar` for direct Kd comparison
- Construct types: ortholog (47), engineered_chelator (28), point_mutant (10), fusion_sensor (4)
- Parent scaffolds: Lanmodulin, Lanmodulin+GFP, Calmodulin, lanpepsy, de_novo, non_LanM_protein

## Key findings from Week 1

**Catching a misclassification.** During literature classification,
the agent surfaced "MIF" — a multimetal ion-stacking metalloprotein
framework from biorxiv 2025.10.21.683075. Initial classification assumed
Lanmodulin lineage. Investigation of the source paper revealed MIF is
derived from **lanpepsy** (a PepSY-domain protein from *Methylobacillus
flagellatus*), making it a structurally distinct β-barrel REE binder with
~4.3 Å inter-metal spacing — architecturally unlike LanM's EF-hand
chemistry. The classifier preserves this distinction in `parent_scaffold`.

**Preventing a silent corruption.** Pre-Block-5 design assumed MOESM3's
measurement column was `log10(Kd)` to be exponentiated to molar.
Empirical inspection showed values in the 0.028–1.0 range with
`measurement_type='normalized_logD'`, matching the Diep et al.
Supplementary Methods description of "logD normalized to the
per-replicate maximum." The semantic mismatch with literature Kd was
preserved in the schema, preventing silent corruption of 9,240 records.

**Recovering wrongly-dropped data.** An initial drop list treated "EF1",
"EF2", "EF3", "EF4" as agent confusion (motif names mistaken for
protein names). Corpus inspection revealed Gutenthaler 2022 measured
isolated 12-residue EF-hand peptides as standalone REE-binding
constructs, with real micromolar Kd values and Gd relaxivity data.
These were promoted from the drop list to `engineered_chelator` /
`Lanmodulin` classifications, recovering 17 measurements.

## Project Status

- [x] **Week 1** — Data pipeline (Blocks 1–5 complete)
    - [x] Block 1 — Environment, API connectivity, project scaffold
    - [x] Block 2 — Pydantic schema + LLM extraction proof-of-concept
    - [x] Block 3 — XLSX loader (MOESM3) + text reader + unified pipeline
    - [x] Block 4 — CrewAI corpus runner + classifier + persisted enrichment
        - [x] 4.1–4.2 — Agent + task + deterministic unit conversion
        - [x] 4.3 — Corpus orchestrator + JSON persistence (15/15 papers, $0.02)
        - [x] 4.4 — Three-tier variant classifier (alias map + construct types + drop rules)
        - [x] 4.5 — Tests for extraction_io, corpus_runner, env_check
    - [x] Block 5 — Dataset assembly: two source-specific long-form CSVs
- [ ] **Week 2** — Sequence feature engineering (Biopython)
- [ ] **Week 3** — ML training (XGBoost, hyperparameter tuning)
- [ ] **Week 4** — Streamlit application
- [ ] **Week 5** — Docker + cloud deployment

## License & Citation

Code: MIT (see LICENSE).
Data: cite Diep et al. 2026 (DOI above) and follow the licenses of each
linked publication.
