# REE-Binding Protein Selectivity Predictor

End-to-end machine learning project that predicts rare earth element (REE)
selectivity of Lanmodulin (LanM) protein orthologs from amino acid sequence
features. Combines a published high-throughput selectivity dataset (616 orthologs
× 15 REEs) with literature-derived binding-constant annotations extracted by a
CrewAI agent from peer-reviewed papers.

**Live demo**: [reeproteins.streamlit.app](https://reeproteins.streamlit.app/)

## Project Goal

Build a sequence-to-selectivity predictor that can:
1. Take a protein amino acid sequence as input
2. Predict its selectivity profile across 15 rare earth elements
3. Help R&D teams in biomining/separation prioritize variants for wet-lab testing

## Tech Stack

- **Data pipeline:** pandas, Pydantic, CrewAI, OpenAI API
- **Bioinformatics:** Biopython sequence feature engineering
- **ML:** scikit-learn, XGBoost (planned for Week 3)
- **App / MLOps:** Streamlit, Docker (planned for Weeks 4–5)
- **Dev:** pytest, flake8, pylint

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
The literature dataset is a heterogeneous validation corpus containing
binding constants, EC50 values, relaxivity measurements, and process metrics.
Its `measurement_family` column separates scientifically compatible subsets.
After deterministic unit normalization, 60 of the 88 records have molar values.
See `data/processed/extractions/`.

### Crucial semantic distinction

The two datasets measure different quantities:

| Source | Measures | Units | Use |
|---|---|---|---|
| MOESM3 | Normalized log distribution coefficient | unitless (0–1) | Primary training target |
| Literature | Heterogeneous published measurements | M, nM, unitless, etc. | Filtered orthogonal validation |

These cannot be merged into a single target column without corrupting the
science. Each lives in its own CSV with its own column schema.

### Data not redistributed

Source PDFs and XLSX supplementary files are copyrighted and excluded from
git. They are downloadable from the cited publications.

## Local App Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For full local development, including tests, data processing, and
literature extraction agents:

```bash
pip install -r requirements-dev.txt
```

## Reproducing the Full Project

1. Download `41589_2026_2176_MOESM3_ESM.xlsx` from Diep et al. 2026
   (Nature Chemical Biology) and place it in
   `data/raw/supplementary/`.

2. Install development dependencies:

```bash
   pip install -r requirements-dev.txt
```

3. Optional: create `.env` from `.env.example` and add an OpenAI API key
   if re-running literature extraction.

4. Run the test suite:

```bash
   pytest -v
```
5. Optional: re-run literature extraction:

```bash
   python -m agentic_ai.agents.corpus_runner --save
```

6. Re-assemble source-specific CSV datasets:

```bash
   python -m agentic_ai.loaders.dataset_assembly
```

7. Build the ML-ready feature matrix:

```bash
   python -m agentic_ai.features.build_matrix
```

8. Train the baseline model:

```bash
   python -m agentic_ai.models.train
```


## Performance on held-out test set

| Metric | Value |
|---|---|
| Macro per-element Spearman | **0.557** (rank candidate proteins for a target REE) |
| Macro per-variant Spearman | **0.938** (recover a protein's selectivity profile) |
| R² | **0.851** |
| RMSE | 0.128 (target range 0–1) |

## Dataset Summary

### `moesm3_selectivity_data.csv` — Primary Training Data
- 9,240 rows, 11 columns
- 616 unique LanM orthologs × 15 REEs
- Target column: `value` (normalized_logD, 0–1 unitless)
- Sequences inline for direct ML tokenization

### `literature_binding_data.csv` — Validation Corpus
- 88 rows, 17 columns
- 27 unique variants across 13 REEs from 15 papers
- Mixed `value_type`: Kd, Kd_app, Binding, logD, and process metrics
- 60 records have parseable `value_in_molar`
- 56 records are marked as direct molar affinity-validation candidates
- `measurement_family` must be used to select compatible validation targets

### `ml_ready_features.parquet` — Baseline Training Matrix
- 9,240 rows, 128 model features, 6 metadata columns
- Train/test assignment grouped by variant and stratified by selectivity cluster
- Companion artifacts:
  - `ml_ready_features_schema.json`
  - `ml_ready_features_encoder.pkl`
 
## Key Findings

### Preventing Target Corruption

An early design assumed MOESM3 values represented log10(Kd) and could
be exponentiated to molar units. Empirical inspection showed values in
the 0.028–1.0 range with `measurement_type='normalized_logD'`, matching
the Diep et al. supplementary-methods description. The project preserves
MOESM3 and literature affinity measurements as separate target families.

### Hyperparameter Tuning Result

Randomized XGBoost tuning selected a model with slightly better RMSE and
R² but slightly worse macro per-element Spearman than the baseline.
Because macro per-element Spearman was the pre-specified headline metric,
the baseline model was deployed.

### Samarium Confidence Caveat

Samarium had unusually weak per-element ranking performance. Follow-up
diagnostics showed severe target saturation: 44.1% of training orthologs
have Sm normalized logD = 1.0, and 81.5% have Sm ≥ 0.95. This limits
recoverable rank information and is shown as a reduced-confidence caveat
in the app.

### Cross-Assay Concordance

Exploratory comparison against literature in-solution Kd measurements
showed partial directional concordance with important rank-order
differences. The app therefore describes outputs strictly as normalized
on-resin selectivity predictions, not binding affinities.

## Project Status

- [x] **Week 1** — Data pipeline
- [x] **Week 2** — Baseline feature engineering
- [x] **Week 3** — ML training and model selection
- [x] **Week 4** — Streamlit application
- [x] **Week 5** — Cloud deployment

## License & Citation

Code: MIT (see LICENSE).
Data: cite Diep et al. 2026 (DOI above) and follow the licenses of each
linked publication.
