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
- **Bioinformatics:** Biopython (sequence feature engineering)
- **ML:** scikit-learn, XGBoost
- **App / MLOps:** Streamlit, Docker
- **Dev:** pytest (with branch coverage), flake8, pylint

## Data Sources

### Primary dataset

**Diep et al. 2026 — A family portrait of lanmodulin selectivity for enhanced
rare-earth separations.** *Nature Chemical Biology* 22, 829–839.
DOI: [10.1038/s41589-026-02176-3](https://doi.org/10.1038/s41589-026-02176-3)

Supplementary Data 1 from this paper provides a high-throughput selectivity
dataset of 616 LanM orthologs across 15 rare earth elements, including full
amino acid sequences, source organism taxonomy, raw ICP-MS measurements (3
replicates), normalized logD values, and 8 agglomerative selectivity clusters.
This is the project's main ML training set (~9,240 variant × element records).

### Literature annotation corpus

15 peer-reviewed papers on Lanmodulin engineering and characterization, used
by the CrewAI agent to enrich the primary dataset with published binding
constants (Kd, Kd_app), engineered mutations, and coordination chemistry
notes. Papers were pre-processed with Gemini 2.5 Pro to extract verbatim
relevant passages prior to LLM extraction. See `agentic_ai/inputs/processed/`.

### Data not redistributed

Source PDFs and XLSX supplementary files are copyrighted and excluded from
git. They are downloadable from the cited publications.

## Project Structure


agentic_ai/                  # Main package
├── agents/                  # CrewAI agent definitions
├── inputs/                  # Source corpus
│   └── processed/           # Gemini-curated text extracts (committed)
├── schemas.py               # Pydantic data models
├── main.py                  # Entry point
└── utils/                   # Helpers (env, extraction, loaders)
data/
├── raw/supplementary/       # Author-published datasets (NOT in git)
└── processed/               # Cleaned ML-ready CSVs (committed)
models/                      # Trained .pkl files (committed)
notebooks/                   # EDA + training experiments
tests/                       # pytest suite (branch coverage)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OpenAI API key
```

## Reproducing the project

To reproduce the full pipeline you need the supplementary data:

1. Download `41589_2026_2176_MOESM3_ESM.xlsx` from the Diep et al. 2026
   paper (Nature Chemical Biology) → place in `data/raw/supplementary/`
2. Run `python -m agentic_ai.main` to verify environment + API connectivity
3. Run `pytest` to confirm the test suite passes
4. (Further steps documented as project progresses through Weeks 2-5)

## Project Status

- [x] Week 1 Block 1 — Environment, API connectivity, project scaffold
- [x] Week 1 Block 2 — Pydantic schema + LLM extraction proof-of-concept
- [ ] Week 1 Block 3 — XLSX loader (MOESM3) + text reader + Path C two-tier schema
- [ ] Week 1 Block 4 — CrewAI agent for literature annotation enrichment
- [ ] Week 1 Block 5 — DataFrame assembly + validation
- [ ] Week 2 — Sequence feature engineering (Biopython)
- [ ] Week 3 — ML training (XGBoost, hyperparameter tuning)
- [ ] Week 4 — Streamlit application
- [ ] Week 5 — Docker + cloud deployment

## License & Citation

Code: MIT (see LICENSE).
Data: cite Diep et al. 2026 (DOI above) and follow the licenses of each
linked publication.
