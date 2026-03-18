# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Context

Kaggle Playground Series S6E3 — binary classification (customer churn). Evaluation metric is **ROC-AUC**. Submission format: `id, Churn` where `Churn` is a float probability.

## Environment Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Running Notebooks

```bash
# From the project root
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --output notebooks/02_modeling.ipynb
```

## Architecture

### Data Flow
- Raw CSVs live in `data/raw/` (not committed — see `.gitignore`)
- Processed outputs (figures, `submission.csv`) go to `data/processed/`
- Trained model pipelines are saved as `.pkl` files in `src/models/`

### Key Data Quirks
- `TotalCharges` is stored as a string with whitespace for `tenure=0` rows — `load_data()` coerces to numeric, producing `NaN`s that are median-imputed in the pipeline
- `Churn` in `train.csv` is `"Yes"/"No"` — always pass through `encode_target()` before modeling
- `SeniorCitizen` is already `0/1`; all other binary columns are `"Yes"/"No"`

### `src/preprocessing.py`
Single source of truth for feature definitions (`BINARY_COLS`, `MULTI_CAT_COLS`, `NUMERIC_COLS`, `ALL_FEATURE_COLS`). The `build_preprocessor()` function returns a fitted-ready `ColumnTransformer`. If you add or rename features, update these lists here — both notebooks import from this module.

### `src/model_utils.py`
Contains the cost parameters (`COST_FN`, `COST_FP`, `VALUE_TP`) used in the business cost analysis sections. Adjust these constants to reflect real-world retention economics before any deployment or business presentation.

### Notebooks
- `01_eda.ipynb` — imports from `src/preprocessing` via `sys.path.append('..')`
- `02_modeling.ipynb` — full pipeline: trains 3 models, CV evaluation, submission generation, model serialisation. The "best model" is determined dynamically by CV AUC ranking.
