# Eagle Blend Optimizer

This repository now follows a product-first structure built around the canonical Streamlit application at `apps/streamlit/streamlit_app.py`.

## Canonical entrypoint

Run the application from the repository root with:

```bash
streamlit run apps/streamlit/streamlit_app.py
```

## Repository map

```text
.
├── apps/               # user-facing application entrypoints
├── src/                # product code and importable modules
├── data/               # runtime database, templates, and reference CSVs
├── models/             # production and experimental model artifacts
├── research/           # notebooks, experiments, and imported bundles
├── docs/               # architecture notes, diagrams, and submission collateral
├── deploy/             # deployment assets such as Docker and Hugging Face docs
├── archive/            # preserved lineages, prototypes, caches, and safety snapshots
└── tests/              # smoke and regression checks
```

## Structure policy

- `apps/streamlit/streamlit_app.py` is the active application entrypoint.
- `src/eagle_blend` is the active Python import root.
- Runtime assets live under `data/` and `models/production/`.
- Research work stays under `research/`.
- Historical lineages and non-canonical artifacts are preserved under `archive/` rather than left at the repo root.

## Preservation policy

No file lineages were discarded during the restructure.

- The original `App/` workspace is preserved at `archive/legacy_lineages/App_workspace`.
- The original `dev/` workspace is preserved at `archive/legacy_lineages/dev_workspace`.
- The temporary root safety copies created during the restructure are preserved at `archive/legacy_snapshots/root_restore_safeguard_2026_03_31`.

## Important paths

- App entrypoint: `apps/streamlit/streamlit_app.py`
- Path configuration: `src/eagle_blend/config/paths.py`
- Production predictor: `src/eagle_blend/ml/predictor_xgb.py`
- Optimization module: `src/eagle_blend/optimization/nsga2.py`
- Seed database: `data/seed/eagleblend.db`
- Templates: `data/templates/`
- Production model artifacts: `models/production/xgboost/`
