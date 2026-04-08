# Eagle Blend Optimizer

**Eagle Blend Optimizer** is Eagle Team's fuel-blending intelligence platform for the **Shell AI Global Hackathon 2025**. The project won the **Phase 1 leaderboard round** against **7K+ participants** and went on to finish among the **Top 5 presenters in Phase 2**.

The repository formulates fuel-blend design as a constrained surrogate-modeling and multi-objective optimization problem defined over a **55-dimensional blend representation** comprising **5 component fractions** and **50 component-property descriptors**, with regression targets spanning **10 blend properties**. The production inference path uses a **scaler -> PCA -> XGBoost** stack for multivariate property prediction, while the optimization layer solves a **Pareto-constrained search problem** that minimizes squared property deviation and, when enabled, formulation cost under simplex constraints and frozen-property tolerances via **NSGA-II** with a custom **normalization repair operator**.

Beyond the production predictor, the research stack explores **hybrid target-wise modeling** with **TabPFN ensembles**, polynomial regressors, and structured nonlinear feature transformations. In particular, the project develops explicit mixture-feature operators over weighted component properties, including higher-order polynomial expansions, pairwise and triple interaction terms, symmetric polynomial statistics, and nonlinear transforms such as log, exponential, trigonometric, and weighted moment features to better capture non-additive blending behavior.

The experimentation layer further incorporates **synthetic-data augmentation** through **TVAE-based generative modeling** for continuous blend/property manifolds, alongside additional synthetic baselines, to expand coverage of sparse composition regimes and stress-test downstream predictive generalization. These modeling and optimization components are operationalized through a **Streamlit-based analytical interface** and a **SQLite-backed registry** for components, blends, activity traces, and model-quality metadata, making the repository both a research artifact and a deployable decision-support system.

## Competition Context

- **Team Name:** Eagle Team
- **Team Lead:** Destiny Otto
- **Team Members:** Alexander Ifenaike, Williams Alabi, Godswill Otto
- **Competition:** Shell AI Global Hackathon 2025
- **Phase 1:** Winner of the leaderboard-style qualification round among 7K+ participants
- **Phase 2:** Selected among the Top 5 final presenters

Competition references:

- [Shell AI Level 2 2025 participant leaderboard](https://shellai-level2-2025.hackerearth.com/challenges/hackathon/shellai-level2-2025/participants/#participants)
- [Shell AI Hackathon 2025 challenge page](https://www.hackerearth.com/challenges/competitive/shellai-hackathon-2025/instructions/)

## Why This System Is Interesting

Fuel-blend formulation is a coupled prediction-and-search problem. A candidate blend is defined by:

- **5 component fractions**
- **10 properties per component**
- a resulting **55-dimensional input representation**
- and **10 target blend properties** that must be inferred or optimized jointly

The repository tackles this with a layered approach:

1. **Predict** final blend behavior from component-level composition.
2. **Search** the combinatorial design space for feasible, high-quality, low-cost blends.
3. **Persist and compare** blends inside a product interface rather than leaving the work in notebooks.
4. **Research beyond the production model** using TabPFN-driven hybrid modeling for hard-to-fit targets.

In short, the repo operationalizes a competition solution into a reusable blend intelligence system.

## Technical Summary

### Input and output contract

The canonical blend schema used by the product consists of:

- `Component1_fraction` to `Component5_fraction`
- `Component{i}_Property{j}` for `i in [1..5]`, `j in [1..10]`
- predicted outputs `BlendProperty1` to `BlendProperty10`

That gives the production predictor a **55-feature input tensor**:

```text
5 mixture fractions
+ 50 component property features
= 55 model inputs
```

This same schema powers:

- manual blend design,
- batch CSV prediction,
- optimization setup,
- database persistence,
- and comparison workflows.

## System Architecture

### 1. Product runtime

The active application entrypoint is:

```bash
streamlit run apps/streamlit/streamlit_app.py
```

The current repository follows a product-first structure:

- `apps/streamlit/` for the live application surface
- `src/eagle_blend/` for package-owned code
- `data/` for templates, reference data, and the seeded SQLite database
- `models/production/` for runtime model artifacts
- `research/` for experimental modeling and notebooks
- `archive/` for preserved lineages and safety snapshots

This layout is intentional: the repo separates production paths from experimentation while preserving historical work instead of deleting it.

### 2. Production prediction pipeline

The canonical runtime predictor lives in `src/eagle_blend/ml/predictor_xgb.py`.

Production inference is built around:

- a persisted **scaler**
- a persisted **PCA transform**
- a persisted **XGBoost model**

The runtime flow is:

```text
55-D component/blend feature vector
-> scaler.transform(...)
-> pca.transform(...)
-> xgboost model.predict(...)
-> BlendProperty1..BlendProperty10
```

Two prediction paths are exposed:

- `predict_all(...)` for all 10 blend properties
- `predict_fast(...)` for a reduced set of high-value properties:
  - `BlendProperty1`
  - `BlendProperty2`
  - `BlendProperty5`
  - `BlendProperty6`
  - `BlendProperty7`
  - `BlendProperty10`

That fast path is a key design decision: it reduces optimization-loop cost while preserving a full prediction route for downstream UI use.

### 3. Optimization engine

The optimization logic is implemented in both the modular path `src/eagle_blend/optimization/nsga2.py` and the live app workflow in `apps/streamlit/streamlit_app.py`.

The optimizer models blend design as a constrained search problem over **5 decision variables**:

- each variable is a component fraction,
- each fraction is bounded in `[0, 1]`,
- and the fractions must sum to exactly `1`.

The solver uses **NSGA-II** from `pymoo` with a custom repair operator:

- **NormalizationRepair** enforces the sum-to-one constraint after mutation/crossover
- this avoids clumsy penalty-only treatment of a core physical constraint
- duplicates are eliminated to maintain a cleaner candidate frontier

The optimization objective is:

```text
error(x) = sum((predicted_properties - target_properties)^2)
```

Optional cost-aware optimization adds:

```text
cost(x) = fractions dot component_costs
```

So the system can operate in either:

- **single-objective mode**: minimize property error only
- **multi-objective mode**: minimize property error and cost simultaneously

Additional technical details:

- hard constraints can be applied to the fast-property subset `{1, 2, 5, 6, 7, 10}`
- constraint violation tolerance is `1e-3`
- optimization progress is streamed back into the Streamlit UI via callback-driven progress bars
- final solutions are surfaced as a **Pareto front** of error-cost trade-offs

### 4. Product data layer

The application persists state in `data/seed/eagleblend.db`, a SQLite database with the following key tables:

- `components`
- `blends`
- `activity_log`
- `models_registry`

This is not an incidental storage layer. It is central to the app:

- the **Fuel Registry** reads and writes component and blend records,
- the **Dashboard** summarizes recent blends and logged activity,
- the **Model Insights** tab reads from `models_registry`,
- and the **Optimization Engine** can persist selected solutions for later comparison.

The seeded database snapshot currently contains:

- **10,070 component rows**
- **2,524 blend rows**
- **45 activity log events**
- **2 registered model entries**

The saved blend schema also stores optimization-aware fields such as:

- `PreOpt_Cost`
- `Optimized_Cost`
- `Quality_Score`
- `activity`
- `created_at`

### 5. Research: TabPFN hybrid modeling

One of the strongest parts of the repo is that it does not stop at the production XGBoost path. Under `research/tabpfn_hybrid/`, the team explored a richer hybrid modeling strategy built around **TabPFN**, feature-engineered regressors, and property-specific post-processing.

The research stack includes:

- a **5-fold TabPFN ensemble** loader in `research/tabpfn_hybrid/code/inference.py`
- model download automation via Hugging Face in `research/tabpfn_hybrid/code/download_models.py`
- property-specific inference logic in `research/tabpfn_hybrid/code/predictor.py`
- notebooks for training, inference, and experimentation

The hybrid path is especially interesting because it is **not** a monolithic one-model-for-all-targets design. Different targets are handled differently:

- **BlendProperty1, 2, 6, 10** use linear models with polynomial feature transforms
- **BlendProperty5 and 7** use dedicated TabPFN-based predictors
- **BlendProperty3, 4, 8, 9** are inferred through a TabPFN ensemble predictor
- **BlendProperty8** receives a piecewise correction around low-magnitude regions
- **BlendProperty7** uses custom nonlinear mixture feature generation instead of a trivial weighted average

That mixture-feature generator is a serious modeling idea on its own. It constructs:

- weighted sums,
- weighted powers,
- weighted `tanh`, `log`, `exp`, `sin`, and `cos` transforms,
- pairwise and triple interactions,
- symmetric polynomial combinations,
- weighted differences,
- entropy-like and normalization-aware descriptors,
- and other interaction terms designed to capture nonlinear blending behavior.

This research track is exactly the kind of technical depth that separates a polished competition product from a shallow dashboard wrapper.

## What the Product Can Do

### Dashboard

The dashboard acts as the operational control room for the application. It surfaces:

- recent blend activity,
- aggregate usage logs,
- model quality indicators,
- and cost-saving signals from stored optimization runs.

### Blend Designer

The designer supports two major workflows:

- **manual design**, where users define up to five components and receive instant predictions
- **batch design**, where a CSV of many candidate blends is scored in one pass

The batch interface validates the 55-column feature structure and returns an output table with appended predicted blend properties.

### Optimization Engine

The optimizer lets users:

- define target blend properties,
- optionally freeze selected fast properties as hard constraints,
- include cost as a competing objective,
- generate multiple high-quality trade-off solutions,
- visualize the Pareto frontier,
- and save chosen candidates back into the database.

### Fuel Registry

The registry is the central data management layer for:

- component ingestion,
- blend ingestion,
- CSV template downloads,
- search,
- and record deletion.

Blend uploads can be expanded into component-level rows so both granular and aggregate views stay aligned.

### Blend Comparison

The comparison workspace supports side-by-side evaluation of saved blends using multiple visual forms, including:

- cost comparison,
- radar-style property comparison,
- composition views,
- and combined multi-chart summaries

### Model Insights

The model-insights surface reads from the database-backed model registry and exposes:

- overall model quality,
- error indicators,
- and per-property score breakdowns across the 10 blend targets

## Quantitative Snapshot

The latest model entry in `models_registry` reports:

- **Model name:** `v2_2025-08-15_08-00-00`
- **Overall R^2:** `0.92`
- **MSE:** `0.01`

Per-property scores in the seeded model registry:

| Blend property | Score |
| --- | ---: |
| BlendProperty1 | 0.99 |
| BlendProperty2 | 0.97 |
| BlendProperty3 | 0.88 |
| BlendProperty4 | 0.98 |
| BlendProperty5 | 0.92 |
| BlendProperty6 | 0.95 |
| BlendProperty7 | 0.85 |
| BlendProperty8 | 0.89 |
| BlendProperty9 | 0.90 |
| BlendProperty10 | 0.96 |

Additional seeded-database signals:

- the app stores **2,524** blend records and **10,070** component records
- the saved optimization records capture up to **11.98** units of cost reduction in the current demo snapshot
- the activity log shows prediction, save, optimization, and save-optimization events already wired into the product

These numbers are important because they show the repo is not an empty UI shell. It contains persisted operational state, model metadata, and a working product loop.

## Repository Layout

```text
.
|-- apps/
|   `-- streamlit/                  # Canonical app entrypoint
|-- src/
|   `-- eagle_blend/                # Package-owned runtime code
|-- data/
|   |-- reference/                  # Static reference tables
|   |-- samples/                    # Example batch input
|   |-- seed/                       # SQLite product database
|   `-- templates/                  # Downloadable CSV templates
|-- models/
|   |-- production/xgboost/         # Production inference assets
|   `-- experimental/xgboost/       # Experimental model artifacts
|-- research/
|   |-- experiments/                # Optimization, synthetic data, DB notebooks
|   `-- tabpfn_hybrid/              # TabPFN hybrid research track
|-- docs/
|   |-- architecture/               # Technical notes
|   |-- diagrams/                   # Optimization flow diagrams
|   `-- submissions/                # Submission collateral
|-- deploy/                         # Deployment collateral
|-- archive/                        # Preserved legacy lineages and snapshots
`-- tests/                          # Smoke checks and future regression tests
```

## Getting Started

### Prerequisites

- Python `>= 3.10`
- a local environment capable of running Streamlit and common scientific Python packages

### Install the canonical app

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Run the application

```bash
streamlit run apps/streamlit/streamlit_app.py
```

### Core runtime assets expected by the app

- `data/seed/eagleblend.db`
- `models/production/xgboost/xmodel.joblib`
- `models/production/xgboost/scaler.joblib`
- `models/production/xgboost/pca.joblib`

### Batch prediction template

Use:

- `data/templates/batch_template.csv`

to prepare valid batch input with the full 55-column feature layout.

## Research Environment

The TabPFN hybrid work has its own dependency path:

```bash
pip install -r research/tabpfn_hybrid/code/requirements.txt
```

To download the research models used by that track:

```bash
cd research/tabpfn_hybrid/code
python download_models.py
```

That script pulls:

- fold models from the Hugging Face repository `willieseun/Eagle-Team-TabPFN`
- additional snapshot assets from `wayne-chi/Eagle_Team`

## Engineering Notes

- The current repo preserves historical lineages under `archive/` rather than deleting them.
- Smoke tests verify canonical repo zones and path wiring.
- The product runtime is already separated from research and archival material.
- The live Streamlit app is still substantial and monolithic, which makes this repo a good candidate for further modular extraction into `src/eagle_blend/ui/` and service-level modules.

## Why This Repo Matters

Many hackathon repositories stop at notebooks or a thin front-end layer. This one does more:

- it preserves the experimental path,
- packages a production inference path,
- exposes optimization and comparison through an application surface,
- tracks model quality inside the product,
- and stores operational data in a reusable database.

That combination of modeling depth, optimization logic, productization, and research preservation is the reason this repo deserves a technical README instead of a one-line project blurb.

## Team

**Eagle Team**

- **Destiny Otto** - Team Lead
- **Alexander Ifenaike**
- **Williams Alabi**
- **Godswill Otto**

## Canonical Paths

- App entrypoint: `apps/streamlit/streamlit_app.py`
- Package root: `src/eagle_blend/`
- Production predictor: `src/eagle_blend/ml/predictor_xgb.py`
- Optimization module: `src/eagle_blend/optimization/nsga2.py`
- Path configuration: `src/eagle_blend/config/paths.py`
- Product database: `data/seed/eagleblend.db`
- Templates: `data/templates/`
- TabPFN research track: `research/tabpfn_hybrid/`
