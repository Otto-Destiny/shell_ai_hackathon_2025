# Repository Structure

## Active product zones

- `apps/streamlit`: the canonical Streamlit delivery surface.
- `src/eagle_blend`: package-owned code, paths, services, database helpers, and future tab extraction targets.
- `data`: runtime state, templates, and reference data used by the product.
- `models/production`: model artifacts used by the product runtime.

## Research zones

- `research/tabpfn_hybrid`: notebooks, inference code, and imported model bundles for the TabPFN and hybrid experimentation stream.
- `research/experiments`: optimization notebooks, synthetic-data work, and database exploration notebooks.

## Preservation zones

- `docs`: architecture notes, diagrams, and competition submission assets.
- `deploy`: Docker and Hugging Face deployment collateral.
- `archive`: preserved legacy workspaces, prototypes, caches, and safety snapshots.

## Why this is more professional

- It separates runtime concerns from experiments and historical material.
- It gives the application one obvious entrypoint and one obvious import root.
- It keeps every old lineage available without leaving the repo root cluttered.
- It creates a stable foundation for future refactors of the Streamlit monolith into package-owned modules.
