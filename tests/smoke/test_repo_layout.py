from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_repo_zones_exist() -> None:
    expected = [
        REPO_ROOT / "apps",
        REPO_ROOT / "src",
        REPO_ROOT / "data",
        REPO_ROOT / "models",
        REPO_ROOT / "research",
        REPO_ROOT / "docs",
        REPO_ROOT / "deploy",
        REPO_ROOT / "archive",
        REPO_ROOT / "tests",
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / ".gitignore",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected path: {path}"


def test_canonical_runtime_assets_exist() -> None:
    expected = [
        REPO_ROOT / "apps/streamlit/streamlit_app.py",
        REPO_ROOT / "apps/streamlit/.streamlit/config.toml",
        REPO_ROOT / "data/seed/eagleblend.db",
        REPO_ROOT / "data/reference/fuel_properties.csv",
        REPO_ROOT / "data/templates/batch_template.csv",
        REPO_ROOT / "data/templates/blends_template.csv",
        REPO_ROOT / "data/templates/components_template.csv",
        REPO_ROOT / "models/production/xgboost/xmodel.joblib",
        REPO_ROOT / "models/production/xgboost/pca.joblib",
        REPO_ROOT / "models/production/xgboost/scaler.joblib",
        REPO_ROOT / "models/experimental/xgboost/xgb_model.joblib",
    ]
    for path in expected:
        assert path.exists(), f"Missing runtime asset: {path}"


def test_legacy_material_is_preserved_in_archive() -> None:
    expected = [
        REPO_ROOT / "archive/legacy_lineages/App_workspace",
        REPO_ROOT / "archive/legacy_lineages/dev_workspace",
        REPO_ROOT / "archive/legacy_snapshots/root_restore_safeguard_2026_03_31",
        REPO_ROOT / "archive/prototypes/ui/app.py",
        REPO_ROOT / "archive/prototypes/ui/dash.html",
        REPO_ROOT / "archive/prototypes/ui/dummy.html",
    ]
    for path in expected:
        assert path.exists(), f"Missing preserved lineage: {path}"
