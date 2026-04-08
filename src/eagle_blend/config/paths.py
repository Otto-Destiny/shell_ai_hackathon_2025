from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
APPS_DIR = REPO_ROOT / "apps"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"
RESEARCH_DIR = REPO_ROOT / "research"
ARCHIVE_DIR = REPO_ROOT / "archive"

STREAMLIT_APP_PATH = APPS_DIR / "streamlit" / "streamlit_app.py"
STREAMLIT_CONFIG_PATH = APPS_DIR / "streamlit" / ".streamlit" / "config.toml"

APP_DB_PATH = DATA_DIR / "seed" / "eagleblend.db"
REFERENCE_FUEL_PROPERTIES_PATH = DATA_DIR / "reference" / "fuel_properties.csv"
TEMPLATES_DIR = DATA_DIR / "templates"
BATCH_TEMPLATE_PATH = TEMPLATES_DIR / "batch_template.csv"
BLENDS_TEMPLATE_PATH = TEMPLATES_DIR / "blends_template.csv"
COMPONENTS_TEMPLATE_PATH = TEMPLATES_DIR / "components_template.csv"
BATCH_SAMPLE_PATH = DATA_DIR / "samples" / "batchblend.csv"

PRODUCTION_MODEL_DIR = MODELS_DIR / "production" / "xgboost"
EXPERIMENTAL_MODEL_DIR = MODELS_DIR / "experimental" / "xgboost"

LEGACY_LINEAGES_DIR = ARCHIVE_DIR / "legacy_lineages"
LEGACY_APP_WORKSPACE = LEGACY_LINEAGES_DIR / "App_workspace"
LEGACY_DEV_WORKSPACE = LEGACY_LINEAGES_DIR / "dev_workspace"
LEGACY_ROOT_SNAPSHOT_DIR = ARCHIVE_DIR / "legacy_snapshots" / "root_restore_safeguard_2026_03_31"
LEGACY_ROOT_FILES = (
    LEGACY_ROOT_SNAPSHOT_DIR / "app.py",
    LEGACY_ROOT_SNAPSHOT_DIR / "dash.html",
    LEGACY_ROOT_SNAPSHOT_DIR / "dummy.html",
    LEGACY_ROOT_SNAPSHOT_DIR / "Dockerfile",
    LEGACY_ROOT_SNAPSHOT_DIR / "Eagle_Team_.pdf",
    LEGACY_ROOT_SNAPSHOT_DIR / "Level_2_Prototype.png",
    LEGACY_ROOT_SNAPSHOT_DIR / "Optimization_Engine (1).ipynb",
    LEGACY_ROOT_SNAPSHOT_DIR / "Optimization_Engine_2 (1).ipynb",
    LEGACY_ROOT_SNAPSHOT_DIR / "tabpfn-linR.ipynb",
)


def ensure_src_on_path() -> None:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
