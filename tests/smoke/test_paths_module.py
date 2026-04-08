import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eagle_blend.config import paths


def test_paths_module_points_to_canonical_locations() -> None:
    assert paths.REPO_ROOT == REPO_ROOT
    assert paths.STREAMLIT_APP_PATH == REPO_ROOT / "apps/streamlit/streamlit_app.py"
    assert paths.APP_DB_PATH == REPO_ROOT / "data/seed/eagleblend.db"
    assert paths.PRODUCTION_MODEL_DIR == REPO_ROOT / "models/production/xgboost"
    assert paths.EXPERIMENTAL_MODEL_DIR == REPO_ROOT / "models/experimental/xgboost"


def test_paths_module_points_to_preserved_lineages() -> None:
    assert paths.LEGACY_APP_WORKSPACE == REPO_ROOT / "archive/legacy_lineages/App_workspace"
    assert paths.LEGACY_DEV_WORKSPACE == REPO_ROOT / "archive/legacy_lineages/dev_workspace"
    assert paths.LEGACY_ROOT_SNAPSHOT_DIR == REPO_ROOT / "archive/legacy_snapshots/root_restore_safeguard_2026_03_31"


def test_paths_are_resolved_inside_repo() -> None:
    for candidate in [
        paths.STREAMLIT_APP_PATH,
        paths.APP_DB_PATH,
        paths.PRODUCTION_MODEL_DIR,
        paths.LEGACY_APP_WORKSPACE,
        paths.LEGACY_DEV_WORKSPACE,
    ]:
        candidate.resolve().relative_to(REPO_ROOT.resolve())
