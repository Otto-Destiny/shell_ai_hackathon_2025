from __future__ import annotations

from pathlib import Path

from eagle_blend.config.paths import (
    APP_DB_PATH,
    BATCH_TEMPLATE_PATH,
    BLENDS_TEMPLATE_PATH,
    COMPONENTS_TEMPLATE_PATH,
)

TEMPLATE_PATHS = {
    "batch": BATCH_TEMPLATE_PATH,
    "blends": BLENDS_TEMPLATE_PATH,
    "components": COMPONENTS_TEMPLATE_PATH,
}


def get_registry_db_path() -> Path:
    return APP_DB_PATH


def get_template_path(name: str) -> Path:
    key = name.strip().lower()
    if key not in TEMPLATE_PATHS:
        raise KeyError(f"Unknown template '{name}'. Expected one of: {', '.join(TEMPLATE_PATHS)}")
    return TEMPLATE_PATHS[key]


def read_template_bytes(name: str) -> bytes:
    return get_template_path(name).read_bytes()
