from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from eagle_blend.config.paths import APP_DB_PATH


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path else APP_DB_PATH


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(resolve_db_path(db_path))


def read_sql(query: str, params: Iterable[Any] | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=list(params or []))


def read_table(table_name: str, db_path: str | Path | None = None, order_by: str | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    if order_by:
        query += f" ORDER BY {order_by}"
    return read_sql(query, db_path=db_path)


def append_frame(frame: pd.DataFrame, table_name: str, db_path: str | Path | None = None) -> int:
    with connect(db_path) as conn:
        frame.to_sql(table_name, conn, if_exists="append", index=False)
    return len(frame)


def execute(query: str, params: Iterable[Any] | None = None, db_path: str | Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(query, tuple(params or []))
        conn.commit()
