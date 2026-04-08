from __future__ import annotations

import pandas as pd


def summarize_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(columns=["metric", "mean", "min", "max"])
    summary = numeric.agg(["mean", "min", "max"]).transpose().reset_index()
    return summary.rename(columns={"index": "metric"})
