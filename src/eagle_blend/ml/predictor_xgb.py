from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eagle_blend.config.paths import PRODUCTION_MODEL_DIR

ALL_BLEND_COLUMNS = [f"BlendProperty{i}" for i in range(1, 11)]
FAST_BLEND_COLUMNS = [
    "BlendProperty1",
    "BlendProperty2",
    "BlendProperty5",
    "BlendProperty6",
    "BlendProperty7",
    "BlendProperty10",
]


class EagleBlendPredictor:
    """Load and serve the production XGBoost blend-property model."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else PRODUCTION_MODEL_DIR
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.pca = joblib.load(self.model_dir / "pca.joblib")
        self.model = joblib.load(self.model_dir / "xmodel.joblib")

    def predict_arr(self, X_new: np.ndarray | list[list[float]]) -> np.ndarray:
        X_array = np.asarray(X_new)
        X_scaled = self.scaler.transform(X_array)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)

    def predict_all(self, X_new: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X_new, pd.DataFrame):
            raise TypeError("Input X_new must be a pandas DataFrame.")
        predictions = self.predict_arr(X_new)
        return pd.DataFrame(predictions, columns=ALL_BLEND_COLUMNS, index=X_new.index)

    def predict_fast(self, X_new: pd.DataFrame) -> pd.DataFrame:
        predictions_df = self.predict_all(X_new)
        return predictions_df[FAST_BLEND_COLUMNS]
