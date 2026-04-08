from __future__ import annotations

from pathlib import Path

from eagle_blend.ml.predictor_xgb import EagleBlendPredictor


def load_predictor(model_dir: str | Path | None = None) -> EagleBlendPredictor:
    return EagleBlendPredictor(model_dir=model_dir)
