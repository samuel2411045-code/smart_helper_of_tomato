from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass(frozen=True)
class YieldPrediction:
    label: str
    value: float | None
    units: str | None
    note: str | None


def _bucketize(yield_value: float) -> str:
    # Simple default buckets; tune once you have real data.
    if yield_value < 8:
        return "Low"
    if yield_value < 15:
        return "Medium"
    return "High"


def predict_yield(
    *,
    features: dict[str, float],
    model_path: str | Path = "models/yield_model.joblib",
    output: str = "bucket",  # "bucket" or "numeric"
) -> YieldPrediction:
    """
    Predict tomato yield using a scikit-learn model if present.
    If missing, returns an explainable heuristic baseline.
    """

    model_path = Path(model_path)
    x = np.array(
        [
            [
                features.get("temp_mean_c", 0.0),
                features.get("rainfall_mm", 0.0),
                features.get("soil_n", 0.0),
                features.get("soil_p", 0.0),
                features.get("soil_k", 0.0),
                features.get("soil_ph", 0.0),
                features.get("organic_carbon", 0.0),
                features.get("season_kharif", 0.0),
                features.get("season_rabi", 0.0),
                features.get("season_summer", 0.0),
            ]
        ],
        dtype="float32",
    )

    if model_path.exists():
        model = joblib.load(model_path)
        y = model.predict(x)[0]
        y = float(y)
        if output == "numeric":
            return YieldPrediction(label="Predicted yield", value=y, units="(model units)", note=None)
        return YieldPrediction(label=_bucketize(y), value=y, units="(model units)", note=None)

    # Heuristic baseline: penalize extreme rainfall, reward balanced NPK + near-neutral pH
    temp = features.get("temp_mean_c", 28.0)
    rain = features.get("rainfall_mm", 100.0)
    ph = features.get("soil_ph", 6.8)
    n = features.get("soil_n", 200.0)
    p = features.get("soil_p", 15.0)
    k = features.get("soil_k", 200.0)
    oc = features.get("organic_carbon", 0.7)

    score = 0.0
    score += max(0.0, 1.0 - abs(temp - 26.0) / 12.0) * 0.25
    score += max(0.0, 1.0 - abs(rain - 120.0) / 200.0) * 0.25
    score += max(0.0, 1.0 - abs(ph - 6.8) / 2.0) * 0.2
    score += max(0.0, 1.0 - abs(n - 220.0) / 220.0) * 0.1
    score += max(0.0, 1.0 - abs(p - 18.0) / 18.0) * 0.1
    score += max(0.0, 1.0 - abs(k - 220.0) / 220.0) * 0.05
    score += max(0.0, min(1.0, oc / 1.2)) * 0.05

    # Map [0,1] score to a pseudo numeric yield (e.g., tons/acre placeholder)
    pseudo_yield = 5.0 + score * 15.0
    label = _bucketize(pseudo_yield) if output != "numeric" else "Heuristic yield"

    return YieldPrediction(
        label=label,
        value=float(pseudo_yield),
        units="(heuristic, not calibrated)",
        note=f"Model missing at `{model_path.as_posix()}` — using heuristic baseline.",
    )

