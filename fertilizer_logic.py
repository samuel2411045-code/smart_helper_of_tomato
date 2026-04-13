from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoilCard:
    n: float  # Nitrogen (kg/ha) or ppm (user-entered; rules use relative thresholds)
    p: float  # Phosphorus
    k: float  # Potassium
    ph: float
    organic_carbon: float  # %


@dataclass(frozen=True)
class FertilizerRecommendation:
    summary: str
    chemical: list[str]
    bio: list[str]
    organic: list[str]
    notes: list[str]


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "medium"


def recommend_fertilizer(soil: SoilCard) -> FertilizerRecommendation:
    """
    Rule-based baseline mapping Soil Health Card values → recommendations.
    This is intentionally simple and explainable; you can replace with ML later.
    """

    # NOTE: These thresholds are *baseline* and should be calibrated to your dataset/region units.
    n_band = _band(soil.n, low=120.0, high=280.0)
    p_band = _band(soil.p, low=10.0, high=25.0)
    k_band = _band(soil.k, low=110.0, high=280.0)
    oc_band = _band(soil.organic_carbon, low=0.5, high=0.9)

    chemical: list[str] = []
    bio: list[str] = []
    organic: list[str] = []
    notes: list[str] = []

    # Nitrogen
    if n_band == "low":
        chemical.append("Nitrogen source: Urea (apply in split doses)")
        bio.append("Azotobacter / Azospirillum (for N support)")
    elif n_band == "high":
        notes.append("Nitrogen seems high — avoid over-urea; focus on balanced nutrition.")

    # Phosphorus
    if p_band == "low":
        chemical.append("Phosphorus source: DAP / SSP (as per local guidance)")
        bio.append("PSB (Phosphate Solubilizing Bacteria)")

    # Potassium
    if k_band == "low":
        chemical.append("Potassium source: MOP/SOP (as per local guidance)")

    # pH guidance
    if soil.ph < 6.0:
        notes.append("Soil is acidic — consider liming (as per local agri office).")
        bio.append("PSB (often helpful in acidic soils)")
    elif soil.ph > 8.0:
        notes.append("Soil is alkaline — prefer organic matter + gypsum guidance if needed.")

    # Organic carbon
    if oc_band == "low":
        organic.append("Add FYM/compost/vermicompost to improve organic carbon.")
        notes.append("Low organic carbon — prioritize organic matter + mulching.")
    elif oc_band == "high":
        notes.append("Organic carbon looks good — maintain with compost/mulch.")

    # Always-helpful baseline
    bio.append("Trichoderma (soil/seed treatment; disease suppression support)")

    summary = (
        f"N={n_band}, P={p_band}, K={k_band}, pH={'acidic' if soil.ph < 6 else 'alkaline' if soil.ph > 8 else 'near-neutral'}, "
        f"OC={oc_band}"
    )

    # De-duplicate while preserving order
    def uniq(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for it in items:
            if it not in seen:
                out.append(it)
                seen.add(it)
        return out

    return FertilizerRecommendation(
        summary=summary,
        chemical=uniq(chemical),
        bio=uniq(bio),
        organic=uniq(organic),
        notes=uniq(notes),
    )

