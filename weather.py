from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import requests
import geocoder


@dataclass(frozen=True)
class WeatherSummary:
    tmean_c: float
    tmin_c: float
    tmax_c: float
    rainfall_mm: float
    humidity_mean: float | None


def fetch_open_meteo_daily(
    *,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> WeatherSummary:
    """
    Fetch daily weather from Open-Meteo (no key).
    We compute simple summary stats over the date range.
    """

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
            ]
        ),
        "timezone": "auto",
    }
    # Relative humidity daily mean is not always available on forecast endpoint;
    # keep it optional.

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily") or {}
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_sum") or []

    if not tmax or not tmin:
        raise ValueError("Open-Meteo returned empty temperature series for the requested dates.")

    tmax_mean = sum(tmax) / len(tmax)
    tmin_mean = sum(tmin) / len(tmin)
    tmean = (tmax_mean + tmin_mean) / 2.0
    rainfall = float(sum(rain)) if rain else 0.0

    return WeatherSummary(
        tmean_c=float(tmean),
        tmin_c=float(min(tmin)),
        tmax_c=float(max(tmax)),
        rainfall_mm=float(rainfall),
        humidity_mean=None,
    )


def get_location_from_ip() -> tuple[float, float] | None:
    """
    Get approximate latitude and longitude based on IP address.
    """
    try:
        g = geocoder.ip("me")
        if g.latlng:
            return float(g.latlng[0]), float(g.latlng[1])
    except Exception:
        pass
    return None
