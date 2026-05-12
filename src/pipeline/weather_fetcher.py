# src/pipeline/weather_fetcher.py
"""
Fetches historical weather data from Open-Meteo API for all F1 circuits.
Also builds circuit-level uncertainty features (SC rate, DNF rate, etc.)
from historical race data.
"""
import urllib.request
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Circuit coordinates (lat, lon)
CIRCUIT_COORDS = {
    "albert_park":    (-37.8497, 144.9680),
    "americas":       (30.1328, -97.6411),
    "bahrain":        (26.0325, 50.5106),
    "baku":           (40.3725, 49.8533),
    "catalunya":      (41.5700, 2.2611),
    "hockenheimring": (49.3278, 8.5656),
    "hungaroring":    (47.5789, 19.2486),
    "imola":          (44.3439, 11.7167),
    "interlagos":     (-23.7036, -46.6997),
    "istanbul":       (40.9517, 29.4050),
    "jeddah":         (21.6319, 39.1044),
    "losail":         (25.4900, 51.4542),
    "marina_bay":     (1.2914, 103.8639),
    "miami":          (25.9581, -80.2389),
    "monaco":         (43.7347, 7.4206),
    "monza":          (45.6156, 9.2811),
    "mugello":        (43.9975, 11.3719),
    "nurburgring":    (50.3356, 6.9475),
    "portimao":       (37.2272, -8.6267),
    "red_bull_ring":  (47.2197, 14.7647),
    "ricard":         (43.2506, 5.7917),
    "rodriguez":      (19.4042, -99.0907),
    "shanghai":       (31.3389, 121.2197),
    "silverstone":    (52.0786, -1.0169),
    "sochi":          (43.4057, 39.9578),
    "spa":            (50.4372, 5.9714),
    "suzuka":         (34.8431, 136.5408),
    "vegas":          (36.1147, -115.1728),
    "villeneuve":     (45.5000, -73.5228),
    "yas_marina":     (24.4672, 54.6031),
    "zandvoort":      (52.3888, 4.5409),
}


def fetch_race_weather(circuit_id: str, date: str) -> dict:
    """Fetch weather for a specific circuit and race date."""
    if circuit_id not in CIRCUIT_COORDS:
        return {}

    lat, lon = CIRCUIT_COORDS[circuit_id]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,"
        f"windspeed_10m_max,precipitation_hours"
        f"&timezone=auto"
    )

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        daily = data.get("daily", {})
        precip     = daily.get("precipitation_sum",  [None])[0]
        temp_max   = daily.get("temperature_2m_max", [None])[0]
        temp_min   = daily.get("temperature_2m_min", [None])[0]
        wind_max   = daily.get("windspeed_10m_max",  [None])[0]
        precip_hrs = daily.get("precipitation_hours",[None])[0]

        return {
            "precipitation_mm":    float(precip)     if precip     is not None else 0.0,
            "temp_max_c":          float(temp_max)   if temp_max   is not None else np.nan,
            "temp_min_c":          float(temp_min)   if temp_min   is not None else np.nan,
            "wind_max_kmh":        float(wind_max)   if wind_max   is not None else np.nan,
            "precipitation_hours": float(precip_hrs) if precip_hrs is not None else 0.0,
            "rainfall":            (float(precip) > 0.5) if precip is not None else False,
        }
    except Exception as e:
        logger.warning(f"  Weather fetch failed for {circuit_id} {date}: {e}")
        return {}


def build_weather_dataset(
    results_df: pd.DataFrame,
    save_path: str = "data/weather/race_weather.csv",
) -> pd.DataFrame:
    """
    Fetch weather for all races in the results DataFrame.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing if any
    save_file = Path(save_path)
    if save_file.exists():
        existing = pd.read_csv(save_path)
        done = set(zip(existing["season"], existing["round"]))
        rows = existing.to_dict("records")
        logger.info(f"Already fetched: {len(done)} races")
    else:
        done = set()
        rows = []

    # Get unique race-circuit-date combinations
    races = (
        results_df[["season", "round", "circuit_id", "date"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )

    total   = len(races)
    fetched = 0
    errors  = 0

    for _, race in races.iterrows():
        season    = int(race["season"])
        round_num = int(race["round"])

        if (season, round_num) in done:
            continue

        fetched += 1
        pct = (len(done) + fetched) / total * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(
            f"\r[{bar}] {pct:.1f}% | {season} R{round_num} | "
            f"Fetched: {fetched} | Errors: {errors}",
            end="", flush=True,
        )

        weather = fetch_race_weather(race["circuit_id"], race["date"])
        if not weather:
            errors += 1
            rows.append({
                "season": season, "round": round_num,
                "circuit_id": race["circuit_id"],
                "date": race["date"],
                "precipitation_mm": 0.0,
                "temp_max_c": np.nan,
                "temp_min_c": np.nan,
                "wind_max_kmh": np.nan,
                "precipitation_hours": 0.0,
                "rainfall": False,
            })
        else:
            rows.append({
                "season":    season,
                "round":     round_num,
                "circuit_id": race["circuit_id"],
                "date":      race["date"],
                **weather,
            })

        # Save every 10 races
        if fetched % 10 == 0:
            pd.DataFrame(rows).to_csv(save_path, index=False)

        time.sleep(0.2)  # be polite to the API

    print()
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info(f"✓ Weather data: {len(df)} races | Errors: {errors}")
    return df


def build_circuit_uncertainty_features(
    results_df: pd.DataFrame,
    save_path: str = "data/weather/circuit_uncertainty.csv",
) -> pd.DataFrame:
    """
    Build circuit-level uncertainty features from historical patterns:
    - SC rate (estimated from DNF/incident patterns)
    - DNF rate per constructor
    - Overtake difficulty
    - Historical wet race rate
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for circuit_id, grp in results_df.groupby("circuit_id"):
        total_races  = grp["round"].nunique() if "round" in grp else len(grp) // 20

        # DNF rate
        dnf_rate = grp["dnf"].mean() if "dnf" in grp.columns else 0.1

        # Positions gained (overtake index)
        if "grid_position" in grp.columns and "finish_position" in grp.columns:
            valid = grp[(grp["grid_position"] > 0) & (grp["finish_position"] > 0)]
            avg_positions_gained = (
                valid["grid_position"] - valid["finish_position"]
            ).mean() if not valid.empty else 0.0
        else:
            avg_positions_gained = 0.0

        rows.append({
            "circuit_id":           circuit_id,
            "historical_dnf_rate":  round(float(dnf_rate), 4),
            "avg_positions_gained": round(float(avg_positions_gained), 4),
            "total_races":          int(total_races),
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info(f"✓ Circuit uncertainty: {len(df)} circuits")
    return df


if __name__ == "__main__":
    results = pd.read_csv("data/jolpica/results.csv")
    logger.info("Building weather dataset...")
    weather = build_weather_dataset(results)
    logger.info("Building circuit uncertainty features...")
    uncertainty = build_circuit_uncertainty_features(results)
    print("Done!")