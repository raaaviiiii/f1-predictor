# src/pipeline/jolpica_fetcher.py
"""
Fetches rich F1 data from the Jolpica API (Ergast replacement).
Covers race results, qualifying times, pit stops, driver standings.
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

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SLEEP    = 0.35  # be polite to the API


def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logger.warning(f"Failed to fetch {url}: {e}")
                return {}


def lap_to_ms(lap_str: str) -> float:
    """Convert '1:29.179' or '29.179' to milliseconds."""
    if not lap_str or lap_str == "":
        return np.nan
    try:
        parts = lap_str.strip().split(":")
        if len(parts) == 2:
            return (int(parts[0]) * 60 + float(parts[1])) * 1000
        return float(parts[0]) * 1000
    except:
        return np.nan


def fetch_season_results(season: int) -> pd.DataFrame:
    """Fetch all race results for a season."""
    logger.info(f"  Fetching {season} race results...")
    rows = []

    # Get schedule first
    sched = fetch_json(f"{BASE_URL}/{season}/?format=json")
    if not sched:
        return pd.DataFrame()

    races = sched["MRData"]["RaceTable"]["Races"]

    for race in races:
        round_num   = int(race["round"])
        race_name   = race["raceName"]
        circuit_id  = race["Circuit"]["circuitId"]
        circuit_name = race["Circuit"]["circuitName"]
        country     = race["Circuit"]["Location"]["country"]
        date        = race["date"]

        # Race results
        res = fetch_json(f"{BASE_URL}/{season}/{round_num}/results/?format=json")
        time.sleep(SLEEP)

        if not res:
            continue

        race_results = res["MRData"]["RaceTable"]["Races"]
        if not race_results:
            continue

        for r in race_results[0]["Results"]:
            driver_id   = r["Driver"]["driverId"]
            driver_code = r["Driver"].get("code", driver_id[:3].upper())
            constructor = r["Constructor"]["name"]
            grid        = int(r.get("grid", 0))
            position    = r.get("positionText", "")
            laps        = int(r.get("laps", 0))
            status      = r.get("status", "")
            points      = float(r.get("points", 0))

            # Finish position
            try:
                finish_pos = int(r["position"])
            except:
                finish_pos = 20  # DNF

            dnf = 0 if status == "Finished" or "Lap" in status else 1

            # Fastest lap
            fl = r.get("FastestLap", {})
            fl_time_ms = lap_to_ms(fl.get("Time", {}).get("time", ""))
            fl_rank    = int(fl.get("rank", 99))

            # Race time gap to winner
            race_time = r.get("Time", {})
            time_millis = np.nan
            if "millis" in race_time:
                time_millis = float(race_time["millis"])

            rows.append({
                "season":        season,
                "round":         round_num,
                "race_name":     race_name,
                "circuit_id":    circuit_id,
                "circuit_name":  circuit_name,
                "country":       country,
                "date":          date,
                "driver":        driver_code,
                "driver_id":     driver_id,
                "team":          constructor,
                "grid_position": grid,
                "finish_position": finish_pos,
                "points_scored": points,
                "laps_completed": laps,
                "status":        status,
                "dnf":           dnf,
                "fastest_lap_ms": fl_time_ms,
                "fastest_lap_rank": fl_rank,
                "race_time_ms":  time_millis,
            })

    return pd.DataFrame(rows)


def fetch_season_qualifying(season: int) -> pd.DataFrame:
    """Fetch qualifying results for all rounds in a season."""
    logger.info(f"  Fetching {season} qualifying...")
    rows = []

    sched = fetch_json(f"{BASE_URL}/{season}/?format=json")
    if not sched:
        return pd.DataFrame()

    races = sched["MRData"]["RaceTable"]["Races"]

    for race in races:
        round_num = int(race["round"])

        res = fetch_json(f"{BASE_URL}/{season}/{round_num}/qualifying/?format=json")
        time.sleep(SLEEP)

        if not res:
            continue

        race_results = res["MRData"]["RaceTable"]["Races"]
        if not race_results:
            continue

        qual_results = race_results[0].get("QualifyingResults", [])
        if not qual_results:
            continue

        # Find pole time (Q3 or Q2 or Q1)
        pole_time_ms = np.nan
        for r in qual_results:
            if int(r["position"]) == 1:
                for q in ["Q3", "Q2", "Q1"]:
                    t = lap_to_ms(r.get(q, ""))
                    if not np.isnan(t):
                        pole_time_ms = t
                        break

        for r in qual_results:
            driver_code = r["Driver"].get("code", r["Driver"]["driverId"][:3].upper())
            quali_pos   = int(r["position"])

            # Best qualifying time
            best_ms = np.nan
            for q in ["Q3", "Q2", "Q1"]:
                t = lap_to_ms(r.get(q, ""))
                if not np.isnan(t):
                    best_ms = t
                    break

            gap_to_pole_ms  = best_ms - pole_time_ms if not np.isnan(best_ms) and not np.isnan(pole_time_ms) else np.nan
            gap_to_pole_pct = (gap_to_pole_ms / pole_time_ms * 100) if not np.isnan(gap_to_pole_ms) else np.nan

            q1_ms = lap_to_ms(r.get("Q1", ""))
            q2_ms = lap_to_ms(r.get("Q2", ""))
            q3_ms = lap_to_ms(r.get("Q3", ""))

            rows.append({
                "season":           season,
                "round":            round_num,
                "driver":           driver_code,
                "quali_position":   quali_pos,
                "q1_ms":            q1_ms,
                "q2_ms":            q2_ms,
                "q3_ms":            q3_ms,
                "best_quali_ms":    best_ms,
                "pole_time_ms":     pole_time_ms,
                "gap_to_pole_ms":   gap_to_pole_ms,
                "gap_to_pole_pct":  gap_to_pole_pct,
            })

    return pd.DataFrame(rows)


def fetch_season_pitstops(season: int) -> pd.DataFrame:
    """Fetch pit stop data for all rounds."""
    logger.info(f"  Fetching {season} pit stops...")
    rows = []

    sched = fetch_json(f"{BASE_URL}/{season}/?format=json")
    if not sched:
        return pd.DataFrame()

    races = sched["MRData"]["RaceTable"]["Races"]

    for race in races:
        round_num = int(race["round"])

        res = fetch_json(f"{BASE_URL}/{season}/{round_num}/pitstops/?format=json&limit=100")
        time.sleep(SLEEP)

        if not res:
            continue

        race_results = res["MRData"]["RaceTable"]["Races"]
        if not race_results:
            continue

        pit_data = race_results[0].get("PitStops", [])
        driver_stops = {}
        for p in pit_data:
            drv = p.get("driverId", "")
            driver_stops[drv] = driver_stops.get(drv, 0) + 1

        for driver_id, stop_count in driver_stops.items():
            rows.append({
                "season":    season,
                "round":     round_num,
                "driver_id": driver_id,
                "pit_stops": stop_count,
            })

    return pd.DataFrame(rows)


def fetch_all_seasons(
    seasons: list,
    save_dir: str = "data/jolpica",
) -> dict:
    """
    Fetch race results + qualifying + pit stops for all seasons.
    Saves CSVs to save_dir and returns DataFrames.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_results   = []
    all_quali     = []
    all_pitstops  = []

    for season in seasons:
        logger.info(f"Fetching season {season}...")

        results = fetch_season_results(season)
        if not results.empty:
            all_results.append(results)

        quali = fetch_season_qualifying(season)
        if not quali.empty:
            all_quali.append(quali)

        pitstops = fetch_season_pitstops(season)
        if not pitstops.empty:
            all_pitstops.append(pitstops)

        logger.info(f"  ✓ {season}: {len(results)} rows")

    results_df  = pd.concat(all_results,  ignore_index=True) if all_results  else pd.DataFrame()
    quali_df    = pd.concat(all_quali,    ignore_index=True) if all_quali    else pd.DataFrame()
    pitstops_df = pd.concat(all_pitstops, ignore_index=True) if all_pitstops else pd.DataFrame()

    results_df.to_csv(f"{save_dir}/results.csv",   index=False)
    quali_df.to_csv(f"{save_dir}/qualifying.csv",  index=False)
    pitstops_df.to_csv(f"{save_dir}/pitstops.csv", index=False)

    logger.info(f"✓ Saved to {save_dir}")
    logger.info(f"  Results:   {len(results_df)} rows")
    logger.info(f"  Quali:     {len(quali_df)} rows")
    logger.info(f"  Pit stops: {len(pitstops_df)} rows")

    return {
        "results":  results_df,
        "quali":    quali_df,
        "pitstops": pitstops_df,
    }


if __name__ == "__main__":
    seasons = list(range(2018, 2027))
    logger.info(f"Fetching seasons {seasons}...")
    data = fetch_all_seasons(seasons)
    print("Done.")