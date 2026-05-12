# src/pipeline/fastf1_fetcher.py
"""
Fetches weather, tyre, and safety car data from FastF1.
Supplements Jolpica data with session-level features.
"""
import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

fastf1.Cache.enable_cache("data/fastf1_cache")


def timeout_handler(signum, frame):
    raise TimeoutError("FastF1 session load timed out")


def fetch_session_features(season: int, round_num: int, timeout: int = 90) -> dict:
    """
    Fetch weather, tyre, and SC data for one race session.
    Returns a dict of race-level and driver-level features.
    Times out after `timeout` seconds to avoid hanging.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        session = fastf1.get_session(season, round_num, "R")
        session.load(
            laps=True,
            telemetry=False,
            weather=True,
            messages=False,
        )
        signal.alarm(0)
    except TimeoutError:
        logger.warning(f"  TIMEOUT {season} R{round_num} — skipping")
        signal.alarm(0)
        return {}
    except Exception as e:
        logger.warning(f"  Failed {season} R{round_num}: {e}")
        signal.alarm(0)
        return {}

    features = {}

    # ── Weather features ───────────────────────────────────────────────────
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            features["rainfall"]          = bool(weather["Rainfall"].any())
            features["avg_air_temp"]      = float(weather["AirTemp"].mean())
            features["avg_track_temp"]    = float(weather["TrackTemp"].mean())
            features["avg_humidity"]      = float(weather["Humidity"].mean())
            features["avg_wind_speed"]    = float(weather["WindSpeed"].mean())
            features["max_track_temp"]    = float(weather["TrackTemp"].max())
    except Exception as e:
        logger.warning(f"  Weather error {season} R{round_num}: {e}")

    # ── Safety car / VSC / Red flag features ──────────────────────────────
    try:
        laps = session.laps
        if laps is not None and not laps.empty:
            total_laps = laps["LapNumber"].max()

            sc_laps  = laps[laps["TrackStatus"].astype(str).str.contains("4")]["LapNumber"].nunique()
            vsc_laps = laps[laps["TrackStatus"].astype(str).str.contains("6")]["LapNumber"].nunique()
            red_laps = laps[laps["TrackStatus"].astype(str).str.contains("5")]["LapNumber"].nunique()

            features["sc_laps"]       = int(sc_laps)
            features["vsc_laps"]      = int(vsc_laps)
            features["red_flag_laps"] = int(red_laps)
            features["sc_pct"]        = float(sc_laps  / max(total_laps, 1))
            features["vsc_pct"]       = float(vsc_laps / max(total_laps, 1))
            features["had_sc"]        = int(sc_laps > 0)
            features["had_vsc"]       = int(vsc_laps > 0)
            features["had_red_flag"]  = int(red_laps > 0)

            # ── Per-driver tyre features ───────────────────────────────────
            driver_features = []
            for driver in laps["Driver"].unique():
                drv_laps = laps[laps["Driver"] == driver]

                start_compound = drv_laps.iloc[0]["Compound"] if not drv_laps.empty else "UNKNOWN"
                stints         = drv_laps["Compound"].ne(drv_laps["Compound"].shift()).cumsum().max()

                soft_laps   = int((drv_laps["Compound"] == "SOFT").sum())
                medium_laps = int((drv_laps["Compound"] == "MEDIUM").sum())
                hard_laps   = int((drv_laps["Compound"] == "HARD").sum())
                total_drv   = len(drv_laps)

                driver_features.append({
                    "driver":         driver,
                    "start_compound": start_compound,
                    "num_stints":     int(stints) if not pd.isna(stints) else 1,
                    "soft_laps":      soft_laps,
                    "medium_laps":    medium_laps,
                    "hard_laps":      hard_laps,
                    "soft_pct":       soft_laps   / max(total_drv, 1),
                    "medium_pct":     medium_laps / max(total_drv, 1),
                    "hard_pct":       hard_laps   / max(total_drv, 1),
                })

            features["driver_tyres"] = driver_features

    except Exception as e:
        logger.warning(f"  Laps error {season} R{round_num}: {e}")

    return features


def build_fastf1_dataset(
    seasons: list,
    save_dir: str = "data/fastf1",
) -> tuple:
    """
    Build race-level and driver-level FastF1 feature datasets.
    Resumes from existing saved data if interrupted.
    Returns (race_df, driver_df).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load existing data
    race_path   = Path(f"{save_dir}/race_conditions.csv")
    driver_path = Path(f"{save_dir}/driver_tyres.csv")

    existing_race   = pd.read_csv(race_path)   if race_path.exists()   else pd.DataFrame()
    existing_driver = pd.read_csv(driver_path) if driver_path.exists() else pd.DataFrame()

    already_done = set()
    if not existing_race.empty:
        for _, row in existing_race.iterrows():
            already_done.add((int(row["season"]), int(row["round"])))

    logger.info(f"Already fetched: {len(already_done)} rounds")

    race_rows   = existing_race.to_dict("records")   if not existing_race.empty   else []
    driver_rows = existing_driver.to_dict("records") if not existing_driver.empty else []
    errors      = 0
    total_done  = len(already_done)

    # Get schedules
    schedules    = {}
    total_rounds = 0
    for season in seasons:
        try:
            sched  = fastf1.get_event_schedule(season, include_testing=False)
            rounds = [r for r in sched["RoundNumber"].tolist() if r > 0]
            schedules[season] = rounds
            total_rounds += len(rounds)
        except Exception:
            schedules[season] = list(range(1, 23))
            total_rounds += 22

    remaining = total_rounds - len(already_done)
    logger.info(f"Remaining: {remaining} rounds to fetch")

    for season in seasons:
        rounds = schedules[season]
        for round_num in rounds:
            if (season, round_num) in already_done:
                continue

            total_done += 1
            pct = total_done / total_rounds * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(
                f"\r[{bar}] {pct:.1f}% | {season} R{round_num} | "
                f"Done: {total_done}/{total_rounds} | Errors: {errors}",
                end="", flush=True,
            )

            feats = fetch_session_features(season, round_num)

            if not feats:
                errors += 1
                pd.DataFrame(race_rows).to_csv(race_path,   index=False)
                pd.DataFrame(driver_rows).to_csv(driver_path, index=False)
                continue

            race_rows.append({
                "season":         season,
                "round":          round_num,
                "rainfall":       feats.get("rainfall",       False),
                "avg_air_temp":   feats.get("avg_air_temp",   np.nan),
                "avg_track_temp": feats.get("avg_track_temp", np.nan),
                "avg_humidity":   feats.get("avg_humidity",   np.nan),
                "avg_wind_speed": feats.get("avg_wind_speed", np.nan),
                "max_track_temp": feats.get("max_track_temp", np.nan),
                "sc_laps":        feats.get("sc_laps",        0),
                "vsc_laps":       feats.get("vsc_laps",       0),
                "red_flag_laps":  feats.get("red_flag_laps",  0),
                "sc_pct":         feats.get("sc_pct",         0.0),
                "vsc_pct":        feats.get("vsc_pct",        0.0),
                "had_sc":         feats.get("had_sc",         0),
                "had_vsc":        feats.get("had_vsc",        0),
                "had_red_flag":   feats.get("had_red_flag",   0),
            })

            for drv in feats.get("driver_tyres", []):
                driver_rows.append({
                    "season":         season,
                    "round":          round_num,
                    "driver":         drv["driver"],
                    "start_compound": drv["start_compound"],
                    "num_stints":     drv["num_stints"],
                    "soft_laps":      drv["soft_laps"],
                    "medium_laps":    drv["medium_laps"],
                    "hard_laps":      drv["hard_laps"],
                    "soft_pct":       drv["soft_pct"],
                    "medium_pct":     drv["medium_pct"],
                    "hard_pct":       drv["hard_pct"],
                })

            # Save every 5 rounds
            if total_done % 5 == 0:
                pd.DataFrame(race_rows).to_csv(race_path,   index=False)
                pd.DataFrame(driver_rows).to_csv(driver_path, index=False)

    print()

    race_df   = pd.DataFrame(race_rows)
    driver_df = pd.DataFrame(driver_rows)

    race_df.to_csv(race_path,   index=False)
    driver_df.to_csv(driver_path, index=False)

    logger.info(f"✓ Done. Race conditions: {len(race_df)}, Driver tyres: {len(driver_df)}, Errors: {errors}")

    return race_df, driver_df


if __name__ == "__main__":
    seasons = list(range(2018, 2027))
    logger.info(f"Fetching FastF1 data for {seasons}...")
    race_df, driver_df = build_fastf1_dataset(seasons)
    print("Done.")