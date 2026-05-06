# src/pipeline/loader.py
import time
import warnings
from pathlib import Path
from typing import Optional

import fastf1 
import pandas as pd 
import numpy as np

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    CACHE_DIR, RAW_DIR, POINTS_SYSTEM,
    SPRINT_POINTS, FASTEST_LAP_POINT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tell FastF1 where to cache downloaded sessions
fastf1.Cache.enable_cache(str(CACHE_DIR))


class SessionFetcher:
    """
    Fetches a single FastF1 session with automatic retries.
    Returns None if the session doesn't exist or fails after 3 attempts.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds between retries

    def fetch(
        self,
        year: int,
        round_number: int,
        session_type: str,  # 'R' = Race, 'Q' = Qualifying, 'S' = Sprint
    ) -> Optional[object]:

        for attempt in range(self.MAX_RETRIES):
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load(
                    laps=True,
                    telemetry=False,
                    weather=True,
                    messages=False,
                )
                return session

            except Exception as exc:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"  Attempt {attempt + 1} failed for "
                        f"{year} R{round_number} {session_type}: {exc}. Retrying..."
                    )
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.warning(
                        f"  Skipping {year} R{round_number} {session_type} "
                        f"after {self.MAX_RETRIES} attempts: {exc}"
                    )
                    return None


class RaceParser:
    """
    Turns a FastF1 Race session into a flat DataFrame.
    One row per driver with their result, lap stats, pit stops, and weather.
    """

    def parse(self, session) -> pd.DataFrame:
        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        rows = []
        for _, row in results.iterrows():
            driver = row.get("Abbreviation", "")
            if not driver:
                continue

            # Lap time stats for this driver
            driver_laps = session.laps.pick_driver(driver)
            valid_laps  = driver_laps[driver_laps["LapTime"].notna()]

            fastest_lap = valid_laps["LapTime"].min() if not valid_laps.empty else pd.NaT
            median_lap  = valid_laps["LapTime"].median() if not valid_laps.empty else pd.NaT
            lap_count   = len(driver_laps)

            # Pit stop count
            pit_stops = int(driver_laps["PitOutTime"].notna().sum()) if not driver_laps.empty else 0

            # Tyre compounds used
            compounds = (
                driver_laps["Compound"].dropna().unique().tolist()
                if not driver_laps.empty else []
            )

            # Points
            finish_pos = row.get("Position", np.nan)
            try:
                finish_pos = int(finish_pos)
            except (ValueError, TypeError):
                finish_pos = 0

            base_points = POINTS_SYSTEM.get(finish_pos, 0)

            # Fastest lap bonus (post-2019, top-10 only)
            fl_bonus = 0
            if session.event["EventDate"].year >= 2019 and finish_pos <= 10:
                try:
                    overall_fastest = session.laps["LapTime"].min()
                    if fastest_lap == overall_fastest:
                        fl_bonus = FASTEST_LAP_POINT
                except Exception:
                    pass

            # Weather summary
            weather     = session.weather_data
            track_temp  = np.nan
            air_temp    = np.nan
            rainfall    = False

            if weather is not None and not weather.empty:
                track_temp = weather["TrackTemp"].mean()
                air_temp   = weather["AirTemp"].mean()
                rainfall   = bool(weather["Rainfall"].any())

            # DNF check
            status = str(row.get("Status", "")).strip()
            dnf    = status not in ("Finished", "") and not status.startswith("+")

            rows.append({
                "season":           session.event["EventDate"].year,
                "round":            session.event["RoundNumber"],
                "circuit_id":       session.event["Location"].replace(" ", "_").lower(),
                "event_name":       session.event["EventName"],
                "event_date":       session.event["EventDate"],
                "driver":           driver,
                "driver_full":      row.get("FullName", ""),
                "team":             row.get("TeamName", ""),
                "grid_position":    int(row.get("GridPosition", 0) or 0),
                "finish_position":  finish_pos,
                "status":           status,
                "dnf":              dnf,
                "points_scored":    base_points + fl_bonus,
                "fl_bonus":         fl_bonus,
                "lap_count":        lap_count,
                "fastest_lap_ms":   (
                    fastest_lap.total_seconds() * 1000
                    if pd.notna(fastest_lap) else np.nan
                ),
                "median_lap_ms":    (
                    median_lap.total_seconds() * 1000
                    if pd.notna(median_lap) else np.nan
                ),
                "pit_stops":        pit_stops,
                "compounds_used":   ",".join(str(c) for c in compounds),
                "avg_track_temp_c": track_temp,
                "avg_air_temp_c":   air_temp,
                "rainfall":         rainfall,
            })

        return pd.DataFrame(rows)


class QualiParser:
    """
    Turns a FastF1 Qualifying session into a flat DataFrame.
    One row per driver with Q1/Q2/Q3 times and final grid position.
    """

    def parse(self, session) -> pd.DataFrame:
        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        rows = []
        for _, row in results.iterrows():
            driver = row.get("Abbreviation", "")
            if not driver:
                continue

            def lap_ms(col):
                val = row.get(col, pd.NaT)
                if pd.isna(val):
                    return np.nan
                try:
                    return val.total_seconds() * 1000
                except Exception:
                    return np.nan

            q1 = lap_ms("Q1")
            q2 = lap_ms("Q2")
            q3 = lap_ms("Q3")

            times = [t for t in [q1, q2, q3] if not np.isnan(t)]
            best  = min(times) if times else np.nan

            rows.append({
                "season":         session.event["EventDate"].year,
                "round":          session.event["RoundNumber"],
                "circuit_id":     session.event["Location"].replace(" ", "_").lower(),
                "driver":         driver,
                "team":           row.get("TeamName", ""),
                "quali_position": int(row.get("Position", 0) or 0),
                "q1_time_ms":     q1,
                "q2_time_ms":     q2,
                "q3_time_ms":     q3,
                "best_quali_ms":  best,
            })

        return pd.DataFrame(rows)


class SprintParser:
    """
    Turns a FastF1 Sprint session into a flat DataFrame.
    One row per driver with finish position and sprint points.
    """

    def parse(self, session) -> pd.DataFrame:
        results = session.results
        if results is None or results.empty:
            return pd.DataFrame()

        rows = []
        for _, row in results.iterrows():
            driver = row.get("Abbreviation", "")
            if not driver:
                continue

            finish_pos = row.get("Position", np.nan)
            try:
                finish_pos = int(finish_pos)
            except (ValueError, TypeError):
                finish_pos = 0

            rows.append({
                "season":           session.event["EventDate"].year,
                "round":            session.event["RoundNumber"],
                "circuit_id":       session.event["Location"].replace(" ", "_").lower(),
                "driver":           driver,
                "team":             row.get("TeamName", ""),
                "sprint_finish_pos": finish_pos,
                "sprint_points":    SPRINT_POINTS.get(finish_pos, 0),
            })

        return pd.DataFrame(rows)


class DataLoader:
    """
    Main entry point for data collection.
    Loops over every season and round, fetches Race + Qualifying + Sprint,
    and saves per-season parquet files to data/raw/.

    Already downloaded seasons are loaded from disk automatically.

    Usage:
        loader = DataLoader()
        data = loader.run(seasons=[2023, 2024])
        data = loader.run()               # all seasons in config
        data = loader.load_from_disk()    # skip API entirely
    """

    def __init__(self):
        self.fetcher       = SessionFetcher()
        self.race_parser   = RaceParser()
        self.quali_parser  = QualiParser()
        self.sprint_parser = SprintParser()

    def run(
        self,
        seasons: list = None,
        force_reload: bool = False,
    ) -> dict:

        from config.config import SEASONS
        seasons = seasons or SEASONS

        all_races   = []
        all_qualis  = []
        all_sprints = []

        for season in seasons:
            logger.info(f"{'─' * 50}")
            logger.info(f"Season {season}")

            race_path  = RAW_DIR / f"season_{season}.parquet"
            quali_path = RAW_DIR / f"quali_{season}.parquet"
            sprint_path= RAW_DIR / f"sprint_{season}.parquet"

            # Load from disk if already downloaded
            if race_path.exists() and not force_reload:
                logger.info(f"  Already downloaded — loading from disk")
                all_races.append(pd.read_parquet(race_path))
                if quali_path.exists():
                    all_qualis.append(pd.read_parquet(quali_path))
                if sprint_path.exists():
                    all_sprints.append(pd.read_parquet(sprint_path))
                continue

            season_races   = []
            season_qualis  = []
            season_sprints = []

            schedule = fastf1.get_event_schedule(season, include_testing=False)
            rounds   = schedule["RoundNumber"].tolist()

            for rnd in rounds:
                name = schedule.loc[
                    schedule["RoundNumber"] == rnd, "EventName"
                ].values
                name = name[0] if len(name) else f"Round {rnd}"
                logger.info(f"  Round {rnd:02d} — {name}")

                # Race
                race_session = self.fetcher.fetch(season, rnd, "R")
                if race_session:
                    df = self.race_parser.parse(race_session)
                    if not df.empty:
                        season_races.append(df)
                        logger.info(f"    ✓ Race: {len(df)} drivers")

                # Qualifying
                quali_session = self.fetcher.fetch(season, rnd, "Q")
                if quali_session:
                    df = self.quali_parser.parse(quali_session)
                    if not df.empty:
                        season_qualis.append(df)
                        logger.info(f"    ✓ Qualifying: {len(df)} drivers")

                # Sprint (only some rounds, post-2021)
                if season >= 2021:
                    sprint_session = self.fetcher.fetch(season, rnd, "S")
                    if sprint_session:
                        df = self.sprint_parser.parse(sprint_session)
                        if not df.empty:
                            season_sprints.append(df)
                            logger.info(f"    ✓ Sprint: {len(df)} drivers")

                time.sleep(0.3)

            # Save season to disk
            if season_races:
                df = pd.concat(season_races, ignore_index=True)
                df.to_parquet(race_path, index=False)
                all_races.append(df)
                logger.info(f"  Saved {len(df)} race rows for {season}")

            if season_qualis:
                df = pd.concat(season_qualis, ignore_index=True)
                df.to_parquet(quali_path, index=False)
                all_qualis.append(df)

            if season_sprints:
                df = pd.concat(season_sprints, ignore_index=True)
                df.to_parquet(sprint_path, index=False)
                all_sprints.append(df)

        result = {}
        if all_races:
            result["races"] = pd.concat(all_races, ignore_index=True)
        if all_qualis:
            result["qualifying"] = pd.concat(all_qualis, ignore_index=True)
        if all_sprints:
            result["sprints"] = pd.concat(all_sprints, ignore_index=True)

        logger.info(f"\nDone. Races: {len(result.get('races', []))}, "
                    f"Quali: {len(result.get('qualifying', []))}, "
                    f"Sprints: {len(result.get('sprints', []))}")

        return result

    def load_from_disk(self, seasons: list = None) -> dict:
        """Load already-downloaded parquet files without hitting the API."""
        from config.config import SEASONS
        seasons = seasons or SEASONS

        races, qualis, sprints = [], [], []
        for season in seasons:
            rp = RAW_DIR / f"season_{season}.parquet"
            qp = RAW_DIR / f"quali_{season}.parquet"
            sp = RAW_DIR / f"sprint_{season}.parquet"
            if rp.exists():
                races.append(pd.read_parquet(rp))
            if qp.exists():
                qualis.append(pd.read_parquet(qp))
            if sp.exists():
                sprints.append(pd.read_parquet(sp))

        result = {}
        if races:
            result["races"]      = pd.concat(races, ignore_index=True)
        if qualis:
            result["qualifying"] = pd.concat(qualis, ignore_index=True)
        if sprints:
            result["sprints"]    = pd.concat(sprints, ignore_index=True)
        return result