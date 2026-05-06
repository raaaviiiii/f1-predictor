# src/pipeline/loader.py
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    RAW_DIR, POINTS_SYSTEM, SPRINT_POINTS,
    FASTEST_LAP_POINT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DIR = Path(__file__).resolve().parents[2] / "data" / "kaggle"


class KaggleLoader:
    """
    Loads F1 data from the TracingInsights CSV dataset.
    Produces the same output format as the original FastF1 loader
    so the rest of the pipeline works unchanged.

    Usage:
        loader = KaggleLoader()
        data = loader.run(seasons=list(range(2018, 2026)))
    """

    def __init__(self):
        logger.info("Loading CSV files from data/kaggle/...")
        self.races        = pd.read_csv(KAGGLE_DIR / "races.csv")
        self.results      = pd.read_csv(KAGGLE_DIR / "results.csv")
        self.drivers      = pd.read_csv(KAGGLE_DIR / "drivers.csv")
        self.constructors = pd.read_csv(KAGGLE_DIR / "constructors.csv")
        self.circuits     = pd.read_csv(KAGGLE_DIR / "circuits.csv")
        self.qualifying   = pd.read_csv(KAGGLE_DIR / "qualifying.csv")
        self.pit_stops    = pd.read_csv(KAGGLE_DIR / "pit_stops.csv")
        self.lap_times    = pd.read_csv(KAGGLE_DIR / "lap_times.csv")
        self.sprints      = pd.read_csv(KAGGLE_DIR / "sprint_results.csv")
        self.status       = pd.read_csv(KAGGLE_DIR / "status.csv")
        logger.info("  ✓ All CSV files loaded")

    def run(
        self,
        seasons: list = None,
        save: bool = True,
    ) -> dict:
        from config.config import SEASONS
        seasons = seasons or SEASONS

        logger.info(f"Processing seasons: {seasons}")

        races_df   = self._build_races(seasons)
        quali_df   = self._build_qualifying(seasons)
        sprint_df  = self._build_sprints(seasons)

        if save:
            for season in seasons:
                r = races_df[races_df["season"] == season]
                q = quali_df[quali_df["season"] == season]
                s = sprint_df[sprint_df["season"] == season]

                if not r.empty:
                    r.to_parquet(RAW_DIR / f"season_{season}.parquet", index=False)
                if not q.empty:
                    q.to_parquet(RAW_DIR / f"quali_{season}.parquet", index=False)
                if not s.empty:
                    s.to_parquet(RAW_DIR / f"sprint_{season}.parquet", index=False)

            logger.info(f"  ✓ Saved parquet files to data/raw/")

        logger.info(
            f"\nDone. Races: {len(races_df)}, "
            f"Quali: {len(quali_df)}, "
            f"Sprints: {len(sprint_df)}"
        )

        return {
            "races":      races_df,
            "qualifying": quali_df,
            "sprints":    sprint_df,
        }

    def load_from_disk(self, seasons: list = None) -> dict:
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

    # ── Private builders ───────────────────────────────────────────────────

    def _build_races(self, seasons: list) -> pd.DataFrame:
        logger.info("  Building race results...")

        # Filter races to requested seasons
        races = self.races[self.races["year"].isin(seasons)].copy()

        # Merge results
        results = self.results.merge(races[["raceId", "year", "round", "name", "date", "circuitId"]], on="raceId")

        # Merge driver abbreviations
        drivers = self.drivers[["driverId", "code", "forename", "surname"]].copy()
        drivers["driver_full"] = drivers["forename"] + " " + drivers["surname"]
        results = results.merge(drivers, on="driverId")

        # Merge constructor names
        constructors = self.constructors[["constructorId", "name"]].rename(columns={"name": "team"})
        results = results.merge(constructors, on="constructorId")

        # Merge circuit names
        circuits = self.circuits[["circuitId", "location"]].copy()
        results = results.merge(circuits, on="circuitId")

        # Merge status
        results = results.merge(self.status, on="statusId")

        # Merge pit stops count per driver per race
        pit_counts = (
            self.pit_stops.groupby(["raceId", "driverId"])
            .size()
            .reset_index(name="pit_stops")
        )
        results = results.merge(pit_counts, on=["raceId", "driverId"], how="left")
        results["pit_stops"] = results["pit_stops"].fillna(0).astype(int)

        # Finish position
        results["finish_position"] = pd.to_numeric(results["position"], errors="coerce").fillna(0).astype(int)

        # DNF check
        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps"}
        results["dnf"] = ~results["status"].isin(finished_statuses)

        # Points — use our points system for consistency
        results["points_scored"] = results["finish_position"].map(POINTS_SYSTEM).fillna(0)

        # Fastest lap bonus (post-2019, rank=1 means fastest lap)
        results["fl_bonus"] = 0
        fl_mask = (
            (results["year"] >= 2019) &
            (results["rank"] == "1") &
            (results["finish_position"] <= 10)
        )
        results.loc[fl_mask, "fl_bonus"] = FASTEST_LAP_POINT
        results["points_scored"] = results["points_scored"] + results["fl_bonus"]

        # Fastest lap in ms
        def _lap_to_ms(t):
            try:
                parts = str(t).split(":")
                if len(parts) == 2:
                    return (float(parts[0]) * 60 + float(parts[1])) * 1000
                return np.nan
            except Exception:
                return np.nan

        results["fastest_lap_ms"] = results["fastestLapTime"].apply(_lap_to_ms)

        # Circuit ID
        results["circuit_id"] = results["location"].str.replace(" ", "_").str.lower()

        # Final columns matching original loader output
        output = pd.DataFrame({
            "season":           results["year"].astype(int),
            "round":            results["round"].astype(int),
            "circuit_id":       results["circuit_id"],
            "event_name":       results["name"],
            "event_date":       pd.to_datetime(results["date"]),
            "driver":           results["code"].fillna("UNK"),
            "driver_full":      results["driver_full"],
            "team":             results["team"],
            "grid_position":    pd.to_numeric(results["grid"], errors="coerce").fillna(0).astype(int),
            "finish_position":  results["finish_position"],
            "status":           results["status"],
            "dnf":              results["dnf"],
            "points_scored":    results["points_scored"],
            "fl_bonus":         results["fl_bonus"],
            "lap_count":        pd.to_numeric(results["laps"], errors="coerce").fillna(0).astype(int),
            "fastest_lap_ms":   results["fastest_lap_ms"],
            "median_lap_ms":    np.nan,   # not in this dataset
            "pit_stops":        results["pit_stops"],
            "compounds_used":   "",       # not in this dataset
            "avg_track_temp_c": np.nan,
            "avg_air_temp_c":   np.nan,
            "rainfall":         False,
        })

        return output.reset_index(drop=True)

    def _build_qualifying(self, seasons: list) -> pd.DataFrame:
        logger.info("  Building qualifying results...")

        races = self.races[self.races["year"].isin(seasons)][["raceId", "year", "round", "circuitId"]]
        quali = self.qualifying.merge(races, on="raceId")

        drivers = self.drivers[["driverId", "code"]].copy()
        quali = quali.merge(drivers, on="driverId")

        constructors = self.constructors[["constructorId", "name"]].rename(columns={"name": "team"})
        quali = quali.merge(constructors, on="constructorId")

        circuits = self.circuits[["circuitId", "location"]].copy()
        quali = quali.merge(circuits, on="circuitId")

        def _lap_to_ms(t):
            try:
                parts = str(t).split(":")
                if len(parts) == 2:
                    return (float(parts[0]) * 60 + float(parts[1])) * 1000
                return np.nan
            except Exception:
                return np.nan

        quali["q1_time_ms"] = quali["q1"].apply(_lap_to_ms)
        quali["q2_time_ms"] = quali["q2"].apply(_lap_to_ms)
        quali["q3_time_ms"] = quali["q3"].apply(_lap_to_ms)

        quali["best_quali_ms"] = quali[["q1_time_ms", "q2_time_ms", "q3_time_ms"]].min(axis=1)

        output = pd.DataFrame({
            "season":         quali["year"].astype(int),
            "round":          quali["round"].astype(int),
            "circuit_id":     quali["location"].str.replace(" ", "_").str.lower(),
            "driver":         quali["code"].fillna("UNK"),
            "team":           quali["team"],
            "quali_position": pd.to_numeric(quali["position"], errors="coerce").fillna(0).astype(int),
            "q1_time_ms":     quali["q1_time_ms"],
            "q2_time_ms":     quali["q2_time_ms"],
            "q3_time_ms":     quali["q3_time_ms"],
            "best_quali_ms":  quali["best_quali_ms"],
        })

        return output.reset_index(drop=True)

    def _build_sprints(self, seasons: list) -> pd.DataFrame:
        logger.info("  Building sprint results...")

        races = self.races[self.races["year"].isin(seasons)][["raceId", "year", "round", "circuitId"]]
        sprints = self.sprints.merge(races, on="raceId")

        if sprints.empty:
            return pd.DataFrame()

        drivers = self.drivers[["driverId", "code"]].copy()
        sprints = sprints.merge(drivers, on="driverId")

        constructors = self.constructors[["constructorId", "name"]].rename(columns={"name": "team"})
        sprints = sprints.merge(constructors, on="constructorId")

        circuits = self.circuits[["circuitId", "location"]].copy()
        sprints = sprints.merge(circuits, on="circuitId")

        sprints["sprint_finish_pos"] = pd.to_numeric(sprints["position"], errors="coerce").fillna(0).astype(int)
        sprints["sprint_points"] = sprints["sprint_finish_pos"].map(SPRINT_POINTS).fillna(0)

        output = pd.DataFrame({
            "season":            sprints["year"].astype(int),
            "round":             sprints["round"].astype(int),
            "circuit_id":        sprints["location"].str.replace(" ", "_").str.lower(),
            "driver":            sprints["code"].fillna("UNK"),
            "team":              sprints["team"],
            "sprint_finish_pos": sprints["sprint_finish_pos"],
            "sprint_points":     sprints["sprint_points"],
        })

        return output.reset_index(drop=True)


# Keep DataLoader as an alias so the rest of the code doesn't break
DataLoader = KaggleLoader