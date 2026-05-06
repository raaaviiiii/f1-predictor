# src/pipeline/assembler.py
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    PROCESSED_DIR, POINTS_SYSTEM, SPRINT_POINTS,
    REGULATION_ERAS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRACK_TYPES = {
    "abu_dhabi":     "street_hybrid",
    "bahrain":       "street_hybrid",
    "baku":          "street",
    "jeddah":        "street",
    "las_vegas":     "street",
    "miami":         "street_hybrid",
    "monaco":        "street",
    "singapore":     "street",
    "melbourne":     "street_hybrid",
    "lusail":        "street_hybrid",
    "barcelona":     "traditional",
    "budapest":      "traditional",
    "imola":         "traditional",
    "montreal":      "traditional",
    "monza":         "traditional",
    "austin":        "traditional",
    "mexico_city":   "traditional",
    "são_paulo":     "traditional",
    "silverstone":   "traditional",
    "spa-francorchamps": "traditional",
    "spielberg":     "traditional",
    "suzuka":        "traditional",
    "zandvoort":     "traditional",
    "shanghai":      "traditional",
    "sochi":         "street_hybrid",
    "portimão":      "traditional",
    "istanbul":      "traditional",
    "mugello":       "traditional",
    "nürburg":       "traditional",
    "yas_island":    "street_hybrid",
}

HIGH_OVERTAKE_CIRCUITS = {
    "bahrain", "monza", "spa-francorchamps", "austin",
    "mexico_city", "são_paulo", "lusail", "shanghai", "budapest",
}


class MasterDatasetAssembler:

    def build(
        self,
        data: dict,
        save: bool = True,
    ) -> tuple:
        logger.info("Building master dataset...")

        races   = data.get("races",      pd.DataFrame())
        qualis  = data.get("qualifying", pd.DataFrame())
        sprints = data.get("sprints",    pd.DataFrame())

        if races.empty:
            raise ValueError("No race data — run loader first.")

        # Step 1: Clean
        races = self._clean_races(races)

        # Step 2: Merge qualifying
        master = self._merge_qualifying(races, qualis)

        # Step 3: Merge sprints
        master = self._merge_sprints(master, sprints)

        # Step 4: Regulation era
        master = self._add_regulation_era(master)

        # Step 5: Track metadata
        master = self._add_track_metadata(master)

        # Step 6: Build standings
        standings_df = self._build_standings(master)

        # Step 7: Merge standings as pre-race features
        master = self._merge_standings(master, standings_df)

        # Step 8: Season progress
        master = self._add_season_progress(master)

        # Step 9: Sort
        master = master.sort_values(
            ["season", "round", "finish_position"]
        ).reset_index(drop=True)

        logger.info(
            f"Master dataset: {len(master)} rows × {len(master.columns)} cols | "
            f"Seasons: {sorted(master['season'].unique())}"
        )

        if save:
            master.to_parquet(PROCESSED_DIR / "master.parquet", index=False)
            standings_df.to_parquet(PROCESSED_DIR / "standings.parquet", index=False)
            logger.info("  ✓ Saved master.parquet and standings.parquet")

        return master, standings_df

    # ── Private helpers ────────────────────────────────────────────────────

    def _clean_races(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["season"]          = df["season"].astype(int)
        df["round"]           = df["round"].astype(int)
        df["grid_position"]   = pd.to_numeric(df["grid_position"], errors="coerce").fillna(0).astype(int)
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce").fillna(0).astype(int)
        df["dnf"]             = df["dnf"].astype(bool)
        df["pit_stops"]       = pd.to_numeric(df["pit_stops"], errors="coerce").fillna(1).astype(int)
        df["event_date"]      = pd.to_datetime(df["event_date"])

        # Remove rows where both grid and finish are 0 (non-starters)
        df = df[~((df["grid_position"] == 0) & (df["finish_position"] == 0))]

        # Pit lane starts get grid position 20
        df.loc[df["grid_position"] == 0, "grid_position"] = 20

        # Remove duplicates
        df = df.drop_duplicates(subset=["season", "round", "driver"])

        logger.info(f"  Races cleaned: {len(df)} rows")
        return df

    def _merge_qualifying(self, races: pd.DataFrame, qualis: pd.DataFrame) -> pd.DataFrame:
        if qualis.empty:
            logger.warning("  No qualifying data")
            races["quali_position"]       = np.nan
            races["best_quali_ms"]        = np.nan
            races["q1_time_ms"]           = np.nan
            races["q2_time_ms"]           = np.nan
            races["q3_time_ms"]           = np.nan
            races["quali_gap_to_pole_pct"] = np.nan
            return races

        qualis = qualis.drop_duplicates(subset=["season", "round", "driver"])
        cols   = ["season", "round", "driver", "quali_position",
                  "best_quali_ms", "q1_time_ms", "q2_time_ms", "q3_time_ms"]
        merged = races.merge(
            qualis[[c for c in cols if c in qualis.columns]],
            on=["season", "round", "driver"],
            how="left",
        )

        # Gap to pole as percentage
        pole = (
            merged.groupby(["season", "round"])["best_quali_ms"]
            .min()
            .rename("pole_time_ms")
            .reset_index()
        )
        merged = merged.merge(pole, on=["season", "round"], how="left")
        merged["quali_gap_to_pole_pct"] = (
            (merged["best_quali_ms"] - merged["pole_time_ms"])
            / merged["pole_time_ms"]
        ) * 100

        logger.info(f"  Merged qualifying. Shape: {merged.shape}")
        return merged

    def _merge_sprints(self, master: pd.DataFrame, sprints: pd.DataFrame) -> pd.DataFrame:
        if sprints.empty:
            master["sprint_points"]      = 0
            master["sprint_finish_pos"]  = np.nan
            return master

        sprints = sprints.drop_duplicates(subset=["season", "round", "driver"])
        cols    = ["season", "round", "driver", "sprint_points", "sprint_finish_pos"]
        master  = master.merge(
            sprints[[c for c in cols if c in sprints.columns]],
            on=["season", "round", "driver"],
            how="left",
        )
        master["sprint_points"]    = master["sprint_points"].fillna(0)
        master["sprint_finish_pos"] = master.get("sprint_finish_pos", np.nan)
        return master

    def _add_regulation_era(self, df: pd.DataFrame) -> pd.DataFrame:
        era_bins  = sorted(REGULATION_ERAS.keys())
        era_names = [REGULATION_ERAS[y] for y in era_bins]

        def _era(season):
            for i in range(len(era_bins) - 1, -1, -1):
                if season >= era_bins[i]:
                    return era_names[i]
            return "pre_hybrid"

        def _era_age(season):
            for i in range(len(era_bins) - 1, -1, -1):
                if season >= era_bins[i]:
                    return season - era_bins[i]
            return 0

        df["regulation_era"]     = df["season"].apply(_era)
        df["regulation_era_age"] = df["season"].apply(_era_age)
        return df

    def _add_track_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df["track_type"] = df["circuit_id"].map(TRACK_TYPES).fillna("traditional")
        df["high_overtake_circuit"] = df["circuit_id"].isin(HIGH_OVERTAKE_CIRCUITS).astype(int)
        return df

    def _build_standings(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Building championship standings...")

        df = df.copy()
        df["total_round_points"] = df["points_scored"] + df.get("sprint_points", pd.Series(0, index=df.index)).fillna(0)
        df = df.sort_values(["season", "round"])

        driver_records = []
        con_records    = []

        for season, s_df in df.groupby("season"):
            driver_points      = {}
            constructor_points = {}

            for rnd in sorted(s_df["round"].unique()):
                rnd_df = s_df[s_df["round"] == rnd]

                for _, row in rnd_df.iterrows():
                    drv  = row["driver"]
                    team = row["team"]
                    pts  = row["total_round_points"]
                    driver_points[drv]       = driver_points.get(drv, 0) + pts
                    constructor_points[team] = constructor_points.get(team, 0) + pts

                for drv, pts in driver_points.items():
                    driver_records.append({
                        "season":            season,
                        "after_round":       rnd,
                        "driver":            drv,
                        "driver_cum_points": pts,
                    })

                for team, pts in constructor_points.items():
                    con_records.append({
                        "season":                  season,
                        "after_round":             rnd,
                        "team":                    team,
                        "constructor_cum_points":  pts,
                    })

        standings = pd.DataFrame(driver_records)
        standings["wdc_rank"] = (
            standings.groupby(["season", "after_round"])["driver_cum_points"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        con_standings = pd.DataFrame(con_records)
        con_standings["wcc_rank"] = (
            con_standings.groupby(["season", "after_round"])["constructor_cum_points"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        # Attach team to driver standings
        driver_teams = (
            df[["season", "round", "driver", "team"]]
            .drop_duplicates()
            .rename(columns={"round": "after_round"})
        )
        standings = standings.merge(driver_teams, on=["season", "after_round", "driver"], how="left")
        standings = standings.merge(con_standings, on=["season", "after_round", "team"], how="left")

        logger.info(f"  Standings built: {len(standings)} rows")
        return standings

    def _merge_standings(self, master: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
        # Shift standings forward by 1 round to avoid leakage
        standings_shifted = standings.copy()
        standings_shifted["round"] = standings_shifted["after_round"] + 1

        slim = standings_shifted[[
            "season", "round", "driver",
            "driver_cum_points", "wdc_rank",
            "constructor_cum_points", "wcc_rank",
        ]].rename(columns={
            "driver_cum_points":      "pts_before_race",
            "wdc_rank":               "wdc_rank_before_race",
            "constructor_cum_points": "constructor_pts_before_race",
            "wcc_rank":               "wcc_rank_before_race",
        })

        master = master.merge(slim, on=["season", "round", "driver"], how="left")
        master["pts_before_race"]              = master["pts_before_race"].fillna(0)
        master["constructor_pts_before_race"]  = master["constructor_pts_before_race"].fillna(0)
        master["wdc_rank_before_race"]         = master["wdc_rank_before_race"].fillna(20)
        master["wcc_rank_before_race"]         = master["wcc_rank_before_race"].fillna(10)
        return master

    def _add_season_progress(self, df: pd.DataFrame) -> pd.DataFrame:
        total = df.groupby("season")["round"].max().rename("total_rounds")
        df    = df.merge(total, on="season", how="left")
        df["season_progress"] = df["round"] / df["total_rounds"]
        df["races_remaining"] = df["total_rounds"] - df["round"]
        return df