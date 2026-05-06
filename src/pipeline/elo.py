# src/pipeline/elo.py
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import (
    ELO_INITIAL_RATING, ELO_K_FACTOR, ELO_DECAY_PER_YEAR,
    REGULATION_ERAS, F2_TO_F1_ELO_MAP, KNOWN_ROOKIES_BY_SEASON,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ERA_YEARS = sorted(REGULATION_ERAS.keys())


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class EloEngine:
    """
    Builds ELO ratings for every driver and constructor.
    Processes races chronologically and records the rating
    BEFORE each race (no data leakage).

    Key design decisions:
    - Pairwise updates: every pair of finishers is treated as a match
    - Off-season decay: 10% regression toward mean each year
    - Era decay: 35% constructor regression at regulation changes
    - Rookies seeded from F2 performance mapping
    """

    def __init__(self):
        self.driver_ratings:       dict = {}
        self.constructor_ratings:  dict = {}
        self.driver_prev_elo:      dict = {}
        self._last_season:         int  = None

    def compute_and_attach(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing ELO ratings...")

        df = df.sort_values(["season", "round"]).copy()
        df["elo_rating"]          = np.nan
        df["constructor_elo"]     = np.nan
        df["elo_delta_last_race"] = np.nan

        for (season, rnd), race_df in df.groupby(["season", "round"]):

            # Off-season decay
            if self._last_season is not None and season != self._last_season:
                self._apply_off_season_decay()

            # Regulation era constructor decay
            if season in ERA_YEARS and rnd == 1:
                self._apply_era_constructor_decay()

            # Seed new drivers
            for _, row in race_df.iterrows():
                drv  = row["driver"]
                team = row["team"]
                if drv not in self.driver_ratings:
                    self.driver_ratings[drv]  = self._seed_driver(drv, season)
                    self.driver_prev_elo[drv] = self.driver_ratings[drv]
                if team not in self.constructor_ratings:
                    self.constructor_ratings[team] = ELO_INITIAL_RATING

            # Record ELO BEFORE race
            for idx, row in race_df.iterrows():
                drv  = row["driver"]
                team = row["team"]
                df.at[idx, "elo_rating"]          = self.driver_ratings[drv]
                df.at[idx, "constructor_elo"]     = self.constructor_ratings[team]
                df.at[idx, "elo_delta_last_race"]  = (
                    self.driver_ratings[drv] - self.driver_prev_elo.get(drv, self.driver_ratings[drv])
                )

            # Update ELO AFTER race
            self._update_after_race(race_df)
            self._last_season = season

        top5 = sorted(self.driver_ratings.items(), key=lambda x: -x[1])[:5]
        logger.info(f"  ELO done. Top 5: {top5}")
        return df

    def _seed_driver(self, driver: str, season: int) -> float:
        rookie_map = {
            drv: s
            for s, drivers in KNOWN_ROOKIES_BY_SEASON.items()
            for drv in drivers
        }
        if driver in rookie_map:
            return F2_TO_F1_ELO_MAP["champion"]
        return ELO_INITIAL_RATING

    def _apply_off_season_decay(self):
        mean = np.mean(list(self.driver_ratings.values()))
        for drv in self.driver_ratings:
            self.driver_ratings[drv] = (
                self.driver_ratings[drv] * (1 - ELO_DECAY_PER_YEAR)
                + mean * ELO_DECAY_PER_YEAR
            )

    def _apply_era_constructor_decay(self):
        if not self.constructor_ratings:
            return
        mean  = np.mean(list(self.constructor_ratings.values()))
        decay = 0.35
        for team in self.constructor_ratings:
            self.constructor_ratings[team] = (
                self.constructor_ratings[team] * (1 - decay)
                + mean * decay
            )
        logger.info("  Applied constructor ELO era decay")

    def _update_after_race(self, race_df: pd.DataFrame):
        finishers = race_df[~race_df["dnf"]].sort_values("finish_position")
        dnfs      = race_df[race_df["dnf"]]
        ordered   = pd.concat([finishers, dnfs]).reset_index(drop=True)
        n         = len(ordered)

        new_driver = {
            row["driver"]: self.driver_ratings[row["driver"]]
            for _, row in ordered.iterrows()
        }
        new_con = self.constructor_ratings.copy()

        for i in range(n):
            for j in range(i + 1, n):
                drv_i  = ordered.iloc[i]["driver"]
                drv_j  = ordered.iloc[j]["driver"]
                team_i = ordered.iloc[i]["team"]
                team_j = ordered.iloc[j]["team"]

                r_i  = self.driver_ratings[drv_i]
                r_j  = self.driver_ratings[drv_j]
                rc_i = self.constructor_ratings[team_i]
                rc_j = self.constructor_ratings[team_j]

                e_i  = _expected_score(r_i, r_j)
                ec_i = _expected_score(rc_i, rc_j)

                k = ELO_K_FACTOR / max(1, n - 1)

                new_driver[drv_i]  += k * (1 - e_i)
                new_driver[drv_j]  += k * (0 - (1 - e_i))
                new_con[team_i]    += k * 0.5 * (1 - ec_i)
                new_con[team_j]    += k * 0.5 * (0 - (1 - ec_i))

        for drv in new_driver:
            self.driver_prev_elo[drv] = self.driver_ratings[drv]

        self.driver_ratings.update(new_driver)
        self.constructor_ratings.update(new_con)