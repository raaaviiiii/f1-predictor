# src/models/championship.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import MODELS_DIR, LIGHTGBM_DEFAULTS
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHAMPIONSHIP_FEATURES = [
    "pts_before_race",
    "pts_gap_to_leader",
    "wdc_rank_before_race",
    "elo_rating",
    "elo_delta_last_race",
    "roll_avg_points_3",
    "roll_avg_points_5",
    "roll_avg_points_10",
    "roll_avg_finish_3",
    "roll_avg_finish_5",
    "constructor_pts_before_race",
    "constructor_pts_gap",
    "constructor_roll_pts_3",
    "constructor_roll_pts_5",
    "constructor_elo",
    "season_progress",
    "races_remaining",
    "regulation_era_age",
    "dnf_rate_10",
    "quali_position",
    "grid_position",
    "overtake_index",
]

CONSTRUCTOR_FEATURES = [
    "constructor_pts_before_race",
    "constructor_pts_gap",
    "constructor_roll_pts_3",
    "constructor_roll_pts_5",
    "constructor_elo",
    "wcc_rank_before_race",
    "season_progress",
    "races_remaining",
    "regulation_era_age",
]


class ChampionshipModel:

    def __init__(self):
        self.wdc_model    = None
        self.wcc_model    = None
        self.wdc_features = CHAMPIONSHIP_FEATURES
        self.wcc_features = CONSTRUCTOR_FEATURES

    def train(
        self,
        features_df: pd.DataFrame,
        standings_df: pd.DataFrame,
        train_seasons: list,
        test_seasons: list,
    ) -> dict:
        logger.info("Training championship models...")
        wdc_metrics = self._train_wdc(features_df, standings_df, train_seasons, test_seasons)
        wcc_metrics = self._train_wcc(features_df, standings_df, train_seasons, test_seasons)
        return {**wdc_metrics, **wcc_metrics}

    def predict_season(
        self,
        features_df: pd.DataFrame,
        season: int,
        after_round: int,
    ) -> dict:
        if self.wdc_model is None:
            raise ValueError("Model not trained.")

        season_df = features_df[
            (features_df["season"] == season) &
            (features_df["round"] == after_round)
        ].copy()

        if season_df.empty:
            raise ValueError(f"No data for season {season} round {after_round}")

        # WDC
        wdc_available = [f for f in self.wdc_features if f in season_df.columns]
        X_wdc = season_df[wdc_available].fillna(0)
        season_df["projected_final_pts"] = self.wdc_model.predict(X_wdc)

        wdc = (
            season_df.groupby("driver")
            .agg(
                team=("team", "last"),
                current_pts=("pts_before_race", "last"),
                projected_final_pts=("projected_final_pts", "mean"),
            )
            .reset_index()
            .sort_values("projected_final_pts", ascending=False)
            .reset_index(drop=True)
        )
        wdc["position"] = wdc.index + 1

        # WCC
        con_df = (
            season_df.groupby("team")
            .agg({f: "mean" for f in self.wcc_features if f in season_df.columns})
            .reset_index()
        )
        wcc_available = [f for f in self.wcc_features if f in con_df.columns]
        X_wcc = con_df[wcc_available].fillna(0)
        con_df["projected_final_pts"] = self.wcc_model.predict(X_wcc)

        con_pts = (
            season_df.groupby("team")["constructor_pts_before_race"]
            .max()
            .reset_index()
            .rename(columns={"constructor_pts_before_race": "current_pts"})
        )
        wcc = con_df.merge(con_pts, on="team", how="left")
        wcc = wcc[["team", "current_pts", "projected_final_pts"]].sort_values(
            "projected_final_pts", ascending=False
        ).reset_index(drop=True)
        wcc["position"] = wcc.index + 1

        return {"wdc": wdc, "wcc": wcc}

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "championship.pkl"
        joblib.dump({
            "wdc_model":    self.wdc_model,
            "wcc_model":    self.wcc_model,
            "wdc_features": self.wdc_features,
            "wcc_features": self.wcc_features,
        }, path)
        logger.info(f"  ✓ Championship model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "championship.pkl"
        obj = joblib.load(path)
        self.wdc_model    = obj["wdc_model"]
        self.wcc_model    = obj["wcc_model"]
        self.wdc_features = obj["wdc_features"]
        self.wcc_features = obj["wcc_features"]
        logger.info(f"  ✓ Championship model loaded from {path}")

    # ── Private ────────────────────────────────────────────────────────────

    def _build_wdc_targets(self, features_df, standings_df):
        final_pts = (
            standings_df.groupby(["season", "driver"])["driver_cum_points"]
            .max()
            .reset_index()
            .rename(columns={"driver_cum_points": "final_season_pts"})
        )
        return features_df.merge(final_pts, on=["season", "driver"], how="left")

    def _build_wcc_targets(self, features_df, standings_df):
        final_con_pts = (
            standings_df.groupby(["season", "team"])["constructor_cum_points"]
            .max()
            .reset_index()
            .rename(columns={"constructor_cum_points": "final_constructor_pts"})
        )
        return features_df.merge(final_con_pts, on=["season", "team"], how="left")

    def _train_wdc(self, features_df, standings_df, train_seasons, test_seasons):
        logger.info("  Training WDC model...")

        df = self._build_wdc_targets(features_df, standings_df)
        df = df[df["finish_position"] > 0].dropna(subset=["final_season_pts"])

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"].isin(test_seasons)]

        available = [f for f in self.wdc_features if f in df.columns]
        self.wdc_features = available

        X_train = train[available].fillna(0)
        y_train = train["final_season_pts"]
        X_test  = test[available].fillna(0)
        y_test  = test["final_season_pts"]

        self.wdc_model = LGBMRegressor(**LIGHTGBM_DEFAULTS)
        self.wdc_model.fit(X_train, y_train)

        metrics = {
            "wdc_train_mae": round(mean_absolute_error(y_train, self.wdc_model.predict(X_train)), 1),
            "wdc_test_mae":  round(mean_absolute_error(y_test,  self.wdc_model.predict(X_test)), 1),
        }
        logger.info(f"  WDC — Train MAE: {metrics['wdc_train_mae']} pts | "
                    f"Test MAE: {metrics['wdc_test_mae']} pts")
        return metrics

    def _train_wcc(self, features_df, standings_df, train_seasons, test_seasons):
        logger.info("  Training WCC model...")

        df = self._build_wcc_targets(features_df, standings_df)
        df = df[df["finish_position"] > 0].dropna(subset=["final_constructor_pts"])

        con_df = (
            df.groupby(["season", "round", "team"])
            .agg({**{f: "mean" for f in self.wcc_features if f in df.columns},
                  "final_constructor_pts": "first"})
            .reset_index()
        )

        train = con_df[con_df["season"].isin(train_seasons)]
        test  = con_df[con_df["season"].isin(test_seasons)]

        available = [f for f in self.wcc_features if f in con_df.columns]
        self.wcc_features = available

        X_train = train[available].fillna(0)
        y_train = train["final_constructor_pts"]
        X_test  = test[available].fillna(0)
        y_test  = test["final_constructor_pts"]

        self.wcc_model = LGBMRegressor(**LIGHTGBM_DEFAULTS)
        self.wcc_model.fit(X_train, y_train)

        metrics = {
            "wcc_train_mae": round(mean_absolute_error(y_train, self.wcc_model.predict(X_train)), 1),
            "wcc_test_mae":  round(mean_absolute_error(y_test,  self.wcc_model.predict(X_test)), 1),
        }
        logger.info(f"  WCC — Train MAE: {metrics['wcc_train_mae']} pts | "
                    f"Test MAE: {metrics['wcc_test_mae']} pts")
        return metrics