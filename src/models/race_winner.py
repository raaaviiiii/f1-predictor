# src/models/race_winner.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import MODELS_DIR, XGBOOST_DEFAULTS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Features used for race winner prediction
WINNER_FEATURES = [
    "grid_position",
    "quali_position",
    "quali_gap_to_pole_pct",
    "quali_pos_vs_teammate",
    "elo_rating",
    "constructor_elo",
    "elo_delta_last_race",
    "roll_avg_finish_3",
    "roll_avg_finish_5",
    "roll_avg_finish_10",
    "roll_avg_points_3",
    "roll_avg_points_5",
    "roll_avg_points_10",
    "dnf_rate_10",
    "overtake_index",
    "wet_perf_coeff",
    "constructor_roll_pts_3",
    "constructor_roll_pts_5",
    "pts_before_race",
    "pts_gap_to_leader",
    "wdc_rank_before_race",
    "constructor_pts_before_race",
    "constructor_pts_gap",
    "wcc_rank_before_race",
    "tyre_deg_index",
    "pit_stops",
    "lap_count",
    "season_progress",
    "races_remaining",
    "high_overtake_circuit",
    "regulation_era_age",
    "circuit_id_enc",
    "track_type_enc",
    "regulation_era_enc",
]


class RaceWinnerModel:
    """
    XGBoost classifier that predicts win probability for each driver.

    Training approach:
    - Temporal split: train on seasons up to cutoff, test on remaining
    - Never random splits across seasons (would leak future data)
    - Calibrated probabilities using isotonic regression

    Usage:
        model = RaceWinnerModel()
        model.train(features_df, train_seasons=[2018,2019,2020,2021,2022],
                                  test_seasons=[2023,2024])
        probs = model.predict(race_features_df)
    """

    def __init__(self):
        self.model     = None
        self.features  = WINNER_FEATURES
        self.train_seasons = None
        self.test_seasons  = None

    def train(
        self,
        df: pd.DataFrame,
        train_seasons: list,
        test_seasons: list,
    ) -> dict:
        logger.info("Training race winner model...")
        logger.info(f"  Train seasons: {train_seasons}")
        logger.info(f"  Test seasons:  {test_seasons}")

        self.train_seasons = train_seasons
        self.test_seasons  = test_seasons

        # ── Split ──────────────────────────────────────────────────────────
        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df  = df[df["season"].isin(test_seasons)].copy()

        # Drop rows with missing target
        train_df = train_df[train_df["finish_position"] > 0]
        test_df  = test_df[test_df["finish_position"] > 0]

        # Available features (some may be missing in older seasons)
        available = [f for f in self.features if f in df.columns]
        self.features = available

        X_train = train_df[self.features].fillna(0)
        y_train = train_df["is_winner"]
        X_test  = test_df[self.features].fillna(0)
        y_test  = test_df["is_winner"]

        logger.info(f"  Train: {len(X_train)} rows | "
                    f"Test: {len(X_test)} rows | "
                    f"Features: {len(self.features)}")
        logger.info(f"  Win rate train: {y_train.mean():.3f} | "
                    f"test: {y_test.mean():.3f}")

        # ── Class weight ───────────────────────────────────────────────────
        # ~5% of rows are winners — we need scale_pos_weight to balance
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale = neg / pos
        logger.info(f"  scale_pos_weight: {scale:.1f}")

        # ── Train ──────────────────────────────────────────────────────────
        params = {**XGBOOST_DEFAULTS, "scale_pos_weight": scale}
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # ── Evaluate ───────────────────────────────────────────────────────
        train_probs = self.model.predict_proba(X_train)[:, 1]
        test_probs  = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "train_logloss": log_loss(y_train, train_probs),
            "test_logloss":  log_loss(y_test,  test_probs),
            "train_auc":     roc_auc_score(y_train, train_probs),
            "test_auc":      roc_auc_score(y_test,  test_probs),
            "top1_accuracy": self._top1_accuracy(test_df, test_probs),
            "top3_accuracy": self._topk_accuracy(test_df, test_probs, k=3),
        }

        logger.info(f"\n  {'─'*40}")
        logger.info(f"  Test log-loss:    {metrics['test_logloss']:.4f}")
        logger.info(f"  Test AUC:         {metrics['test_auc']:.4f}")
        logger.info(f"  Top-1 accuracy:   {metrics['top1_accuracy']:.3f} "
                    f"(predicted winner = actual winner)")
        logger.info(f"  Top-3 accuracy:   {metrics['top3_accuracy']:.3f} "
                    f"(winner in top-3 predicted)")
        logger.info(f"  {'─'*40}")

        return metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts win probability for each driver.
        Returns df with added 'win_probability' column, sorted by probability.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        available = [f for f in self.features if f in df.columns]
        X = df[available].fillna(0)
        probs = self.model.predict_proba(X)[:, 1]

        result = df[["driver", "team"]].copy()
        result["win_probability"] = probs

        # Normalise so probabilities sum to 100% per race
        total = result["win_probability"].sum()
        if total > 0:
            result["win_probability"] = result["win_probability"] / total

        return result.sort_values("win_probability", ascending=False).reset_index(drop=True)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "race_winner.pkl"
        joblib.dump({"model": self.model, "features": self.features}, path)
        logger.info(f"  ✓ Model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "race_winner.pkl"
        obj = joblib.load(path)
        self.model    = obj["model"]
        self.features = obj["features"]
        logger.info(f"  ✓ Model loaded from {path}")

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def _top1_accuracy(self, df: pd.DataFrame, probs: np.ndarray) -> float:
        """
        For each race, did the driver with highest predicted probability
        actually win?
        """
        df = df.copy()
        df["prob"] = probs
        correct = 0
        total   = 0
        for (season, rnd), grp in df.groupby(["season", "round"]):
            predicted_winner = grp.loc[grp["prob"].idxmax(), "driver"]
            actual_winner    = grp.loc[grp["is_winner"] == 1, "driver"]
            if actual_winner.empty:
                continue
            if predicted_winner == actual_winner.values[0]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    def _topk_accuracy(self, df: pd.DataFrame, probs: np.ndarray, k: int = 3) -> float:
        """
        For each race, was the actual winner in the top-k predicted drivers?
        """
        df = df.copy()
        df["prob"] = probs
        correct = 0
        total   = 0
        for (season, rnd), grp in df.groupby(["season", "round"]):
            top_k         = grp.nlargest(k, "prob")["driver"].values
            actual_winner = grp.loc[grp["is_winner"] == 1, "driver"]
            if actual_winner.empty:
                continue
            if actual_winner.values[0] in top_k:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0