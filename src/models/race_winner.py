# src/models/race_winner.py
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import MODELS_DIR, XGBOOST_DEFAULTS
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    "pace_vs_field",
    "roll_pace_vs_field_5"
    "season_avg_finish",
    "season_avg_points",
    "season_win_count",
    "season_podium_count",
]


class RaceWinnerModel:

    def __init__(self):
        self.model         = None
        self.features      = WINNER_FEATURES
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

        train_df = df[df["season"].isin(train_seasons) & (df["finish_position"] > 0)].copy()
        test_df  = df[df["season"].isin(test_seasons)  & (df["finish_position"] > 0)].copy()

        available     = [f for f in self.features if f in df.columns]
        self.features = available

        X_train = train_df[available].fillna(0)
        y_train = train_df["is_winner"]
        X_test  = test_df[available].fillna(0)
        y_test  = test_df["is_winner"]

        logger.info(
            f"  Train: {len(X_train)} | Test: {len(X_test)} | "
            f"Features: {len(available)}"
        )

        neg   = (y_train == 0).sum()
        pos   = (y_train == 1).sum()
        scale = neg / pos
        logger.info(f"  scale_pos_weight: {scale:.1f}")

        params     = {**XGBOOST_DEFAULTS, "scale_pos_weight": scale}
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

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
        logger.info(f"  Top-1 accuracy:   {metrics['top1_accuracy']:.3f}")
        logger.info(f"  Top-3 accuracy:   {metrics['top3_accuracy']:.3f}")
        logger.info(f"  {'─'*40}")

        return metrics

    def predict(
        self,
        df: pd.DataFrame,
        circuit_id: str = None,
    ) -> pd.DataFrame:
        """
        Predicts win probability for each driver.
        If circuit_id is provided, applies a circuit history multiplier
        to adjust probabilities based on past performance at that circuit.
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        available = [f for f in self.features if f in df.columns]
        X         = df[available].fillna(0)
        probs     = self.model.predict_proba(X)[:, 1]

        result                    = df[["driver", "team"]].copy()
        result["win_probability"] = probs

        # ── Circuit history multiplier ─────────────────────────────────────
        if circuit_id is not None:
            try:
                history_path = Path(str(MODELS_DIR)) / "circuit_history.pkl"
                if history_path.exists():
                    circuit_history = joblib.load(history_path)
                    circuit_hist    = circuit_history[
                        circuit_history["circuit_id"] == circuit_id
                    ]

                    if not circuit_hist.empty:
                        field_avg_win_rate = 1 / max(len(result), 1)

                        for idx, row in result.iterrows():
                            driver      = row["driver"]
                            driver_hist = circuit_hist[
                                circuit_hist["driver"] == driver
                            ]

                            if (
                                not driver_hist.empty and
                                driver_hist["appearances"].iloc[0] >= 2
                            ):
                                win_rate    = driver_hist["hist_win_rate"].iloc[0]
                                appearances = driver_hist["appearances"].iloc[0]

                                # Weight increases with more appearances (max 0.4)
                                weight = min(appearances / 10, 0.4)

                                # Multiplier: positive = historically strong here
                                multiplier = 1.0 + weight * (
                                    (win_rate - field_avg_win_rate)
                                    / max(field_avg_win_rate, 0.001)
                                )
                                # Cap between 0.7x and 1.5x
                                multiplier = max(0.7, min(1.5, multiplier))

                                result.at[idx, "win_probability"] *= multiplier

            except Exception as e:
                logger.warning(f"Circuit history multiplier failed: {e}")

        # Normalise so probabilities sum to 1
        total = result["win_probability"].sum()
        if total > 0:
            result["win_probability"] = result["win_probability"] / total

        return result.sort_values(
            "win_probability", ascending=False
        ).reset_index(drop=True)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / "race_winner.pkl"
        joblib.dump({"model": self.model, "features": self.features}, path)
        logger.info(f"  ✓ Model saved to {path}")

    def load(self, path: Path = None):
        path = path or MODELS_DIR / "race_winner.pkl"
        obj          = joblib.load(path)
        self.model   = obj["model"]
        self.features = obj["features"]
        logger.info(f"  ✓ Model loaded from {path}")

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def _top1_accuracy(self, df: pd.DataFrame, probs: np.ndarray) -> float:
        df         = df.copy()
        df["prob"] = probs
        correct    = 0
        total      = 0
        for (season, rnd), grp in df.groupby(["season", "round"]):
            predicted = grp.loc[grp["prob"].idxmax(), "driver"]
            actual    = grp.loc[grp["is_winner"] == 1, "driver"]
            if actual.empty:
                continue
            if predicted == actual.values[0]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    def _topk_accuracy(
        self,
        df: pd.DataFrame,
        probs: np.ndarray,
        k: int = 3,
    ) -> float:
        df         = df.copy()
        df["prob"] = probs
        correct    = 0
        total      = 0
        for (season, rnd), grp in df.groupby(["season", "round"]):
            top_k  = grp.nlargest(k, "prob")["driver"].values
            actual = grp.loc[grp["is_winner"] == 1, "driver"]
            if actual.empty:
                continue
            if actual.values[0] in top_k:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0