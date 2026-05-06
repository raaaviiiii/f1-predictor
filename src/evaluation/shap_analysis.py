# src/evaluation/shap_analysis.py
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import PLOTS_DIR, MODELS_DIR
from src.models.race_winner import RaceWinnerModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapAnalyser:
    """
    Generates SHAP explanations for the race winner model.

    Produces:
      1. Feature importance bar chart (global)
      2. SHAP beeswarm plot (global)
      3. Per-race waterfall plot (local — why did model pick this driver?)
    """

    def __init__(self):
        self.model    = None
        self.explainer = None

    def load_model(self):
        self.model = RaceWinnerModel()
        self.model.load()
        logger.info("  Model loaded for SHAP analysis")

    def compute_shap(self, features_df: pd.DataFrame) -> tuple:
        """
        Computes SHAP values for the entire feature matrix.
        Returns (shap_values, X) tuple.
        """
        if self.model is None:
            self.load_model()

        available = [f for f in self.model.features if f in features_df.columns]
        X = features_df[available].fillna(0)

        logger.info(f"  Computing SHAP values for {len(X)} rows...")
        self.explainer  = shap.TreeExplainer(self.model.model)
        shap_values     = self.explainer.shap_values(X)

        return shap_values, X

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        top_n: int = 20,
        save: bool = True,
    ):
        """Global feature importance — mean absolute SHAP value per feature."""
        logger.info("  Plotting feature importance...")

        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns,
        ).sort_values(ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(mean_shap.index, mean_shap.values, color="#E8393A")
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Feature Importance — Race Winner Model\n(Mean absolute SHAP value)", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels
        for bar, val in zip(bars, mean_shap.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "feature_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  ✓ Saved to {path}")

        plt.close()
        return mean_shap

    def plot_beeswarm(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        top_n: int = 15,
        save: bool = True,
    ):
        """Beeswarm plot — shows direction and magnitude of each feature."""
        logger.info("  Plotting beeswarm...")

        # Get top N features by importance
        mean_shap  = np.abs(shap_values).mean(axis=0)
        top_idx    = np.argsort(mean_shap)[-top_n:]
        X_top      = X.iloc[:, top_idx]
        sv_top     = shap_values[:, top_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv_top, X_top,
            show=False,
            plot_type="dot",
            color_bar=True,
        )
        plt.title("SHAP Beeswarm — Race Winner Model", fontsize=14, pad=20)
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / "shap_beeswarm.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  ✓ Saved to {path}")

        plt.close()

    def explain_race(
        self,
        features_df: pd.DataFrame,
        season: int,
        round_num: int,
        save: bool = True,
    ):
        """
        Waterfall plot for the top predicted driver in a specific race.
        Shows exactly WHY the model thinks they will win.
        """
        if self.model is None:
            self.load_model()

        race_df = features_df[
            (features_df["season"] == season) &
            (features_df["round"] == round_num)
        ].copy()

        if race_df.empty:
            logger.warning(f"  No data for {season} R{round_num}")
            return

        available  = [f for f in self.model.features if f in race_df.columns]
        X_race     = race_df[available].fillna(0)
        probs      = self.model.model.predict_proba(X_race)[:, 1]
        top_idx    = np.argmax(probs)
        top_driver = race_df.iloc[top_idx]["driver"]
        top_prob   = probs[top_idx]

        logger.info(f"  Explaining prediction for {top_driver} "
                    f"(win prob: {top_prob:.1%})")

        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model.model)

        shap_vals  = self.explainer.shap_values(X_race)
        sv_top     = shap_vals[top_idx]
        x_top      = X_race.iloc[top_idx]

        explanation = shap.Explanation(
            values=sv_top,
            base_values=self.explainer.expected_value,
            data=x_top.values,
            feature_names=available,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(
            f"Why {top_driver} is predicted to win\n"
            f"{season} Round {round_num} (prob: {top_prob:.1%})",
            fontsize=13, pad=20,
        )
        plt.tight_layout()

        if save:
            path = PLOTS_DIR / f"shap_race_{season}_r{round_num}_{top_driver}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  ✓ Saved to {path}")

        plt.close()
        return explanation