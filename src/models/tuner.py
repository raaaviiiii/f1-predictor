# src/models/tuner.py
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class WinnerModelTuner:
    """
    Uses Optuna to find optimal XGBoost hyperparameters.
    Runs n_trials experiments and returns the best params.

    Usage:
        tuner = WinnerModelTuner()
        best_params = tuner.tune(features_df, train_seasons, test_seasons)
    """

    def __init__(self, n_trials: int = 50):
        self.n_trials   = n_trials
        self.best_params = None
        self.study       = None

    def tune(
        self,
        df: pd.DataFrame,
        train_seasons: list,
        test_seasons: list,
        features: list,
    ) -> dict:
        logger.info(f"Tuning with Optuna ({self.n_trials} trials)...")

        train_df = df[df["season"].isin(train_seasons) & (df["finish_position"] > 0)]
        test_df  = df[df["season"].isin(test_seasons)  & (df["finish_position"] > 0)]

        available = [f for f in features if f in df.columns]

        X_train = train_df[available].fillna(0)
        y_train = train_df["is_winner"]
        X_test  = test_df[available].fillna(0)
        y_test  = test_df["is_winner"]

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale = neg / pos

        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                "gamma":             trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 0, 2.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 0, 2.0),
                "scale_pos_weight":  scale,
                "random_state":      42,
                "n_jobs":           -1,
                "verbosity":         0,
                "eval_metric":       "logloss",
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            probs = model.predict_proba(X_test)[:, 1]
            return log_loss(y_test, probs)

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_params["scale_pos_weight"] = scale
        self.best_params["random_state"]     = 42
        self.best_params["n_jobs"]           = -1

        logger.info(f"  Best log-loss: {self.study.best_value:.4f}")
        logger.info(f"  Best params:   {self.best_params}")

        joblib.dump(self.best_params, MODELS_DIR / "best_params.pkl")
        logger.info(f"  ✓ Saved best params")

        return self.best_params