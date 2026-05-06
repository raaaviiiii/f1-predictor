# src/features/engineer.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Transforms the master DataFrame into a model-ready feature matrix.

    Features built:
      - Rolling driver form (3, 5, 10 race windows)
      - Rolling constructor form
      - Teammate deltas (quali and race)
      - Pace vs field
      - Overtake index
      - Wet performance coefficient
      - Tyre / pit stop features
      - Championship gap features
      - Encoded categoricals
      - Target columns: is_winner, is_podium
    """

    def __init__(self):
        self._circuit_enc    = LabelEncoder()
        self._track_type_enc = LabelEncoder()
        self._era_enc        = LabelEncoder()
        self._fitted         = False

    def transform(
        self,
        df: pd.DataFrame,
        save: bool = True,
        is_inference: bool = False,
    ) -> pd.DataFrame:

        logger.info("Engineering features...")
        df = df.sort_values(["season", "round", "finish_position"]).copy()

        df = self._rolling_driver_features(df)
        df = self._rolling_constructor_features(df)
        df = self._teammate_features(df)
        df = self._pace_features(df)
        df = self._overtake_index(df)
        df = self._wet_performance(df)
        df = self._tyre_features(df)
        df = self._championship_gaps(df)
        df = self._encode_categoricals(df, fit=not is_inference)
        df = self._finalise(df)

        logger.info(f"Feature matrix: {len(df)} rows × {len(df.columns)} cols")

        if save:
            df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
            logger.info("  ✓ Saved features.parquet")

        return df

    # ── Rolling driver features ────────────────────────────────────────────

    def _rolling_driver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Rolling driver features...")
        df = df.sort_values(["driver", "season", "round"])

        for w in [3, 5, 10]:
            grp = df.groupby("driver")
            df[f"roll_avg_finish_{w}"] = (
                grp["finish_position"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            df[f"roll_avg_points_{w}"] = (
                grp["points_scored"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

        df["dnf_rate_10"] = (
            df.groupby("driver")["dnf"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
        df["positions_gained"] = df["grid_position"] - df["finish_position"]
        return df

    # ── Rolling constructor features ───────────────────────────────────────

    def _rolling_constructor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Rolling constructor features...")

        con_pts = (
            df.groupby(["season", "round", "team"])["points_scored"]
            .sum()
            .reset_index()
            .rename(columns={"points_scored": "constructor_round_pts"})
        )
        df = df.merge(con_pts, on=["season", "round", "team"], how="left")

        con_sorted = con_pts.sort_values(["team", "season", "round"])
        for w in [3, 5]:
            con_sorted[f"constructor_roll_pts_{w}"] = (
                con_sorted.groupby("team")["constructor_round_pts"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
            )

        df = df.merge(
            con_sorted[["season", "round", "team",
                        "constructor_roll_pts_3", "constructor_roll_pts_5"]],
            on=["season", "round", "team"],
            how="left",
        )
        return df

    # ── Teammate features ──────────────────────────────────────────────────

    def _teammate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Teammate features...")

        def _delta(group):
            group = group.copy()
            if len(group) < 2:
                group["quali_pos_vs_teammate"]  = 0.0
                group["finish_pos_vs_teammate"] = 0.0
                return group
            avg_q = group["quali_position"].mean()
            avg_f = group["finish_position"].mean()
            group["quali_pos_vs_teammate"]  = avg_q - group["quali_position"]
            group["finish_pos_vs_teammate"] = avg_f - group["finish_position"]
            return group

        result = (
            df.groupby(["season", "round", "team"], group_keys=False)
            .apply(_delta)
        )
        df["quali_pos_vs_teammate"]  = result["quali_pos_vs_teammate"]
        df["finish_pos_vs_teammate"] = result["finish_pos_vs_teammate"]
        return df

    # ── Pace features ──────────────────────────────────────────────────────

    def _pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Pace features...")

        # Lap time delta vs field median per race
        race_median = (
            df.groupby(["season", "round"])["fastest_lap_ms"]
            .median()
            .rename("field_median_lap_ms")
            .reset_index()
        )
        df = df.merge(race_median, on=["season", "round"], how="left")

        df["pace_vs_field"] = (
            (df["fastest_lap_ms"] - df["field_median_lap_ms"])
            / df["field_median_lap_ms"]
        ) * 100

        # Rolling average pace vs field (last 5 races)
        df = df.sort_values(["driver", "season", "round"])
        df["roll_pace_vs_field_5"] = (
            df.groupby("driver")["pace_vs_field"]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        return df

    # ── Overtake index ─────────────────────────────────────────────────────

    def _overtake_index(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["grid_position"] < 20) & (df["positions_gained"].abs() < 16)
        df["positions_gained_clean"] = df["positions_gained"].where(mask, np.nan)
        df["overtake_index"] = (
            df.groupby("driver")["positions_gained_clean"]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )
        return df

    # ── Wet performance ────────────────────────────────────────────────────

    def _wet_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["driver", "season", "round"])

        def _coeff(driver_df):
            driver_df = driver_df.copy()
            wet  = driver_df["rainfall"] == True
            dry  = ~wet

            wet_avg = driver_df.loc[wet, "finish_position"].shift(1).expanding().mean()
            dry_avg = driver_df.loc[dry, "finish_position"].shift(1).expanding().mean()

            wet_avg = wet_avg.reindex(driver_df.index).ffill()
            dry_avg = dry_avg.reindex(driver_df.index).ffill()

            driver_df["wet_perf_coeff"] = wet_avg - dry_avg
            return driver_df

        df = df.groupby("driver", group_keys=False).apply(_coeff)
        df["wet_perf_coeff"] = df["wet_perf_coeff"].fillna(0)
        return df

    # ── Tyre features ──────────────────────────────────────────────────────

    def _tyre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cpa = (
            df.groupby("circuit_id")["pit_stops"]
            .mean()
            .rename("circuit_avg_pit_stops")
            .reset_index()
        )
        df = df.merge(cpa, on="circuit_id", how="left")
        df["tyre_deg_index"] = df["pit_stops"] / df["circuit_avg_pit_stops"].replace(0, 1)

        cta = (
            df.groupby("circuit_id")["avg_track_temp_c"]
            .mean()
            .rename("circuit_avg_track_temp")
            .reset_index()
        )
        df = df.merge(cta, on="circuit_id", how="left")
        df["avg_track_temp_c"] = df["avg_track_temp_c"].fillna(
            df["circuit_avg_track_temp"]
        ).fillna(30.0)
        return df

    # ── Championship gaps ──────────────────────────────────────────────────

    def _championship_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        leader = (
            df.groupby(["season", "round"])["pts_before_race"]
            .max()
            .rename("leader_pts")
            .reset_index()
        )
        df = df.merge(leader, on=["season", "round"], how="left")
        df["pts_gap_to_leader"] = df["leader_pts"] - df["pts_before_race"]

        con_leader = (
            df.groupby(["season", "round"])["constructor_pts_before_race"]
            .max()
            .rename("con_leader_pts")
            .reset_index()
        )
        df = df.merge(con_leader, on=["season", "round"], how="left")
        df["constructor_pts_gap"] = df["con_leader_pts"] - df["constructor_pts_before_race"]
        return df

    # ── Encode categoricals ────────────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        for col, enc in [
            ("circuit_id",     self._circuit_enc),
            ("track_type",     self._track_type_enc),
            ("regulation_era", self._era_enc),
        ]:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna("unknown").astype(str)
            if fit:
                df[f"{col}_enc"] = enc.fit_transform(df[col])
            else:
                known = set(enc.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else enc.classes_[0])
                df[f"{col}_enc"] = enc.transform(df[col])

        self._fitted = True
        return df

    # ── Finalise ───────────────────────────────────────────────────────────

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_winner"] = (df["finish_position"] == 1).astype(int)
        df["is_podium"] = (df["finish_position"] <= 3).astype(int)

        drop_cols = [
            "driver_full", "event_name", "status", "compounds_used",
            "pole_time_ms", "circuit_avg_pit_stops", "circuit_avg_track_temp",
            "leader_pts", "con_leader_pts", "positions_gained_clean",
            "constructor_round_pts", "field_median_lap_ms",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df