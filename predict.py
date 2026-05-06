#!/usr/bin/env python3
# predict.py — F1 Prediction CLI
"""
Usage:
    python predict.py --race "Bahrain" --season 2024 --round 1
    python predict.py --championship --season 2024 --round 10
    python predict.py --race "Monaco" --season 2026 --round 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from src.models.race_winner import RaceWinnerModel
from src.models.championship import ChampionshipModel
from src.utils.logger import get_logger

logger = get_logger("predict")


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          🏎   F1 PREDICTION SYSTEM  🏎               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def print_win_probabilities(predictions: pd.DataFrame, race_name: str, season: int):
    print(f"  WIN PROBABILITIES — {race_name.upper()} {season}")
    print(f"  {'─'*48}")
    print(f"  {'#':<4} {'Driver':<6} {'Team':<22} {'Probability':<12} {'Bar'}")
    print(f"  {'─'*48}")

    for i, row in predictions.head(10).iterrows():
        prob  = row["win_probability"] * 100
        bar   = "█" * int(prob / 2)
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        print(f"  {medal:<4} {row['driver']:<6} {row['team'][:20]:<22} {prob:>6.1f}%      {bar}")

    print()


def print_championship(projections: dict, season: int, after_round: int):
    wdc = projections["wdc"]
    wcc = projections["wcc"]

    print(f"  CHAMPIONSHIP PROJECTIONS — {season} (after round {after_round})")
    print(f"  {'─'*55}")

    print(f"\n  🏆 DRIVERS (WDC)")
    print(f"  {'Pos':<5} {'Driver':<8} {'Team':<22} {'Now':>6} {'Projected':>10}")
    print(f"  {'─'*55}")
    for _, row in wdc.head(10).iterrows():
        medal = ["🥇", "🥈", "🥉"][int(row['position'])-1] if row['position'] <= 3 else f"  {int(row['position'])}."
        print(
            f"  {medal:<5} {row['driver']:<8} {str(row['team'])[:20]:<22} "
            f"{int(row['current_pts']):>6} {int(row['projected_final_pts']):>10}"
        )

    print(f"\n  🏗  CONSTRUCTORS (WCC)")
    print(f"  {'Pos':<5} {'Team':<28} {'Now':>6} {'Projected':>10}")
    print(f"  {'─'*55}")
    for _, row in wcc.head(10).iterrows():
        medal = ["🥇", "🥈", "🥉"][int(row['position'])-1] if row['position'] <= 3 else f"  {int(row['position'])}."
        print(
            f"  {medal:<5} {str(row['team'])[:26]:<28} "
            f"{int(row['current_pts']):>6} {int(row['projected_final_pts']):>10}"
        )
    print()


def get_race_features(features_df: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    race_df = features_df[
        (features_df["season"] == season) &
        (features_df["round"] == round_num)
    ].copy()

    if race_df.empty:
        # For future races, use latest available round as proxy
        season_df = features_df[features_df["season"] == season]
        if season_df.empty:
            # Use last season's data as baseline
            last_season = features_df["season"].max()
            season_df   = features_df[features_df["season"] == last_season]

        latest_round = season_df["round"].max()
        race_df = season_df[season_df["round"] == latest_round].copy()
        logger.info(f"  Using round {latest_round} data as proxy for future race")

    return race_df


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="F1 Prediction System")
    parser.add_argument("--race",           type=str,  help="Race/circuit name (e.g. 'Monaco')")
    parser.add_argument("--season",         type=int,  default=2026, help="Season year")
    parser.add_argument("--round",          type=int,  default=1,    help="Round number")
    parser.add_argument("--championship",   action="store_true",     help="Show championship projections")
    parser.add_argument("--after-round",    type=int,  default=None, help="Championship state after this round")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("  ❌ No features found. Run python run_pipeline.py first.")
        sys.exit(1)

    features = pd.read_parquet(features_path)

    # ── Race prediction ────────────────────────────────────────────────────
    if args.race:
        print(f"  Loading race winner model...")
        winner_model = RaceWinnerModel()
        winner_model.load()

        race_df = get_race_features(features, args.season, args.round)

        if race_df.empty:
            print(f"  ❌ No data found for season {args.season} round {args.round}")
            sys.exit(1)

        predictions = winner_model.predict(race_df)
        print_win_probabilities(predictions, args.race, args.season)

    # ── Championship projection ────────────────────────────────────────────
    if args.championship:
        print(f"  Loading championship model...")
        champ_model = ChampionshipModel()
        champ_model.load()

        after_round = args.after_round or args.round

        # Find closest available round
        season_df = features[features["season"] == args.season]
        if season_df.empty:
            last_season = features["season"].max()
            print(f"  ℹ️  No {args.season} data — using {last_season} as proxy")
            args.season = last_season
            season_df   = features[features["season"] == last_season]

        available_rounds = sorted(season_df["round"].unique())
        if after_round not in available_rounds:
            after_round = max(r for r in available_rounds if r <= after_round) if any(r <= after_round for r in available_rounds) else available_rounds[-1]
            print(f"  ℹ️  Using round {after_round} as closest available")

        projections = champ_model.predict_season(features, args.season, after_round)
        print_championship(projections, args.season, after_round)

    if not args.race and not args.championship:
        parser.print_help()


if __name__ == "__main__":
    main()