#!/usr/bin/env python3
# update.py — Run this after every race to update predictions
"""
Usage:
    python update.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.loader import KaggleLoader
from src.pipeline.assembler import MasterDatasetAssembler
from src.pipeline.elo import EloEngine
from src.features.engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("update")


def main():
    logger.info("Updating F1 data and features...")

    # Step 1: Re-download latest CSV data
    loader = KaggleLoader()
    data   = loader.run(seasons=list(range(2010, 2027)), save=True)

    # Step 2: Rebuild pipeline
    assembler         = MasterDatasetAssembler()
    master, standings = assembler.build(data, save=True)

    elo    = EloEngine()
    master = elo.compute_and_attach(master)

    from config.config import PROCESSED_DIR
    master.to_parquet(PROCESSED_DIR / "master_with_elo.parquet", index=False)

    fe       = FeatureEngineer()
    features = fe.transform(master, save=True)

    logger.info(f"Update complete. {len(features)} rows, "
                f"latest round: {features[features['season']==2026]['round'].max()}")
    logger.info("Run predict.py to see updated predictions.")


if __name__ == "__main__":
    main()