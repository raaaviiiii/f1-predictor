#!/usr/bin/env python3
# run_pipeline.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import SEASONS, PROCESSED_DIR
from src.pipeline.loader import KaggleLoader
from src.pipeline.assembler import MasterDatasetAssembler
from src.pipeline.elo import EloEngine
from src.features.engineer import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")


def main():
    logger.info("=" * 55)
    logger.info("  F1 PREDICTION SYSTEM — DATA PIPELINE")
    logger.info("=" * 55)

    # Step 1: Load
    logger.info("\n[1/4] Loading data...")
    loader = KaggleLoader()
    data   = loader.load_from_disk(seasons=SEASONS)

    # Step 2: Assemble
    logger.info("\n[2/4] Assembling master dataset...")
    assembler       = MasterDatasetAssembler()
    master, standings = assembler.build(data, save=True)

    # Step 3: ELO
    logger.info("\n[3/4] Computing ELO ratings...")
    elo    = EloEngine()
    master = elo.compute_and_attach(master)
    master.to_parquet(PROCESSED_DIR / "master_with_elo.parquet", index=False)

    # Step 4: Features
    logger.info("\n[4/4] Engineering features...")
    fe       = FeatureEngineer()
    features = fe.transform(master, save=True)

    logger.info("\n" + "=" * 55)
    logger.info("  Pipeline complete!")
    logger.info(f"  Feature matrix: {features.shape}")
    logger.info(f"  Saved to: data/processed/")
    logger.info("=" * 55)

    return features, standings


if __name__ == "__main__":
    main()