import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.make_dataset import main as make_dataset_main
from src.features.build_features import main as build_features_main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_feature_pipeline():
    """
    Runs the entire feature engineering pipeline.
    1. Processes raw data into an interim format.
    2. Builds features for the model.
    """
    logger.info("Starting feature pipeline...")

    logger.info("Running make_dataset script...")
    try:
        make_dataset_main()
        logger.info("make_dataset script completed successfully.")
    except Exception as e:
        logger.error(f"Error running make_dataset script: {e}")
        raise

    logger.info("Running build_features script...")
    try:
        build_features_main()
        logger.info("build_features script completed successfully.")
    except Exception as e:
        logger.error(f"Error running build_features script: {e}")
        raise

    logger.info("Feature pipeline finished successfully.")


if __name__ == "__main__":
    run_feature_pipeline()
