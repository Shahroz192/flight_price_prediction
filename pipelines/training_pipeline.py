import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_model import main as train_model_main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_training_pipeline():
    """
    Runs the model training pipeline.
    """
    logger.info("Starting training pipeline...")

    try:
        train_model_main()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error running training pipeline: {e}")
        raise

    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    run_training_pipeline()
