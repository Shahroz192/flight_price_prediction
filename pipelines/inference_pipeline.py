import sys
import os
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.predict_model import main as predict_model_main

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference_pipeline():
    """
    Runs the model inference pipeline.
    """
    logger.info("Starting inference pipeline...")

    input_filepath = Path("data/processed/test_features.csv")
    model_filepath = Path("models/best_model.joblib")
    output_filepath = Path("data/predictions/predictions.csv")

    try:
        predict_model_main(input_filepath, model_filepath, output_filepath)
        logger.info("Inference pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error running inference pipeline: {e}")
        raise

    logger.info("Inference pipeline finished successfully.")


if __name__ == "__main__":
    run_inference_pipeline()
