import os
import sys
import logging
import pandas as pd
import joblib
import mlflow
import warnings
import xgboost as xgb
from typing import Dict, Any
from skopt.space import Real, Integer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import config
from src.models.hyperparameter_tuning import tune_xgboost_hyperparameters


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def train_and_tune(
    X: pd.DataFrame, y: pd.Series, params_space: Dict[str, Any]
) -> xgb.XGBRegressor:
    """
    Orchestrates XGBoost hyperparameter tuning and model training.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target.
        params_space (dict): Dictionary defining the hyperparameter search space.

    Returns:
        xgb.XGBRegressor: The best trained XGBoost model.
    """
    logger.info("Starting XGBoost model training and hyperparameter tuning...")

    return tune_xgboost_hyperparameters(X, y, params_space)


def save_model_locally(model: Any, path: str) -> None:
    """
    Saves the trained model locally using joblib.

    Args:
        model (object): The trained model object.
        path (str): The file path to save the model.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved locally to {path}")
    except Exception as e:
        logger.error(f"Error saving the model locally: {e}")


def main() -> None:
    """
    Main function to run the model training pipeline.
    """
    data_path = config.TRAIN_FEATURES_PATH
    model_save_path = config.BEST_MODEL_PATH

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded from {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(
            f"Error: Data file not found at {data_path}. Please check the path."
        )
        return
    except Exception as e:
        logger.error(f"Error loading the data: {e}")
        return

    X = df.drop("Price", axis=1)
    y = df["Price"]

    param_space = {
        "n_estimators": Integer(100, 1000),
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "subsample": Real(0.5, 1.0, prior="uniform"),
        "colsample_bytree": Real(0.5, 1.0, prior="uniform"),
        "gamma": Real(0, 0.5, prior="uniform"),
        "reg_alpha": Real(0, 1, prior="uniform"),
        "reg_lambda": Real(1, 5, prior="uniform"),
    }

    logger.info("Model training and tuning started.")
    best_trained_model = train_and_tune(X, y, param_space)
    logger.info("Model training and tuning finished.")

    if best_trained_model:
        save_model_locally(best_trained_model, model_save_path)
        logger.info("Best model processing completed.")


if __name__ == "__main__":
    main()
