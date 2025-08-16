import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from typing import Any, Tuple
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src import config

warnings.filterwarnings("ignore")


def load_data(input_filepath: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from a CSV file and split into features and target.

    Args:
        input_filepath (Path): The path to the input CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix and the target variable.
    """
    df = pd.read_csv(input_filepath)
    X = df.drop("Price", axis=1, errors="ignore")
    y = df["Price"]
    return X, y


def predict(X: pd.DataFrame, model: Any) -> Tuple[np.ndarray]:
    """
    Make predictions and evaluate the model.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        model (Any): The trained model.

    Returns:
        Tuple[np.ndarray]: A tuple containing predictions and evaluation metrics.
    """
    return model.predict(X)


def evaluate(
    y: pd.Series, predictions: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the model.

    Args:
        y (pd.Series): The target variable.
        predictions (np.ndarray): The predicted values.

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing evaluation metrics.
    """
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    med = median_absolute_error(y, predictions)
    return mse, mae, r2, mape, med


def main(input_filepath: Path, model_filepath: Path, output_filepath: Path) -> None:
    """
    Load data, load a model, make predictions, evaluate the model, and save the predictions.

    Args:
        input_filepath (Path): The path to the input CSV file.
        model_filepath (Path): The path to the trained model.
        output_filepath (Path): The path to save the predictions.
    """
    logger = logging.getLogger(__name__)
    logger.info("Predicting flight prices")

    try:
        X, y = load_data(input_filepath)
        model = joblib.load(model_filepath)
        predictions = predict(X, model)
        mse, mae, r2, mape, med = evaluate(y, predictions)
        df = pd.DataFrame({"Predicted_Price": predictions})
        df.to_csv(output_filepath, index=False)
    except FileNotFoundError:
        logger.error(f"File not found: {input_filepath}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = config.TEST_FEATURES_PATH
    model_filepath = config.BEST_MODEL_PATH
    output_filepath = config.PREDICTIONS_PATH

    main(input_filepath, model_filepath, output_filepath)
