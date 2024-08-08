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
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def load_data(input_filepath):
    """
    Load data from a CSV file, apply feature engineering, and split into features and target.

    Args:
        input_filepath (Path): The path to the input CSV file.

    Returns:
        X (DataFrame): The feature matrix.
        y (Series): The target variable.
    """
    df = pd.read_csv(input_filepath)
    X = df.drop("Price", axis=1, errors="ignore")
    y = df["Price"]
    return X, y


def predict_and_evaluate(X, y, model):
    """
    Make predictions and evaluate the model.

    Args:
        X (DataFrame): The feature matrix.
        y (Series): The target variable.
        model: The trained model.

    Returns:
        predictions (ndarray): The model's predictions.
        rmse (float): The root mean squared error.
    """
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f"RMSE: {rmse:.2f}")

    mae = mean_absolute_error(y, predictions)
    print(f"MAE: {mae:.2f}")

    r2 = r2_score(y, predictions)
    print(f"R2_score: {r2:.2f}")

    cv_score = cross_val_score(
        model, y, predictions, cv=5, scoring="neg_root_mean_squared_error"
    )
    print(
        f"Cross validation score: {cv_score.mean():.2f} scoring=neg_root_mean_squared_error"
    )

    mape = mean_absolute_percentage_error(y, predictions)
    print(f"MAPE: {mape:.2f}")
    med = median_absolute_error(y, predictions)
    print(f"Median Absolute Error: {med:.2f}")

    return predictions, rmse, mae, r2, cv_score, mape, med


def main(input_filepath, model_filepath, output_filepath):
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
        predictions, rmse, mae, r2, cv_score, mape, med = predict_and_evaluate(
            X, y, model
        )
        df = pd.DataFrame({"Predicted_Price": predictions})
        df.to_csv(output_filepath, index=False)
    except FileNotFoundError:
        logger.error(f"File not found: {input_filepath}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = Path("data/processed/test_features.csv")
    model_filepath = Path("models/model.joblib")
    output_filepath = Path("data/predictions/predictions.csv")

    main(input_filepath, model_filepath, output_filepath)
