from xgboost import XGBRegressor
import pandas as pd
import joblib
from pathlib import Path
import mlflow
import mlflow.xgboost


def train(df):
    """
    Trains a XGBoost Regressor model on the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        XGBRegressor: The trained XGBoost Regressor model.
    """
    mlflow.xgboost.autolog()
    x = df.drop(["Price"], axis=1)
    y = df["Price"]
    model = XGBRegressor()
    model.fit(x, y)
    return model


def save(model, path):
    """
    Saves the given model to the specified path.

    Args:
        model: The model to be saved.
        path (str or Path): The path to save the model to.
    """
    joblib.dump(model, path)


def main():
    """
    The main function.
    """
    # Set the file path for the dataset
    file_path = Path("d:/flight_price_prediction/data/processed/train_features.csv")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Train the model
    model = train(df)
    
    # Set the model save path
    model_path = Path("d:/flight_price_prediction/models/model.joblib")
    
    # Save the model
    save(model, model_path)
    
    # Log the model artifact with MLflow
    with mlflow.start_run() as run:
        mlflow.log_artifact(model_path)
        mlflow.end_run()

# Run the main function
if __name__ == "__main__":
    main()
