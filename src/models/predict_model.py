import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score
from src.features.build_features import features_engineering

def predict(df, model):
    """
    Predicts the price of a flight using a trained XGBoost Regressor model.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        model: The trained XGBoost Regressor model.

    Returns:
        pandas.Series: The predicted flight prices.
    """
    df = features_engineering(df)
    X = df.drop(["Price"], axis=1)
    predictions = model.predict(X)
    return predictions

def main():
    """
    The main function.
    """
    # Set the file path for the dataset
    file_path = Path("d:/flight_price_prediction/data/processed/test.csv")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Apply feature engineering
    df = features_engineering(df)

    # Set the model path
    model_path = Path("d:/flight_price_prediction/models/model.joblib")

    # Load the model
    model = joblib.load(model_path)

    # Make predictions
    predictions = predict(df, model)

    # Save the predictions
    predictions.to_csv("d:/flight_price_prediction/data/processed/test_predictions.csv", index=False)

    print("Predictions saved to d:/flight_price_prediction/data/processed/test_predictions.csv")

    # Evaluate the model
    mse = mean_squared_error(df["Price"], predictions)
    r2 = r2_score(df["Price"], predictions)
    print(f"Mean squared error: {mse}")
    print(f"R2 score: {r2}")

# Run the main function
if __name__ == "__main__":
    main()
