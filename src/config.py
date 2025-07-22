from pathlib import Path

# Data paths
RAW_DATA_PATH = Path("data/raw/flight_price.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_flight.csv")
TRAIN_FEATURES_PATH = Path("data/processed/train_features.csv")
TEST_FEATURES_PATH = Path("data/processed/test_features.csv")
PREDICTIONS_PATH = Path("data/predictions/predictions.csv")

# Model paths
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
ENCODER_DIR = MODEL_DIR / "encoder"
AIRLINE_ENCODER_PATH = ENCODER_DIR / "airline_encoder.joblib"
SOURCE_ENCODER_PATH = ENCODER_DIR / "source_encoder.joblib"
DESTINATION_ENCODER_PATH = ENCODER_DIR / "destination_encoder.joblib"
PREPROCESSOR_PATH = ENCODER_DIR / "preprocessor.joblib"

# MLflow settings
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "Flight Price Prediction"
