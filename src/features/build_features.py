import pandas as pd
import os
import logging
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from feature_engine.datetime import DatetimeFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from pandas import DataFrame
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import config

warnings.filterwarnings("ignore")


def features_engineering(df: DataFrame) -> DataFrame:
    """
    Engineers features for the flight price prediction model.

    Args:
        df (DataFrame): The input DataFrame with processed data.

    Returns:
        DataFrame: The DataFrame with engineered features.
    """
    airline_encoder = TargetEncoder()
    df["Airline"] = airline_encoder.fit_transform(df["Airline"], df["Price"])

    source_encoder = TargetEncoder()
    df["Source"] = source_encoder.fit_transform(df["Source"], df["Price"])

    destination_encoder = TargetEncoder()
    df["Destination"] = destination_encoder.fit_transform(
        df["Destination"], df["Price"]
    )

    os.makedirs(config.ENCODER_DIR, exist_ok=True)
    joblib.dump(airline_encoder, config.AIRLINE_ENCODER_PATH)
    joblib.dump(source_encoder, config.SOURCE_ENCODER_PATH)
    joblib.dump(destination_encoder, config.DESTINATION_ENCODER_PATH)

    addition_info_pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    duration_pipeline = Pipeline(
        [
            ("minmax", MinMaxScaler()),
        ]
    )

    doj_pipeline = Pipeline(
        [
            (
                "extractor_doj",
                DatetimeFeatures(
                    features_to_extract=["month", "day_of_week", "day_of_month"]
                ),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )

    dep_time_pipeline = Pipeline(
        [
            (
                "extractor_time",
                DatetimeFeatures(features_to_extract=["hour", "minute"]),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )

    arr_time_pipeline = Pipeline(
        [
            (
                "extractor_time",
                DatetimeFeatures(features_to_extract=["hour", "minute"]),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("duration", duration_pipeline, ["Duration"]),
            ("doj", doj_pipeline, ["Date_of_Journey"]),
            ("dep_time", dep_time_pipeline, ["Dep_Time"]),
            ("arr_time", arr_time_pipeline, ["Arrival_Time"]),
            ("addition_info", addition_info_pipeline, ["Additional_Info"]),
        ]
    )

    preprocessed = preprocessor.fit_transform(df)
    preprocessed_df = pd.DataFrame(
        preprocessed, columns=preprocessor.get_feature_names_out()
    )

    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    df = pd.concat([df, preprocessed_df], axis=1)
    df.drop(
        ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Additional_Info"],
        axis=1,
        inplace=True,
    )
    df.columns = df.columns.astype(str)

    return df


def main() -> None:
    """
    Main function to run the feature engineering pipeline.
    """
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    input_path = config.PROCESSED_DATA_PATH
    train_output_path = config.TRAIN_FEATURES_PATH
    test_output_path = config.TEST_FEATURES_PATH

    input_df = pd.read_csv(input_path)
    input_df = features_engineering(input_df)
    train_df, test_df = train_test_split(input_df, test_size=0.2, random_state=42)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    logging.info(f"data saved to {train_output_path} and {test_output_path}")


if __name__ == "__main__":
    main()
