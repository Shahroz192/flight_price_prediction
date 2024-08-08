import pandas as pd
from pathlib import Path
import logging
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from feature_engine.datetime import DatetimeFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import warnings

warnings.filterwarnings("ignore")


def features_engineering(df):
    airline_encoder = TargetEncoder()
    df["Airline"] = airline_encoder.fit_transform(df["Airline"], df["Price"])

    source_encoder = TargetEncoder()
    df["Source"] = source_encoder.fit_transform(df["Source"], df["Price"])

    destination_encoder = TargetEncoder()
    df["Destination"] = destination_encoder.fit_transform(
        df["Destination"], df["Price"]
    )

    joblib.dump(airline_encoder, "models/encoder/airline_encoder.joblib")
    joblib.dump(source_encoder, "models/encoder/source_encoder.joblib")
    joblib.dump(destination_encoder, "models/encoder/destination_encoder.joblib")

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

    joblib.dump(preprocessor, "models/encoder/preprocessor.joblib")

    df = pd.concat([df, preprocessed_df], axis=1)
    df.drop(
        ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Additional_Info"],
        axis=1,
        inplace=True,
    )
    df.columns = df.columns.astype(str)

    return df


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    input_path = Path("data/processed/processed_flight.csv")
    train_output_path = Path("data/processed/train_features.csv")
    test_output_path = Path("data/processed/test_features.csv")

    input_df = pd.read_csv(input_path)
    input_df = features_engineering(input_df)
    train_df, test_df = train_test_split(input_df, test_size=0.2, random_state=42)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    logging.info(f"data saved to {train_output_path} and {test_output_path}")


if __name__ == "__main__":
    main()
