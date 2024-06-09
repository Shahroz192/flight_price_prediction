import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder

def airline_encoding(df):
    airline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    encoded_airline = airline.fit_transform(df[["Airline"]])
    encoded_airline_df = pd.DataFrame(encoded_airline, columns=airline.named_steps['ohe'].get_feature_names_out(["Airline"]))
    df = df.join(encoded_airline_df)
    return df

def source_encoding(df):
    source = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rle", RareLabelEncoder(tol=0.01, n_categories=5, replace_with="Rare")),
        ]
    )
    df["Source"] = source.fit_transform(df[["Source"]])
    df = pd.get_dummies(df, columns=["Source"], prefix="Source")
    return df

def destination_encoding(df):
    destination = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rle", RareLabelEncoder(tol=0.01, n_categories=5, replace_with="Rare")),
        ]
    )
    df["Destination"] = destination.fit_transform(df[["Destination"]])
    df = pd.get_dummies(df, columns=["Destination"], prefix="Destination")
    return df

def date_of_journey(df):
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
    df["Journey_month"] = df["Date_of_Journey"].dt.month
    df["Journey_dayofweek"] = df["Date_of_Journey"].dt.dayofweek
    df["Journey_dayofyear"] = df["Date_of_Journey"].dt.dayofyear
    return df

def departure_time(df):
    df["Dep_Time"] = pd.to_datetime(df["Dep_Time"])
    df["Dep_hour"] = df["Dep_Time"].dt.hour
    df["Dep_minute"] = df["Dep_Time"].dt.minute
    return df

def arrival_time(df):
    df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"])
    df["Arrival_hour"] = df["Arrival_Time"].dt.hour
    df["Arrival_minute"] = df["Arrival_Time"].dt.minute
    return df

def additional_information(df):
    info=Pipeline(
        steps=[
            ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    df["Additional_Info"] = info.fit_transform(df[["Additional_Info"]])
    return df

def scale_features(df):
    scaler = StandardScaler()
    features_to_scale = ["Journey_dayofyear", "Dep_hour", "Dep_minute", "Arrival_hour", "Arrival_minute", "Duration"]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df

def features_engineering(df):
    df = airline_encoding(df)
    df = source_encoding(df)
    df = destination_encoding(df)
    df = date_of_journey(df)
    df = departure_time(df)
    df = arrival_time(df)
    df = scale_features(df)
    df = additional_information(df)
    return df

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    # Hardcoded file paths
    input_filepath = Path("d:/flight_price_prediction/data/processed/flight_price_processed.csv")

    try:
        df = pd.read_csv(input_filepath)
        df = features_engineering(df)
        df.to_csv("d:/flight_price_prediction/data/processed/flight_price_features.csv", index=False)
        logger.info("Processed data saved.")
        logger.info("Features engineering completed.")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
