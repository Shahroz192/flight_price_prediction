# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


def remove_outliers(df):
    """
    Removes rows with outliers in the 'Price' column.
    """
    for col in df.columns:
        if df[col].dtype == "int64" or df[col].dtype == "float64":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
    return df


def stop(df):
    """
    Maps the 'Total_Stops' column to numerical values.
    """
    df["Total_Stops"] = df["Total_Stops"].map(
        {
            "non-stop": 0,
            "1 stop": 1,
            "2 stops": 2,
            "3 stops": 3,
            "4 stops": 4,
            "nan": np.nan,
        }
    )
    df["Total_Stops"] = df["Total_Stops"].astype(int)
    return df


def duration(df):
    """
    Converts the 'Duration' column to numeric values (in minutes).
    """
    df["Duration"] = (
        df["Duration"]
        .str.replace("h", "*60")
        .str.replace(" ", "+")
        .str.replace("m", "*1")
        .apply(eval)
    )
    df["Duration"] = df["Duration"].astype(int)
    return df


def to_datetime(df):
    """
    Converts the 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns to datetime format.
    """
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"])
    df["Dep_Time"] = pd.to_datetime(df["Dep_Time"])
    df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"])
    return df


def cleaning(df):
    """
    Performs data cleaning and preprocessing on the input DataFrame.
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(["Route"], axis=1, inplace=True)
    df = duration(df)
    df = stop(df)
    df = to_datetime(df)
    df = remove_outliers(df)
    return df


def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    # Hardcoded file paths
    input_filepath = Path("d:/flight_price_prediction/data/raw/flight_price.csv")

    try:
        df = pd.read_csv(input_filepath)
        cleaned_df = cleaning(df)
        cleaned_df.to_csv(
            "d:/flight_price_prediction/data/processed/flight_price_processed.csv",
            index=False,
        )

        logger.info("Processed data saved.")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
