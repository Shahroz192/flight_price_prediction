import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
import warnings
from pandas import DataFrame
from typing import List
from src import config

warnings.filterwarnings("ignore")


def remove_outliers(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Removes rows with outliers in the specified columns using the IQR method.

    Args:
        df (DataFrame): The input DataFrame.
        columns (List[str]): The list of columns to check for outliers.

    Returns:
        DataFrame: The DataFrame with outliers removed.
    """
    for col in columns:
        if df[col].dtype in ["int64", "float64"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
    return df


def replacement(df: DataFrame) -> DataFrame:
    """
    Cleans the 'Airline' and 'Additional_Info' columns.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with cleaned columns.
    """
    df["Airline"] = df["Airline"].str.replace(" Premium economy", "")
    df["Airline"] = df["Airline"].str.replace("Business", "")
    df["Additional_Info"] = df["Additional_Info"].str.replace("No Info", "No info")
    return df


def stop(df: DataFrame) -> DataFrame:
    """
    Maps the 'Total_Stops' column to numerical values.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with 'Total_Stops' mapped to numbers.
    """
    stop_mapping = {
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4,
    }
    df["Total_Stops"] = df["Total_Stops"].map(stop_mapping).fillna(0).astype(int)
    return df


def arrival_time(df: DataFrame) -> DataFrame:
    """
    Extracts the time from the 'Arrival_Time' column.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with the 'Arrival_Time' column modified.
    """
    df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: x.split(" ")[0])
    return df


def duration(df: DataFrame) -> DataFrame:
    """
    Converts the 'Duration' column to numeric values (in minutes).

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with 'Duration' converted to minutes.
    """

    def convert_to_minutes(x: str) -> int:
        if "h" in x and "m" in x:
            hours, minutes = x.split("h ")
            return int(hours) * 60 + int(minutes.replace("m", ""))
        elif "h" in x:
            return int(x.replace("h", "")) * 60
        elif "m" in x:
            return int(x.replace("m", ""))
        else:
            return np.nan

    df["Duration"] = df["Duration"].apply(convert_to_minutes)
    df["Duration"] = df["Duration"].astype(int)
    return df


def to_datetime(df: DataFrame) -> DataFrame:
    """
    Converts date and time columns to datetime objects.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The DataFrame with columns converted to datetime objects.
    """
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
    df["Dep_Time"] = pd.to_datetime(df["Dep_Time"]).dt.time
    df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"]).dt.time
    return df


def cleaning(df: DataFrame) -> DataFrame:
    """
    Performs data cleaning and preprocessing on the input DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: The cleaned and preprocessed DataFrame.
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(["Route"], axis=1, inplace=True)
    df = replacement(df)
    df = arrival_time(df)
    df = duration(df)
    df = stop(df)
    df = to_datetime(df)
    df = remove_outliers(df, ["Duration"])
    return df


def main() -> None:
    """
    Runs data processing scripts to turn raw data from (../raw)
    into cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    input_filepath = config.RAW_DATA_PATH
    logger.info(f"reading data from {input_filepath}")
    try:
        df = pd.read_csv(input_filepath)
        df = cleaning(df)

        output_filepath = config.PROCESSED_DATA_PATH
        os.makedirs(output_filepath.parent, exist_ok=True)
        df.to_csv(output_filepath, index=False)
        logger.info(f"data saved to {output_filepath}")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
