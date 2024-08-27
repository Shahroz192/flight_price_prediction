import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def remove_outliers(df, columns):
    """
    Removes rows with outliers in the specified columns.
    """
    for col in columns:
        if df[col].dtype in ["int64", "float64"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
    return df


def replacement(df):
    """
    remove Premium economy and business class from the name of the airline
    """
    df["Airline"] = df["Airline"].str.replace(" Premium economy", "")
    df["Airline"] = df["Airline"].str.replace("Business", "")
    df["Additional_Info"] = df["Additional_Info"].str.replace("No Info", "No info")
    return df


def stop(df):
    """
    Maps the 'Total_Stops' column to numerical values.
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


def arrival_time(df):
    """
    split the arrival time in time add time"""

    df["Arrival_Time"] = df["Arrival_Time"].apply(lambda x: x.split(" ")[0])
    return df


def duration(df):
    """
    Converts the 'Duration' column to numeric values (in minutes).
    """

    def convert_to_minutes(x):
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


def to_datetime(df):
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
    df["Dep_Time"] = pd.to_datetime(df["Dep_Time"]).dt.time
    df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"]).dt.time
    return df


def cleaning(df):
    """
    Performs data cleaning and preprocessing on the input DataFrame.
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


def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    input_filepath = Path("data/raw/flight_price.csv")

    try:
        df = pd.read_csv(input_filepath)
        df = cleaning(df)

        output_filepath = Path("data/processed/processed_flight.csv")

        df.to_csv(output_filepath, index=False)

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
