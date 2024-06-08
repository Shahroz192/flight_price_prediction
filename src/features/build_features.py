import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


def feature_encoding(df):
    """
    One-hot encodes the 'Airline', 'Source', 'Destination', and 'Additional_Info' columns.
    """
    df = pd.get_dummies(
        df, columns=["Airline", "Source", "Destination", "Additional_Info"]
    )
    return df


def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    # Hardcoded file paths
    input_filepath = Path(
        "d:/flight_price_prediction/data/processed/flight_price_processed.csv"
    )

    try:
        df = pd.read_csv(input_filepath)
        cleaned_df = feature_encoding(df)
        cleaned_df.to_csv(
            "d:/flight_price_prediction/data/processed/flight_price_features.csv",
            index=False,
        )

        logger.info("Processed data saved.")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

    return


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    main()
