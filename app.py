import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import datetime
from src import config

app = FastAPI()

# Load the trained model and encoders
model = joblib.load(config.BEST_MODEL_PATH)
airline_encoder = joblib.load(config.AIRLINE_ENCODER_PATH)
source_encoder = joblib.load(config.SOURCE_ENCODER_PATH)
destination_encoder = joblib.load(config.DESTINATION_ENCODER_PATH)
preprocessor = joblib.load(config.PREPROCESSOR_PATH)


class Flight(BaseModel):
    airline: str
    source: str
    destination: str
    total_stops: int
    day: int
    month: int
    year: int
    dep_hour: int
    dep_min: int
    arrival_hour: int
    arrival_min: int
    duration_hours: int
    duration_mins: int
    additional_info: str = "No info"


@app.post("/predict")
def predict_price(flight: Flight):
    """
    Predict the price of a flight based on its features.
    """
    data = pd.DataFrame([flight.model_dump()])

    # 1. Create required columns
    data["Date_of_Journey"] = pd.to_datetime(data[["year", "month", "day"]])
    data["Dep_Time"] = data.apply(
        lambda r: datetime.datetime(
            r["year"], r["month"], r["day"], r["dep_hour"], r["dep_min"]
        ),
        axis=1,
    )
    data["Arrival_Time"] = data.apply(
        lambda r: datetime.datetime(
            r["year"], r["month"], r["day"], r["arrival_hour"], r["arrival_min"]
        ),
        axis=1,
    )
    data["Duration"] = data["duration_hours"] * 60 + data["duration_mins"]

    # 2. Rename columns to match those used in training
    data.rename(
        columns={
            "airline": "Airline",
            "source": "Source",
            "destination": "Destination",
            "total_stops": "Total_Stops",
            "additional_info": "Additional_Info",
        },
        inplace=True,
    )

    # 3. Apply target encoders
    data["Airline"] = airline_encoder.transform(data[["Airline"]])
    data["Source"] = source_encoder.transform(data[["Source"]])
    data["Destination"] = destination_encoder.transform(data[["Destination"]])

    # 4. Apply preprocessor
    preprocessed = preprocessor.transform(data)
    preprocessed_df = pd.DataFrame(
        preprocessed, columns=preprocessor.get_feature_names_out()
    )

    # 5. Concatenate dataframes
    final_df = pd.concat(
        [data.reset_index(drop=True), preprocessed_df.reset_index(drop=True)], axis=1
    )

    # 6. Drop original columns that were transformed
    final_df.drop(
        ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Additional_Info"],
        axis=1,
        inplace=True,
    )

    final_df.drop(
        [
            "year",
            "month",
            "day",
            "dep_hour",
            "dep_min",
            "arrival_hour",
            "arrival_min",
            "duration_hours",
            "duration_mins",
        ],
        axis=1,
        inplace=True,
    )

    # Ensure column order is the same as during training
    final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make a prediction
    prediction = model.predict(final_df)

    return {"prediction": float(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
