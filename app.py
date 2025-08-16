import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import datetime
import re
import os
from src import config

app = FastAPI()

# Only mount static files if not in testing mode
if os.environ.get("TESTING", "").lower() != "true":
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory="templates")

model = joblib.load(config.BEST_MODEL_PATH)
airline_encoder = joblib.load(config.AIRLINE_ENCODER_PATH)
source_encoder = joblib.load(config.SOURCE_ENCODER_PATH)
destination_encoder = joblib.load(config.DESTINATION_ENCODER_PATH)
preprocessor = joblib.load(config.PREPROCESSOR_PATH)

airlines = [
    "IndiGo",
    "Air India",
    "Jet Airways",
    "SpiceJet",
    "Multiple carriers",
    "GoAir",
    "Vistara",
    "Air Asia",
    "Vistara Premium economy",
    "Jet Airways Business",
    "Multiple carriers Premium economy",
    "Trujet",
]
sources = ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"]
destinations = ["New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"]
additional_info = [
    "No info",
    "In-flight meal not included",
    "No check-in baggage included",
    "1 Short layover",
    "1 Long layover",
    "Change airports",
    "Business class",
    "Red-eye flight",
    "2 Long layover",
]


def parse_duration(duration_str: str) -> int:
    """
    Parses a duration string (e.g., '2h 30m') into total minutes.
    """
    hours = 0
    minutes = 0
    if "h" in duration_str:
        hours_match = re.search(r"(\d+)h", duration_str)
        if hours_match:
            hours = int(hours_match.group(1))
    if "m" in duration_str:
        minutes_match = re.search(r"(\d+)m", duration_str)
        if minutes_match:
            minutes = int(minutes_match.group(1))
    return hours * 60 + minutes


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "airlines": airlines,
            "sources": sources,
            "destinations": destinations,
            "additional_info": additional_info,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    airline: str = Form(...),
    source: str = Form(...),
    destination: str = Form(...),
    total_stops: int = Form(...),
    date_of_journey: str = Form(...),
    dep_time: str = Form(...),
    arrival_time: str = Form(...),
    duration: str = Form(...),
    additional_info: str = Form(...),
):
    """
    Predict the price of a flight based on its features.
    """
    # Parse date and time
    date_of_journey = datetime.datetime.strptime(date_of_journey, "%Y-%m-%d")
    dep_time = datetime.datetime.strptime(dep_time, "%H:%M")
    arrival_time = datetime.datetime.strptime(arrival_time, "%H:%M")
    duration_minutes = parse_duration(duration)

    flight_data = {
        "airline": airline,
        "source": source,
        "destination": destination,
        "total_stops": total_stops,
        "day": date_of_journey.day,
        "month": date_of_journey.month,
        "year": date_of_journey.year,
        "dep_hour": dep_time.hour,
        "dep_min": dep_time.minute,
        "arrival_hour": arrival_time.hour,
        "arrival_min": arrival_time.minute,
        "duration": duration_minutes,
        "additional_info": additional_info,
    }
    data = pd.DataFrame([flight_data])

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
    data.rename(columns={"duration": "Duration"}, inplace=True)

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

    data["Airline"] = airline_encoder.transform(data[["Airline"]])
    data["Source"] = source_encoder.transform(data[["Source"]])
    data["Destination"] = destination_encoder.transform(data[["Destination"]])
    preprocessed = preprocessor.transform(data)
    preprocessed_df = pd.DataFrame(
        preprocessed, columns=preprocessor.get_feature_names_out()
    )

    final_df = pd.concat(
        [data.reset_index(drop=True), preprocessed_df.reset_index(drop=True)], axis=1
    )
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
        ],
        axis=1,
        inplace=True,
    )

    final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(final_df)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": f"Predicted Price: {prediction[0]:.2f}",
            "airlines": airlines,
            "sources": sources,
            "destinations": destinations,
            "additional_info": additional_info,
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
