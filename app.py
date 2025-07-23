import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import src.config as config
app = FastAPI()

# Load the trained model and encoders
model = joblib.load(config.BEST_MODEL_PATH)
airline_encoder = joblib.load(config.AIRLINE_ENCODER_PATH)
destination_encoder = joblib.load(config.DESTINATION_ENCODER_PATH)
source_encoder = joblib.load(config.SOURCE_ENCODER_PATH)
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

@app.post("/predict")
def predict_price(flight: Flight):
    """
    Predict the price of a flight based on its features.
    """
    # Create a dataframe from the input
    data = pd.DataFrame([flight.model_dump()])

    # Encode the categorical features
    data['airline'] = airline_encoder.transform(data['airline'])
    data['source'] = source_encoder.transform(data['source'])
    data['destination'] = destination_encoder.transform(data['destination'])

    # Preprocess the data
    data = preprocessor.transform(data)

    # Make a prediction
    prediction = model.predict(data)

    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
