import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load the encoders and preprocessor
airline_encoder = joblib.load("../models/encoder/airline_encoder.joblib")
source_encoder = joblib.load("../models/encoder/source_encoder.joblib")
destination_encoder = joblib.load("../models/encoder/destination_encoder.joblib")
preprocessor = joblib.load("../models/encoder/preprocessor.joblib")
model = joblib.load("../models/best_model")

# Streamlit app
st.title("Flight Price Prediction")

airlines = [
    "IndiGo",
    "Air India",
    "Jet Airways",
    "SpiceJet",
    "Multiple carriers",
    "GoAir",
    "Vistara",
    "Air Asia",
    "Jet Airways ",
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

# Create input fields
airline = st.selectbox("Airline", airlines)
source = st.selectbox("Source", sources)
destination = st.selectbox("Destination", destinations)
date_of_journey = st.date_input("Date of Journey")
dep_time = st.time_input("Departure Time")
arrival_time = st.time_input("Arrival Time")
duration = st.text_input("Duration (in minutes)")
total_stops = st.text_input("Total Stops")
additional_info = st.selectbox("Additional Info", additional_info)

# Convert date and time to strings or datetimes
dep_time_str = datetime.combine(date_of_journey, dep_time).strftime("%Y-%m-%d %H:%M:%S")
arrival_time_str = datetime.combine(date_of_journey, arrival_time).strftime("%Y-%m-%d %H:%M:%S")

# Prediction button
if st.button("Predict"):
    # Create a dataframe from the inputs
    data = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Date_of_Journey": date_of_journey.strftime("%Y-%m-%d"),
        "Dep_Time": dep_time_str,
        "Arrival_Time": arrival_time_str,
        "Duration": duration,
        "Total_Stops": total_stops,
        "Additional_Info": additional_info,
    }
    df = pd.DataFrame([data])

    # Apply the same transformations as in training
    df["Airline"] = airline_encoder.transform(df["Airline"])
    df["Source"] = source_encoder.transform(df["Source"])
    df["Destination"] = destination_encoder.transform(df["Destination"])
    preprocessed = preprocessor.transform(df)
    preprocessor_df = pd.DataFrame(
        preprocessed, columns=preprocessor.get_feature_names_out()
    )
    df.columns = df.columns.astype(str)
    df = pd.concat([df, preprocessor_df], axis=1)
    df.drop(
        ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Additional_Info"],
        axis=1,
        inplace=True,
    )
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df["Total_Stops"] = pd.to_numeric(df["Total_Stops"], errors="coerce")

    # Make predictions
    predictions = model.predict(df)

    # Display the result
    st.success(f"Predicted Price: {predictions[0]}")
