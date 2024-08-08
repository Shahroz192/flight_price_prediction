from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the encoders and preprocessor
airline_encoder = joblib.load("../models/encoder/airline_encoder.joblib")
source_encoder = joblib.load("../models/encoder/source_encoder.joblib")
destination_encoder = joblib.load("../models/encoder/destination_encoder.joblib")
preprocessor = joblib.load("../models/encoder/preprocessor.joblib")
model = joblib.load("../models/model.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
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

    return render_template(
        "index.html",
        airlines=airlines,
        sources=sources,
        destinations=destinations,
        additional_info=additional_info,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Apply the same transformations as in training
    df["Airline"] = airline_encoder.transform(df["Airline"])
    df["Source"] = source_encoder.transform(df["Source"])
    df["Destination"] = destination_encoder.transform(df["Destination"])
    preprocessed = preprocessor.transform(df)
    preprocessor_df = pd.DataFrame(preprocessed, columns=preprocessor.get_feature_names_out())
    df.columns = df.columns.astype(str)
    df = pd.concat([df, preprocessor_df], axis=1)
    df.drop(
        ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Additional_Info"],
        axis=1,
        inplace=True,
    )
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    df['Total_Stops'] = pd.to_numeric(df['Total_Stops'], errors='coerce')

    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(debug=True)

