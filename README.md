# Flight Price Prediction

![Build Status](https://img.shields.io/github/actions/workflow/status/shahroz601/flight-price-prediction/main.yml)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/github/license/shahroz601/flight-price-prediction)

## Description

This project is a flight price prediction system that uses machine learning to predict the price of flights
based on various features such as the airline, source, destination, and other relevant factors.

The application provides a web interface where users can input flight details and get an estimated price
prediction. It's built with a machine learning pipeline that processes data, engineers features, and uses an
XGBoost model to make predictions.

## Key Features

- Machine Learning Pipeline: Complete pipeline for data processing, feature engineering, and model training
- Web Interface: User-friendly web interface for inputting flight details and getting price predictions
- REST API: FastAPI-based backend that serves predictions via HTTP endpoints
- Docker Support: Containerized application for easy deployment and scaling- Model Tracking: Integration with
     MLflow for experiment tracking and model management

## Tech Stack

- Language: Python3.11
- Framework: FastAPI- Machine Learning: scikit-learn, XGBoost, category-encoders, feature-engine- Data
     Processing: pandas, numpy
- Visualization: matplotlib, seaborn
- Model Management: MLflow, joblib- Testing: pytest, httpx- Containerization: Docker
- Template Engine: Jinja2

## Installation and Setup

Prerequisites- Python 3.11- pip (Python package installer)

- Docker (optional, for containerized deployment)

### Clone the Repository

   1 git clone <https://github.com/shahroz601/flight-price-prediction.git>
   2 cd flight-price-prediction```
   3
### Install Dependencies

   1 pip install -r requirements.txt
   2 ```
### Configure Environment
The application doesn't require any special environment variables to run. However, for development and testing, you might want to set:
   3
   1 export TESTING=true  # To disable static file mounting during tests
   2```## Usage

### Running the Application

To start the web application:
   1 python app.py

The application will be available at <http://localhost:8000>.

### Using the Web Interface

1. Open your browser and navigate to <http://localhost:8000>
2. Fill in the flight details in the form: - Select the airline - Enter the number of stops
   - Select source and destination   - Enter journey date and times   - Specify the duration (e.g., "2h 30m")
     - Select additional information
   3. Click "Predict Price" to get the estimated flight price

### Using the API

You can also send a POST request directly to the API endpoint:

`bashcurl -X POST "<http://localhost:8000/predict>" \ -H "Content-Type: application/x-www-form-urlencoded" \
       -d "airline=IndiGo&source=Banglore&destination=New Delhi&total_stops=0&date_of_journey=2025-01-01&dep_t
  ime=10:00&arrival_time=12:30&duration=2h30m&additional_info=No info"
   1
### Running with Docker

To build and run this project using Docker:

   1. Build the Docker image:

   1    docker build -t flight-price-prediction .
   2    ```2. **Run the Docker container:**
  `bash   docker run -p 8000:8000 flight-price-prediction
     `3. Access the API:
     The API will be available at <http://localhost:8000>.

### Running Tests

To run the test suite:

   1 pytest

To run a specific test file:

bashpytest tests/test_app.py

## Contributing

Contributions are welcome! Please follow these steps:1. Fork the repository
   2. Create a new branch for your feature or bug fix
   3. Make your changes and commit them with descriptive messages4. Push your changes to your fork
   5. Submit a pull request to the main repository

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
