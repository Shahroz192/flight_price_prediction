from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict_price():
    response = client.post(
        "/predict",
        data={
            "airline": "IndiGo",
            "source": "Banglore",
            "destination": "New Delhi",
            "total_stops": 0,
            "date_of_journey": "2025-01-01",
            "dep_time": "10:00",
            "arrival_time": "12:30",
            "duration": "2h 30m",
            "additional_info": "No info",
        },
    )
    assert response.status_code == 200
    assert "prediction" in response.text  # Check response.text for HTML response
