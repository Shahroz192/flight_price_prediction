from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict_price():
    response = client.post(
        "/predict",
        json={
            "airline": "IndiGo",
            "source": "Banglore",
            "destination": "New Delhi",
            "total_stops": 0,
            "day": 1,
            "month": 1,
            "year": 2025,
            "dep_hour": 10,
            "dep_min": 0,
            "arrival_hour": 12,
            "arrival_min": 30,
            "duration_hours": 2,
            "duration_mins": 30,
            "additional_info": "No info",
        },
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
