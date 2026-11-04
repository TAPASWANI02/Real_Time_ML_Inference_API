
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    sample_passenger = {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0,
        "embarked": "S",
        "who": "woman",
        "adult_male": False,
        "deck": "C",
        "embark_town": "Cherbourg",
        "alone": True
    }

    response = client.post("/predict", json=sample_passenger)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "Survived",
        "Did Not Survive"
    ]
    assert 0 <= data["probability"] <= 1


def test_invalid_input():
    sample_passenger = {
        "pclass": 4,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 100.0,
        "embarked": "S",
        "who": "woman",
        "adult_male": False,
        "deck": "C",
        "embark_town": "Cherbourg",
        "alone": True
    }

    response = client.post("/predict", json=sample_passenger)

    assert response.status_code == 422
