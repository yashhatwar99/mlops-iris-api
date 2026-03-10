from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():

    sample = {
        "features":[5.1,3.5,1.4,0.2]
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], int)