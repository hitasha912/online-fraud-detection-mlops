from fastapi.testclient import TestClient
from api.main import app
import api.main
 
client = TestClient(app)
 
 
def test_health_check():
    response = client.get("/")
 
    assert response.status_code == 200
 
    assert response.json() == {
        "message": "Online Fraud Detection API is running"
    }


def test_load_model_from_mlflow_uri(monkeypatch):
    import mlflow.sklearn

    expected_model = object()
    loaded_uris = []

    def load_mlflow_model(model_uri):
        loaded_uris.append(model_uri)
        return expected_model

    monkeypatch.setattr(api.main, "model", None)
    monkeypatch.setattr(
        api.main,
        "MLFLOW_MODEL_URI",
        "models:/fraud_detection_model/1"
    )
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        load_mlflow_model
    )

    assert api.main.load_model() is expected_model
    assert loaded_uris == ["models:/fraud_detection_model/1"]