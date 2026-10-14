# Local MLflow deployment

Training logs the selected, tuned scikit-learn pipeline to MLflow, records its
metrics and parameters, and registers it as `fraud_detection_model` by default.
The pipeline includes preprocessing and feature engineering, so serving accepts
the same raw transaction fields as the existing API.

## Start local MLflow

From the repository root, start the tracking server (SQLite registry and local
artifacts):

```powershell
mlflow server --backend-store-uri sqlite:///./mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

In a second PowerShell terminal, point training at the server and train:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "online-fraud-detection"
$env:MLFLOW_REGISTERED_MODEL_NAME = "fraud_detection_model"
.\.venv\Scripts\python.exe src\train.py --data data\onlinefraud.csv
```

The dataset is not checked into this repository. Supply its local path with
`--data` or use the default `data/onlinefraud.csv`.

Training can also log directly to a local SQLite store without the server. If
`MLFLOW_TRACKING_URI` is unset, it uses `sqlite:///.../mlflow.db` in the
repository root. Start the server later with the same backend store to inspect
those runs.

## Serve the registered model with MLflow

Find the registered model version in the MLflow UI at
`http://127.0.0.1:5000`, then start MLflow's local scoring server using that
version:

```powershell
mlflow models serve --model-uri "models:/fraud_detection_model/1" --host 127.0.0.1 --port 5001 --env-manager local
```

Replace `1` with the version shown for your run. The scoring endpoint is
`http://127.0.0.1:5001/invocations`; MLflow's model server accepts the
registered model's input signature.

## Use the registered model from FastAPI

The existing FastAPI endpoint continues to load `models/fraud_detection_model.pkl`
by default. To have it load the registered MLflow model instead, set its model
URI and tracking server before starting Uvicorn:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_MODEL_URI = "models:/fraud_detection_model/1"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use the same registered version as above. The prediction route remains
`POST http://127.0.0.1:8000/predict`.

## Configuration

- `MLFLOW_TRACKING_URI`: tracking server URL or MLflow backend URI.
- `MLFLOW_EXPERIMENT_NAME`: experiment name (default:
  `online-fraud-detection`).
- `MLFLOW_REGISTERED_MODEL_NAME`: registry name (default:
  `fraud_detection_model`).
- `MLFLOW_MODEL_URI`: optional registered model URI for FastAPI. When unset,
  FastAPI uses the local pickle as before.
