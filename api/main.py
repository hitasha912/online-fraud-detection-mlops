from pathlib import Path
import logging
import os
import sys
import traceback

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocessing import FraudFeatureEngineer


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.pkl"
)

logger.info("Project directory: %s", BASE_DIR)
logger.info("Model path: %s", MODEL_PATH)
logger.info("Model exists: %s", MODEL_PATH.exists())


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

model = None

MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model():
    """
    Load and cache the trained fraud-detection model.

    If MLFLOW_MODEL_URI is configured, the model is loaded
    from MLflow. Otherwise, the local Joblib model is loaded.
    """

    global model

    # Return the already loaded model
    if model is not None:
        return model

    try:
        # --------------------------------------------------
        # Load from MLflow
        # --------------------------------------------------

        if MLFLOW_MODEL_URI:
            import mlflow.sklearn

            logger.info(
                "Loading MLflow model from: %s",
                MLFLOW_MODEL_URI
            )

            model = mlflow.sklearn.load_model(
                MLFLOW_MODEL_URI
            )

        # --------------------------------------------------
        # Load from local pickle file
        # --------------------------------------------------

        else:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found: {MODEL_PATH}"
                )

            logger.info(
                "Loading local model: %s",
                MODEL_PATH
            )

            # --------------------------------------------------
            # Compatibility fix for old pickle
            # --------------------------------------------------
            #
            # The old best_model.pkl was saved when
            # FraudFeatureEngineer belonged to __main__.
            #
            # This registers the current class under __main__
            # before Joblib tries to deserialize the model.
            #
            # This must execute BEFORE joblib.load().
            # --------------------------------------------------

            main_module = sys.modules.get("__main__")

            if main_module is None:
                raise RuntimeError(
                    "The __main__ module is unavailable."
                )

            setattr(
                main_module,
                "FraudFeatureEngineer",
                FraudFeatureEngineer
            )

            logger.info(
                "Registered FraudFeatureEngineer under __main__"
            )

            model = joblib.load(MODEL_PATH)

        logger.info(
            "Model loaded successfully: %s",
            type(model).__name__
        )

        return model

    except Exception:
        logger.exception("Model loading failed")
        raise


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Online Fraud Detection API",
    description=(
        "API for identifying potentially fraudulent "
        "online transactions."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Request input schema
# --------------------------------------------------

class Transaction(BaseModel):
    step: float
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    nameDest: str


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Online Fraud Detection API is running"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": model is not None,
        "mlflow_model_uri": MLFLOW_MODEL_URI
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Convert Pydantic request object to dictionary
        input_data = transaction.model_dump()

        logger.info(
            "Input received: %s",
            input_data
        )

        # Create a one-row DataFrame
        data = pd.DataFrame(
            [input_data]
        )

        logger.info(
            "Input columns: %s",
            data.columns.tolist()
        )

        logger.info(
            "Input data types:\n%s",
            data.dtypes
        )

        # Load the trained pipeline
        model_instance = load_model()

        # Generate prediction
        prediction = model_instance.predict(data)[0]

        # Generate fraud probability, if supported
        if hasattr(model_instance, "predict_proba"):
            probabilities = model_instance.predict_proba(data)
            probability = probabilities[0][1]
        else:
            probability = None

        prediction_value = int(prediction)

        result = {
            "prediction": prediction_value,
            "fraud": prediction_value == 1,
            "fraud_probability": (
                float(probability)
                if probability is not None
                else None
            )
        }

        logger.info(
            "Prediction result: %s",
            result
        )

        return result

    except Exception as error:
        logger.error(
            "Prediction failed:\n%s",
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{str(error)}"
            )
        ) from error