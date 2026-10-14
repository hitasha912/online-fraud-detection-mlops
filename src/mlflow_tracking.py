from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def log_registered_model(
    model,
    model_name,
    registered_model_name,
    tracking_uri,
    experiment_name,
    input_example,
    metrics,
    parameters
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    predictions = model.predict(input_example)
    signature = infer_signature(input_example, predictions)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            key: str(value)
            for key, value in parameters.items()
        })
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            code_paths=[str(Path(__file__).resolve().parent)],
            skops_trusted_types=[
                "numpy.dtype",
                "src.preprocessing.FraudFeatureEngineer"
            ],
            registered_model_name=registered_model_name
        )

    return model_info
