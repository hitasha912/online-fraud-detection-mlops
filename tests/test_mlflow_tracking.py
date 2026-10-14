import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.mlflow_tracking import log_registered_model
from src.preprocessing import create_feature_pipeline


def test_log_registered_model_can_be_loaded(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    experiment_name = f"fraud-test-{tmp_path.name}"
    registered_model_name = f"fraud-model-{tmp_path.name}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts").as_uri()
    )

    input_example = pd.DataFrame({
        "step": [1.0, 2.0, 3.0, 4.0],
        "type": ["PAYMENT", "TRANSFER", "PAYMENT", "CASH_OUT"],
        "amount": [10.0, 20.0, 30.0, 40.0],
        "oldbalanceOrg": [100.0, 200.0, 300.0, 400.0],
        "newbalanceOrig": [90.0, 180.0, 270.0, 360.0],
        "oldbalanceDest": [0.0, 5.0, 10.0, 15.0],
        "newbalanceDest": [10.0, 25.0, 40.0, 55.0],
        "nameDest": ["C1", "M2", "C3", "C4"]
    })
    model = create_feature_pipeline(
        LogisticRegression(max_iter=1000)
    )
    model.fit(input_example, [0, 1, 0, 1])

    model_info = log_registered_model(
        model=model,
        model_name="Logistic Regression",
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        input_example=input_example,
        metrics={"F1": 1.0},
        parameters={"model__C": 1.0}
    )

    registered_uri = (
        f"models:/{registered_model_name}/"
        f"{model_info.registered_model_version}"
    )
    loaded_model = mlflow.sklearn.load_model(registered_uri)

    assert model_info.run_id
    assert model_info.registered_model_version
    assert loaded_model.predict(input_example).tolist() == (
        model.predict(input_example).tolist()
    )
