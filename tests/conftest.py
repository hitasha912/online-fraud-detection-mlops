import sys
from pathlib import Path
 
import joblib
import pytest
from sklearn.dummy import DummyClassifier
 
 
@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """
    Provide a lightweight dummy model for CI tests.
    The real trained model is not committed to GitHub.
    """
 
    model = DummyClassifier(
        strategy="prior"
    )
 
    model.fit(
        [[0]],
        [0]
    )
 
    import api.main
 
    monkeypatch.setattr(
        api.main,
        "model",
        model
    )