"""
Unit tests for the prediction layer and API endpoints.

Run: pytest tests/
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]
SAMPLE_TRANSACTION = {name: 0.0 for name in FEATURE_NAMES}


class TestPredictModule:
    def test_predict_returns_required_keys(self):
        """predict() must always return fraud_probability, is_fraud, threshold_used."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import predict as predict_module

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        predict_module._model = mock_model
        predict_module._config = {"prediction": {"threshold": 0.3}}

        result = predict_module.predict(SAMPLE_TRANSACTION)

        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "threshold_used" in result

    def test_threshold_applied_correctly(self):
        """A probability above threshold → is_fraud=True; below → False."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import predict as predict_module

        mock_model = MagicMock()

        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
        predict_module._model = mock_model
        predict_module._config = {"prediction": {"threshold": 0.3}}
        assert predict_module.predict(SAMPLE_TRANSACTION)["is_fraud"] is True

        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        assert predict_module.predict(SAMPLE_TRANSACTION)["is_fraud"] is False

    def test_raises_if_not_initialised(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import predict as predict_module

        predict_module._model = None
        predict_module._config = None

        with pytest.raises(RuntimeError, match="load_artifacts"):
            predict_module.predict(SAMPLE_TRANSACTION)


class TestAPI:
    @pytest.fixture
    def client(self):
        from api.main import app
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.65, 0.35]])
        with patch("api.main.model", mock_model):
            with TestClient(app) as c:
                yield c

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_predict_endpoint_shape(self, client):
        resp = client.post("/predict", json=SAMPLE_TRANSACTION)
        assert resp.status_code == 200
        data = resp.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold_used" in data

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V1"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422
