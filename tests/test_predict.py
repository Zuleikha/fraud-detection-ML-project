"""
Unit tests for the prediction layer and API endpoints.

Run: pytest tests/
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


SAMPLE_TRANSACTION = {
    "amount": 249.99,
    "hour_of_day": 2,
    "day_of_week": 5,
    "transactions_last_1h": 3,
    "transactions_last_24h": 7,
    "amount_mean_last_24h": 55.0,
    "amount_std_last_24h": 30.0,
    "merchant_category": "electronics",
    "card_type": "credit",
}


class TestPredictModule:
    def test_predict_returns_required_keys(self):
        """predict() must always return fraud_probability, is_fraud, threshold_used."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        import predict as predict_module

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.7, 0.3]]

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

        # Probability = 0.4, threshold = 0.3 → fraud
        mock_model.predict_proba.return_value = [[0.6, 0.4]]
        predict_module._model = mock_model
        predict_module._config = {"prediction": {"threshold": 0.3}}
        assert predict_module.predict(SAMPLE_TRANSACTION)["is_fraud"] is True

        # Probability = 0.2, threshold = 0.3 → not fraud
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
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
        with patch("predict.load_artifacts"), patch("predict._model") as mock_model, \
             patch("predict._config", {"prediction": {"threshold": 0.3}}):
            mock_model.predict_proba.return_value = [[0.65, 0.35]]
            from app.main import app
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

    def test_invalid_amount_rejected(self, client):
        bad = {**SAMPLE_TRANSACTION, "amount": -10}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422
