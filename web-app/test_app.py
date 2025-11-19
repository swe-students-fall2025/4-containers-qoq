"""Unit tests for the Flask web application."""

# pylint: disable=redefined-outer-name
# Fixture names used as function parameters are standard pytest pattern

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from pymongo.errors import PyMongoError

# Set environment variables before importing app
os.environ["MONGO_URI"] = "mongodb://test:27017"
os.environ["AUDIO_MODEL_PATH"] = os.path.join(
    os.path.dirname(__file__),
    "models",
    "lite-model_yamnet_classification_tflite_1.tflite",
)

# pylint: disable=wrong-import-position
# Import must come after environment variables are set
from app import (
    app,
    _parse_iso_dt,
    _serialize_prediction,
    _store_prediction,
    classify_wav,
)


@pytest.fixture
def test_client():
    """Create a test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_mongo_collection():
    """Mock MongoDB collection."""
    with patch("app.get_collection") as mock_get_collection:
        mock_col = MagicMock()
        mock_get_collection.return_value = mock_col
        yield mock_col


@pytest.fixture
def sample_prediction_doc():
    """Sample prediction document."""
    return {
        "_id": "507f1f77bcf86cd799439011",
        "instrument": "piano",
        "confidence": 0.95,
        "source": "upload",
        "captured_at": datetime.now(timezone.utc),
        "created_at": datetime.now(timezone.utc),
    }


def test_index(test_client):
    """Test the index route."""
    response = test_client.get("/")
    assert response.status_code == 200


def test_dashboard(test_client):
    """Test the dashboard route."""
    response = test_client.get("/dashboard")
    assert response.status_code == 200


def test_login_get_renders_page(test_client):
    """GET /login should return the login page."""
    resp = test_client.get("/login")
    assert resp.status_code == 200
    assert b"Log In" in resp.data


def test_signup_get_renders_page(test_client):
    """GET /signup should return the signup page."""
    resp = test_client.get("/signup")
    assert resp.status_code == 200
    assert b"Create account" in resp.data


def test_login_post_success_renders_index(test_client):
    """POST /login with non-empty credentials should render the index page."""
    resp = test_client.post(
        "/login",
        data={"email": "user@example.com", "password": "secret"},
        follow_redirects=False,
    )

    # Because you're using render_template, this should be 200, not 302
    assert resp.status_code == 200

    # Check for something that clearly belongs to index.html
    # e.g. the main title "Detecting"
    assert b"Detecting" in resp.data


def test_health_ok(mock_mongo_collection):
    """Test health check when MongoDB is connected."""
    mock_mongo_collection.find_one.return_value = {}
    with app.test_client() as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"


def test_health_error(mock_mongo_collection):
    """Test health check when MongoDB connection fails."""
    mock_mongo_collection.find_one.side_effect = PyMongoError("Connection failed")
    with app.test_client() as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert "error" in data["status"]


def test_api_create_prediction_success(mock_mongo_collection, test_client):
    """Test creating a prediction via API."""
    mock_mongo_collection.insert_one.return_value = Mock(inserted_id="test_id")
    payload = {
        "instrument": "guitar",
        "confidence": 0.85,
        "source": "test",
        "captured_at": "2024-01-15T10:30:00Z",
    }
    response = test_client.post("/api/predictions", json=payload)
    assert response.status_code == 201
    data = response.get_json()
    assert data["instrument"] == "guitar"
    assert data["confidence"] == 0.85
    assert data["source"] == "test"


def test_api_create_prediction_missing_instrument(test_client):
    """Test creating prediction without instrument."""
    response = test_client.post("/api/predictions", json={"confidence": 0.5})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_api_create_prediction_invalid_confidence(test_client):
    """Test creating prediction with invalid confidence."""
    response = test_client.post(
        "/api/predictions", json={"instrument": "piano", "confidence": "invalid"}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_api_list_predictions(
    mock_mongo_collection, test_client, sample_prediction_doc
):
    """Test listing predictions."""
    mock_mongo_collection.find.return_value.sort.return_value.limit.return_value = [
        sample_prediction_doc
    ]
    response = test_client.get("/api/predictions")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 1


def test_api_list_predictions_with_limit(
    mock_mongo_collection, test_client, sample_prediction_doc
):
    """Test listing predictions with custom limit."""
    mock_mongo_collection.find.return_value.sort.return_value.limit.return_value = [
        sample_prediction_doc
    ]
    response = test_client.get("/api/predictions?limit=10")
    assert response.status_code == 200
    mock_mongo_collection.find.return_value.sort.return_value.limit.assert_called_with(
        10
    )


def test_api_dashboard_data(mock_mongo_collection, test_client, sample_prediction_doc):
    """Test dashboard data aggregation."""
    mock_mongo_collection.count_documents.return_value = 5
    mock_mongo_collection.find.return_value.sort.return_value.limit.return_value = [
        sample_prediction_doc
    ]
    response = test_client.get("/api/dashboard-data")
    assert response.status_code == 200
    data = response.get_json()
    assert "total_runs" in data
    assert "diff_instruments" in data
    assert "recent" in data
    assert data["total_runs"] == 5


def test_serialize_prediction(sample_prediction_doc):
    """Test prediction serialization."""
    result = _serialize_prediction(sample_prediction_doc)
    assert result["instrument"] == "piano"
    assert result["confidence"] == 0.95
    assert result["id"] == "507f1f77bcf86cd799439011"
    assert "captured_at" in result
    assert "created_at" in result


def test_parse_iso_dt_valid():
    """Test parsing valid ISO datetime."""
    dt_str = "2024-01-15T10:30:00Z"
    result = _parse_iso_dt(dt_str)
    assert result is not None
    assert isinstance(result, datetime)


def test_parse_iso_dt_invalid():
    """Test parsing invalid datetime."""
    assert _parse_iso_dt("invalid") is None
    assert _parse_iso_dt(None) is None
    assert _parse_iso_dt("") is None


@patch("app.get_audio_classifier")
@patch("app.librosa.load")
def test_classify_wav_success(mock_librosa, mock_classifier):
    """Test successful audio classification."""
    # Mock librosa
    mock_wav_data = np.array([0.5, -0.3, 0.8], dtype=np.float32)
    mock_sample_rate = 16000
    mock_librosa.return_value = (mock_wav_data, mock_sample_rate)

    # Mock classifier
    mock_classifier_instance = MagicMock()
    mock_classifier.return_value = mock_classifier_instance

    mock_category = MagicMock()
    mock_category.category_name = "Guitar"
    mock_category.score = 0.92

    mock_classification = MagicMock()
    mock_classification.categories = [mock_category]

    mock_head = MagicMock()
    mock_head.categories = [mock_category]

    mock_result = MagicMock()
    mock_result.classifications = [mock_head]

    mock_classifier_instance.classify.return_value = [mock_result]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b"fake audio data")

    try:
        label, score = classify_wav(tmp_path)
        assert label == "Guitar"
        assert score == 0.92
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@patch("app.librosa.load")
def test_classify_wav_file_error(mock_librosa):
    """Test classification with file read error."""
    mock_librosa.side_effect = OSError("File not found")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with pytest.raises(Exception):  # Should raise AudioClassificationError
            classify_wav(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@patch("app.classify_wav")
@patch("app._store_prediction")
def test_api_classify_upload_success(mock_store, mock_classify, test_client):
    """Test successful audio upload and classification."""
    mock_classify.return_value = ("piano", 0.88)
    mock_store.return_value = {
        "id": "test_id",
        "instrument": "piano",
        "confidence": 0.88,
        "source": "upload",
    }

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(b"fake audio data")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = test_client.post(
                "/api/classify-upload", data={"audio": (f, "test.wav")}
            )
        assert response.status_code == 200
        data = response.get_json()
        assert data["instrument"] == "piano"
        assert data["confidence"] == 0.88
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_api_classify_upload_no_file(test_client):
    """Test upload without file."""
    response = test_client.post("/api/classify-upload")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_api_classify_upload_empty_filename(test_client):
    """Test upload with empty filename."""
    response = test_client.post("/api/classify-upload", data={"audio": (b"", "")})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_store_prediction(mock_mongo_collection):
    """Test storing a prediction."""
    mock_mongo_collection.insert_one.return_value = Mock(inserted_id="test_id")
    result = _store_prediction("violin", 0.75, "test")
    assert result["instrument"] == "violin"
    assert result["confidence"] == 0.75
    assert result["source"] == "test"
    assert result["id"] == "test_id"
    mock_mongo_collection.insert_one.assert_called_once()


def test_store_prediction_with_captured_at(mock_mongo_collection):
    """Test storing prediction with custom captured_at."""
    mock_mongo_collection.insert_one.return_value = Mock(inserted_id="test_id")
    custom_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    result = _store_prediction("guitar", 0.90, "test", captured_at=custom_time)
    assert result["instrument"] == "guitar"
    mock_mongo_collection.insert_one.assert_called_once()
