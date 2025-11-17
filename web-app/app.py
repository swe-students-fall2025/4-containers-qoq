"""Flask backend for the instrument classification web app."""

import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from scipy.io import wavfile

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import audio as mp_audio
from mediapipe.tasks.python.components import containers as mp_containers

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ml_logs")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "predictions")


@lru_cache(maxsize=1)
def _get_mongo_client() -> MongoClient:
    """Create (and memoize) a MongoClient instance."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_collection():
    """Return the MongoDB collection used to store predictions."""
    db = _get_mongo_client()[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]


BaseOptions = mp_python.BaseOptions
AudioClassifier = mp_audio.AudioClassifier
AudioClassifierOptions = mp_audio.AudioClassifierOptions
AudioRunningMode = mp_audio.RunningMode
AudioData = mp_containers.AudioData

MODEL_PATH = os.environ.get(
    "AUDIO_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "models",
        "lite-model_yamnet_classification_tflite_1.tflite",
    ),
)


@lru_cache(maxsize=1)
def get_audio_classifier() -> AudioClassifier:
    """Create and cache the MediaPipe AudioClassifier."""
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = AudioClassifierOptions(
        base_options=base_options,
        running_mode=AudioRunningMode.AUDIO_CLIPS,
        max_results=5,
    )
    return AudioClassifier.create_from_options(options)


class AudioClassificationError(RuntimeError):
    """Raised when an uploaded clip cannot be classified."""


def classify_wav(path: str) -> Tuple[str, float]:
    """Run MediaPipe Audio Classifier on a .wav file and return (label, score)."""
    try:
        sample_rate, wav_data = wavfile.read(path)
    except (OSError, ValueError) as exc:
        raise AudioClassificationError(f"unable to read audio: {exc}") from exc

    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]

    max_val = float(np.max(np.abs(wav_data)))
    norm = wav_data.astype(np.float32)
    if max_val > 0:
        norm = norm / max_val

    audio_data = AudioData.create_from_array(norm, sample_rate)

    classifier = get_audio_classifier()
    try:
        result_list = classifier.classify(audio_data)
    except RuntimeError as exc:
        raise AudioClassificationError(f"classifier failed: {exc}") from exc

    head = result_list[0].classifications[0]
    top_cat = head.categories[0]
    return top_cat.category_name, float(top_cat.score)


def _serialize_prediction(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a MongoDB prediction document into JSON-serializable data."""

    def _dt(value):
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        return value

    return {
        "id": str(doc.get("_id")) if doc.get("_id") else None,
        "instrument": doc.get("instrument", "unknown"),
        "confidence": float(doc.get("confidence", 0.0)),
        "source": doc.get("source", "unknown"),
        "captured_at": _dt(doc.get("captured_at")),
        "created_at": _dt(doc.get("created_at")),
    }


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO 8601 datetime (or return None for invalid input)."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _store_prediction(
    instrument: str,
    confidence: float,
    source: str,
    captured_at: Optional[datetime] = None,
):
    """Persist a prediction document and return its serialized payload."""
    now = datetime.now(timezone.utc)
    if captured_at is None:
        captured_at = now

    doc: Dict[str, Any] = {
        "instrument": instrument,
        "confidence": float(confidence),
        "source": source,
        "captured_at": captured_at,
        "created_at": now,
    }

    col = get_collection()
    result = col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_prediction(doc)


@app.route("/", methods=["GET"])
def index():
    """Render the landing page."""
    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Render the dashboard shell (data fetched via AJAX)."""
    return render_template("dashboard.html")


@app.route("/api/predictions", methods=["POST"])
def api_create_prediction():
    """Manual JSON prediction insertion (used by ML client if needed)."""
    payload = request.get_json(silent=True) or {}

    instrument = (payload.get("instrument") or "").strip()
    confidence = payload.get("confidence")
    source = (payload.get("source") or "").strip() or "unknown"
    captured_at_str = payload.get("captured_at")

    if not instrument:
        return jsonify({"error": "instrument is required"}), 400

    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        return jsonify({"error": "confidence must be a number"}), 400

    captured_at = _parse_iso_dt(captured_at_str)
    pred = _store_prediction(instrument, confidence_val, source, captured_at)
    return jsonify(pred), 201


@app.route("/api/predictions", methods=["GET"])
def api_list_predictions():
    """Return recent predictions."""
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    limit = max(1, min(limit, 200))

    col = get_collection()
    cursor = col.find().sort("captured_at", -1).limit(limit)
    predictions = [_serialize_prediction(d) for d in cursor]
    return jsonify(predictions), 200


@app.route("/api/dashboard-data", methods=["GET"])
def api_dashboard_data():
    """Aggregates data for the dashboard cards + recent log."""
    col = get_collection()

    total_runs = col.count_documents({})

    docs = list(col.find().sort("captured_at", -1).limit(50))
    predictions = [_serialize_prediction(d) for d in docs]

    instruments = set()
    counts: Dict[str, int] = {}
    conf_sums: Dict[str, float] = {}
    last_time: Optional[str] = None

    for p in predictions:
        inst = p["instrument"] or "unknown"
        instruments.add(inst)
        counts[inst] = counts.get(inst, 0) + 1
        conf_sums[inst] = conf_sums.get(inst, 0.0) + p["confidence"]

        if p["captured_at"]:
            if last_time is None or p["captured_at"] > last_time:
                last_time = p["captured_at"]

    diff_instruments = len(instruments)

    top_instrument = None
    top_conf = None
    if counts:
        top_instrument = max(counts.items(), key=lambda kv: kv[1])[0]
        top_conf = conf_sums[top_instrument] / counts[top_instrument]

    recent = predictions[:10]

    return (
        jsonify(
            {
                "total_runs": total_runs,
                "diff_instruments": diff_instruments,
                "top_instrument": top_instrument,
                "top_instrument_confidence": top_conf,
                "last_analysis_time": last_time,
                "recent": recent,
            }
        ),
        200,
    )


@app.route("/api/classify-upload", methods=["POST"])
def api_classify_upload():
    """
    Accept an uploaded audio file, classify it, store the result, and respond.
    """
    if "audio" not in request.files:
        return jsonify({"error": "missing 'audio' file field"}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        label, score = classify_wav(tmp_path)
    except AudioClassificationError as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    pred = _store_prediction(label or "unknown", score, source="upload")
    return jsonify(pred), 200


@app.route("/health", methods=["GET"])
def health():
    """Simple MongoDB connectivity health-check."""
    try:
        col = get_collection()
        col.find_one()
        status = "ok"
    except PyMongoError as exc:
        status = f"error: {exc}"
    return jsonify({"status": status}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
