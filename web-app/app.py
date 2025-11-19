"""Flask backend for the instrument classification web app."""

import os
import csv
import io
import urllib.request
import urllib.error
import tempfile
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import librosa
import tensorflow_hub as hub
from pydub import AudioSegment
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from pymongo.errors import PyMongoError

app = Flask(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ml_logs")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "predictions")

CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)

ALLOWED_INSTRUMENTS = {
    "Guitar",
    "Electric guitar",
    "Bass guitar",
    "Acoustic guitar",
    "Steel guitar, slide guitar",
    "Banjo",
    "Sitar",
    "Mandolin",
    "Zither",
    "Ukulele",
    "Piano",
    "Electric piano",
    "Organ",
    "Electronic organ",
    "Hammond organ",
    "Synthesizer",
    "Sampler",
    "Harpsichord",
    "Drum kit",
    "Drum machine",
    "Drum",
    "Snare drum",
    "Bass drum",
    "Timpani",
    "Tabla",
    "Cymbal",
    "Hi-hat",
    "Wood block",
    "Tambourine",
    "Rattle (instrument)",
    "Maraca",
    "Gong",
    "Tubular bells",
    "Marimba, xylophone",
    "Glockenspiel",
    "Vibraphone",
    "Steelpan",
    "French horn",
    "Trumpet",
    "Trombone",
    "Violin, fiddle",
    "Cello",
    "Double bass",
    "Flute",
    "Saxophone",
    "Clarinet",
    "Harp",
    "Bell",
    "Church bell",
    "Jingle bell",
    "Chime",
    "Wind chime",
    "Harmonica",
    "Accordion",
    "Bagpipes",
    "Didgeridoo",
    "Shofar",
    "Theremin",
    "Singing bowl",
}


def load_class_map():
    """Load class names from csv"""
    try:
        with urllib.request.urlopen(CLASS_MAP_URL, timeout=10) as response:
            content = response.read().decode("utf-8")
        f = io.StringIO(content)
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            print("Error: Downloaded class map is empty.")
            return None
        class_names = [row[2] for row in reader if row]
        if not class_names:
            print("Error: Could not parse any class names from downloaded CSV.")
            return None
        print(f"Successfully loaded {len(class_names)} classes from GitHub.")
        return class_names
    except (OSError, urllib.error.URLError) as e:
        print(f"Error downloading class map: {e}")
        return None


CLASS_NAMES = load_class_map()

try:
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet Model loaded successfully.")
except (OSError, RuntimeError) as e:
    print(f"Error loading model: {e}")
    model = None


@lru_cache(maxsize=1)
def _get_mongo_client() -> MongoClient:
    """Create (and memoize) a MongoClient instance."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_collection():
    """Return the MongoDB collection used to store predictions."""
    db = _get_mongo_client()[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]


class AudioClassificationError(RuntimeError):
    """Raised when an uploaded clip cannot be classified."""


def _convert_audio_with_pydub(path: str) -> Tuple[np.ndarray, int]:
    """Convert audio file to WAV format using pydub and return audio data."""
    audio = AudioSegment.from_file(path)
    wav_bytes = audio.export(
        format="wav", parameters=["-ar", "16000", "-ac", "1"]
    ).read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(wav_bytes)
        tmp_wav_path = tmp_wav.name
    try:
        wav_data, sample_rate = librosa.load(tmp_wav_path, sr=16000, mono=True)
        return wav_data, sample_rate
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def classify_wav(path: str) -> Tuple[str, float]:
    """Classify audio using the YAMNet model"""
    if model is None:
        raise AudioClassificationError("Model failed to load.")
    if CLASS_NAMES is None:
        raise AudioClassificationError("Class map failed to load. Check file path.")
    try:
        wav_data, _ = librosa.load(path, sr=16000, mono=True)
    except (OSError, ValueError) as exc:
        try:
            wav_data, _ = _convert_audio_with_pydub(path)
        except (OSError, ValueError) as conv_exc:
            raise AudioClassificationError(
                f"unable to read audio: {exc} (conversion also failed: {conv_exc})"
            ) from exc

    waveform = librosa.util.normalize(wav_data)
    scores, _, _ = model(waveform)
    processed_scores = np.mean(scores.numpy(), axis=0)

    results = []
    for i, score in enumerate(processed_scores):
        results.append({"class_name": CLASS_NAMES[i], "score": float(score)})

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    for res in sorted_results:
        if res["score"] < 0.005:
            break
        if res["class_name"] in ALLOWED_INSTRUMENTS:
            return res["class_name"], res["score"]
    return "unknown", 0.0


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
        file_ext = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        label, score = classify_wav(tmp_path)
    except AudioClassificationError as exc:
        return jsonify({"error": str(exc)}), 500
    except (OSError, ValueError, RuntimeError) as exc:
        app.logger.error("Unexpected error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": f"processing failed: {str(exc)}"}), 500
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
