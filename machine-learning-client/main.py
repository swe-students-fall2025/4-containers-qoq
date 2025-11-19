"""Machine Learning Client using YAMNET"""

import os
import csv
from datetime import datetime, timezone

import numpy as np
import librosa
import tensorflow_hub as hub
from pymongo import MongoClient

AUDIO_FILE = os.environ.get("AUDIO_FILE", "data.wav")
TARGET_SAMPLE_RATE = 16000
CLASS_MAP_FILE = "yamnet_class_map.csv"

# MongoDB configuration
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ml_logs")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "predictions")
SOURCE_NAME = os.environ.get("SOURCE_NAME", "ml-client")

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

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

try:
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("Model loaded successfully.")
except (OSError, RuntimeError) as e:
    print(f"Error loading model: {e}")
    model = None


def load_class_map():
    """Load class names from csv"""
    if not os.path.exists(CLASS_MAP_FILE):
        print(f"Error: Class map file not found at '{CLASS_MAP_FILE}'")
        return None

    try:
        with open(CLASS_MAP_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                print(f"Error: {CLASS_MAP_FILE} is empty.")
                return None
            class_names = [row[2] for row in reader if row]

        if not class_names:
            print(f"Error: Could not read any class names from {CLASS_MAP_FILE}.")
            return None
        return class_names

    except (OSError, csv.Error) as e:
        print(f"Error reading class map file: {e}")
        return None


CLASS_NAMES = load_class_map()


def load_audio(file_path, target_sr):
    """Load audio file to target sample rate"""
    try:
        waveform, _ = librosa.load(file_path, sr=target_sr, mono=True)
        waveform = librosa.util.normalize(waveform)
        return waveform
    except FileNotFoundError:
        print(f"Error: Audio file not found at '{file_path}'")
        return None
    except librosa.util.exceptions.ParameterError as e:
        print(f"Error loading audio file: {e}")
        return None


def get_collection():
    """Return the MongoDB collection used to store predictions."""
    db = mongo_client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]


def save_prediction(instrument, confidence):
    """Save prediction result to MongoDB."""
    now = datetime.now(timezone.utc)
    doc = {
        "instrument": instrument,
        "confidence": float(confidence),
        "source": SOURCE_NAME,
        "captured_at": now,
        "created_at": now,
    }
    try:
        col = get_collection()
        result = col.insert_one(doc)
        print(f"Saved prediction to MongoDB with ID: {result.inserted_id}")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error saving to MongoDB: {e}")


def classify_waveform(waveform):
    """Classify the audio file and return sorted results."""
    if model is None:
        return None
    scores, _, _ = model(waveform)
    processed_scores = np.mean(scores.numpy(), axis=0)
    results = []
    for i, score in enumerate(processed_scores):
        results.append({"class_name": CLASS_NAMES[i], "score": score})
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    for res in sorted_results:
        if res["score"] < 0.005:
            break
        if res["class_name"] in ALLOWED_INSTRUMENTS:
            return res
    return None


def main():
    """Main function"""
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found.")
        return
    print(f"Loading audio from {AUDIO_FILE}.")
    waveform = load_audio(AUDIO_FILE, TARGET_SAMPLE_RATE)
    if waveform is not None:
        result = classify_waveform(waveform)
        if result:
            print(f"Detected: {result['class_name']} ({result['score']:.4f})")
            save_prediction(result["class_name"], result["score"])
        else:
            print("No specific musical instruments detected.")
            save_prediction("unknown", 0.0)


if __name__ == "__main__":
    main()
