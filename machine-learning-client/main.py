"""Machine Learning Client using YAMNET"""

import os
import csv
import numpy as np
import librosa
import tensorflow_hub as hub

AUDIO_FILE = "data.wav"
TARGET_SAMPLE_RATE = 16000
CLASS_MAP_FILE = "yamnet_class_map.csv"

INSTRUMENT_KEYWORDS = [
    "guitar",
    "piano",
    "violin",
    "cello",
    "flute",
    "trumpet",
    "saxophone",
    "drum",
    "bass",
    "cymbal",
    "harp",
    "organ",
    "synth",
    "banjo",
    "ukulele",
    "accordion",
    "harmonica",
    "clarinet",
    "trombone",
    "tuba",
    "fiddle",
    "oboe",
    "bassoon",
    "xylophone",
    "marimba",
    "vibraphone",
    "string",
    "percussion",
    "wind",
]


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


def load_audio(file_path, target_sr):
    """Load audio file to target sample rate"""
    try:
        waveform, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return waveform
    except FileNotFoundError:
        print(f"Error: Audio file not found at '{file_path}'")
        return None
    except librosa.util.exceptions.ParameterError as e:
        print(f"Error loading audio file: {e}")
        return None


def main():
    """Main function to detect instrument"""
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found.")
        return
    class_names = load_class_map()
    if not class_names:
        return
    print("Loading YAMNet model.")
    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("Model loaded successfully.")
    except (OSError, RuntimeError) as e:
        print(f"Error loading model: {e}")
        return
    print(f"Loading audio from {AUDIO_FILE}.")
    waveform = load_audio(AUDIO_FILE, TARGET_SAMPLE_RATE)
    if waveform is None:
        return
    scores, _, _ = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)
    results = []
    for i, score in enumerate(mean_scores):
        results.append({"class_name": class_names[i], "score": score})
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    most_likely_instrument = None
    for res in sorted_results:
        if res["score"] < 0.01:
            break
        is_instrument = False
        for keyword in INSTRUMENT_KEYWORDS:
            if keyword in res["class_name"].lower():
                most_likely_instrument = res
                is_instrument = True
                break

        if is_instrument:
            break
    if most_likely_instrument:
        print(
            f"{most_likely_instrument['class_name']}: {most_likely_instrument['score']:.4f}"
        )
    else:
        print("No specific musical instruments detected.")


if __name__ == "__main__":
    main()
    