"""Tests for the YAMNet-based machine-learning client."""

import types
from datetime import datetime, timezone

import numpy as np
import main


# ---------------------------------------------------------------------------
# Tests for load_class_map
# ---------------------------------------------------------------------------


def test_load_class_map_missing_file_returns_none(tmp_path, monkeypatch, capsys):
    """If CLASS_MAP_FILE does not exist, load_class_map returns None and prints an error."""
    missing = tmp_path / "no_such_file.csv"
    monkeypatch.setattr(main, "CLASS_MAP_FILE", str(missing))

    result = main.load_class_map()

    captured = capsys.readouterr()
    assert "Class map file not found" in captured.out
    assert result is None


def test_load_class_map_reads_display_names(tmp_path, monkeypatch):
    """load_class_map returns list of class names from the 3rd column."""
    csv_path = tmp_path / "yamnet_class_map.csv"
    csv_content = (
        "index,mid,display_name\n"
        "0,/m/04rlf,guitar\n"
        "1,/m/05r5c,piano\n"
        "2,/m/0fx9l,violin\n"
    )
    csv_path.write_text(csv_content, encoding="utf-8")

    monkeypatch.setattr(main, "CLASS_MAP_FILE", str(csv_path))

    result = main.load_class_map()

    assert result == ["guitar", "piano", "violin"]


# ---------------------------------------------------------------------------
# Tests for load_audio
# ---------------------------------------------------------------------------


def test_load_audio_missing_file_returns_none(tmp_path, capsys):
    """If audio file is missing, load_audio returns None and prints an error."""
    missing = tmp_path / "no_audio.wav"

    result = main.load_audio(str(missing), main.TARGET_SAMPLE_RATE)

    captured = capsys.readouterr()
    assert "Audio file not found" in captured.out
    assert result is None


def test_load_audio_calls_librosa_and_returns_waveform(monkeypatch):
    """load_audio should call librosa.load and return the waveform."""
    fake_waveform = np.array([0.1, 0.2], dtype=float)

    def fake_load(path, sr=None, mono=None):
        # Assert the arguments are passed correctly
        assert path == "somefile.wav"
        assert sr == main.TARGET_SAMPLE_RATE
        assert mono is True
        return fake_waveform, sr

    # ✅ Patch only librosa.load, not the whole librosa module
    monkeypatch.setattr(main.librosa, "load", fake_load)

    result = main.load_audio("somefile.wav", main.TARGET_SAMPLE_RATE)

    # Should return exactly the waveform from fake_load
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, fake_waveform)


# ---------------------------------------------------------------------------
# Tests for save_prediction
# ---------------------------------------------------------------------------


class DummyInsertOneResult:  # pylint: disable=too-few-public-methods
    """Simple dummy result with inserted_id."""

    def __init__(self, inserted_id="fake_id"):
        self.inserted_id = inserted_id


def test_save_prediction_success(monkeypatch, capsys):
    """save_prediction should insert a document into Mongo."""
    saved_docs = []

    class DummyCollection:  # pylint: disable=too-few-public-methods
        """Fake Mongo collection."""

        def insert_one(self, doc):
            """Mock insert_one that stores the document and returns a dummy result."""
            saved_docs.append(doc)
            return DummyInsertOneResult(inserted_id="123456")

    def fake_get_collection():
        return DummyCollection()

    monkeypatch.setattr(main, "get_collection", fake_get_collection)

    main.save_prediction("guitar", 0.95)

    captured = capsys.readouterr()
    assert "Saved prediction to MongoDB with ID: 123456" in captured.out

    assert len(saved_docs) == 1
    doc = saved_docs[0]
    assert doc["instrument"] == "guitar"
    assert doc["confidence"] == 0.95
    assert doc["source"] == main.SOURCE_NAME
    assert isinstance(doc["captured_at"], datetime)
    assert isinstance(doc["created_at"], datetime)
    assert doc["captured_at"].tzinfo is timezone.utc
    assert doc["created_at"].tzinfo is timezone.utc


def test_save_prediction_failure(monkeypatch, capsys):
    """If Mongo insert fails, save_prediction should print an error."""

    class FailingCollection:  # pylint: disable=too-few-public-methods
        """A mock MongoDB collection that always fails on insert_one."""

        def insert_one(self, doc):  # noqa: ARG002
            """Simulate a database failure by raising RuntimeError."""
            raise RuntimeError("boom")

    def fake_get_collection():
        return FailingCollection()

    monkeypatch.setattr(main, "get_collection", fake_get_collection)

    main.save_prediction("piano", 0.5)

    captured = capsys.readouterr()
    assert "Error saving to MongoDB" in captured.out


# ---------------------------------------------------------------------------
# Tests for main.main orchestrator
# ---------------------------------------------------------------------------


def test_main_exits_early_if_audio_missing(monkeypatch, tmp_path, capsys):
    """If AUDIO_FILE does not exist, main() prints an error and returns None."""
    missing_audio = tmp_path / "no_audio_here.wav"
    monkeypatch.setattr(main, "AUDIO_FILE", str(missing_audio))

    # If hub.load were called, we'd know something went wrong.
    def fake_hub_load(_url):  # noqa: ARG001
        raise AssertionError("hub.load should not be called when audio file is missing")

    fake_hub_module = types.SimpleNamespace(load=fake_hub_load)
    monkeypatch.setattr(main, "hub", fake_hub_module)

    main.main()

    captured = capsys.readouterr()
    assert f"Error: {missing_audio} not found." in captured.out


def test_main_happy_path_detects_instrument_and_saves(monkeypatch, tmp_path, capsys):
    """Happy path: main() loads model, processes audio, detects instrument, and saves prediction."""
    audio_path = tmp_path / "data.wav"
    audio_path.write_bytes(b"fake audio bytes")  # existence is enough

    monkeypatch.setattr(main, "AUDIO_FILE", str(audio_path))

    # 1) Fake class map
    fake_class_names = ["guitar", "piano", "drums"]

    def fake_load_class_map():
        return fake_class_names

    monkeypatch.setattr(main, "load_class_map", fake_load_class_map)

    # 2) Fake model (TF Hub)
    class DummyScores:  # pylint: disable=too-few-public-methods
        """Mock scores object that mimics a TensorFlow tensor."""

        def __init__(self, arr):
            """Store the underlying numpy array."""
            self._arr = arr

        def numpy(self):
            """Return the stored array."""
            return self._arr

    class DummyModel:  # pylint: disable=too-few-public-methods
        """Mock YAMNet model returning fixed scores."""

        def __call__(self, waveform):  # noqa: ARG002
            """Return deterministic scores instead of real model output."""
            # Shape [frames, classes]; highest score is for "guitar"
            scores = DummyScores(np.array([[0.9, 0.05, 0.05]]))
            return scores, None, None

    def fake_hub_load(_url):  # noqa: ARG001
        """Return a DummyModel instead of loading from TensorFlow Hub."""
        return DummyModel()

    fake_hub_module = types.SimpleNamespace(load=fake_hub_load)
    monkeypatch.setattr(main, "hub", fake_hub_module)

    # 3) Fake load_audio, just return a dummy waveform
    def fake_load_audio(path, target_sr):
        assert str(path) == str(audio_path)
        assert target_sr == main.TARGET_SAMPLE_RATE
        return np.zeros(target_sr, dtype=float)

    monkeypatch.setattr(main, "load_audio", fake_load_audio)

    # 4) Fake save_prediction to capture what gets saved
    saved = {}

    def fake_save_prediction(instrument, confidence):
        saved["instrument"] = instrument
        saved["confidence"] = float(confidence)
        return True

    monkeypatch.setattr(main, "save_prediction", fake_save_prediction)

    # 5) Ensure instrument keywords include guitar
    monkeypatch.setattr(
        main,
        "ALLOWED_INSTRUMENTS",
        {"guitar", "piano", "violin"},
    )

    main.main()

    captured = capsys.readouterr()

    # Assert we printed the detected instrument
    assert "Detected: guitar (confidence:" in captured.out

    # Saved prediction should match detection
    assert saved["instrument"] == "guitar"
    assert np.isclose(saved["confidence"], 0.9)