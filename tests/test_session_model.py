"""Tests for playlistgen/session_model.py"""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from playlistgen.session_model import (
    load_streaming_history,
    build_sessions,
    build_cooccurrence_matrix,
    recency_scores,
    build_session_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_history_json(entries, path):
    """Write a streaming history JSON file."""
    with open(path, "w") as f:
        json.dump(entries, f)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# load_streaming_history
# ---------------------------------------------------------------------------

def test_load_classic_format():
    with tempfile.TemporaryDirectory() as tmp:
        entries = [
            {"endTime": "2024-01-01 10:00", "artistName": "Radiohead", "trackName": "Karma Police", "msPlayed": 250000},
            {"endTime": "2024-01-01 10:04", "artistName": "Beck", "trackName": "Loser", "msPlayed": 200000},
        ]
        p = Path(tmp) / "StreamingHistory0.json"
        _make_history_json(entries, p)
        df = load_streaming_history(str(p))
    assert len(df) == 2
    assert "track_id" in df.columns
    assert "radiohead - karma police" in df["track_id"].values


def test_load_extended_format():
    with tempfile.TemporaryDirectory() as tmp:
        entries = [
            {
                "ts": "2024-01-01T10:00:00Z",
                "master_metadata_track_artist_name": "Radiohead",
                "master_metadata_track_name": "Creep",
                "ms_played": 220000,
            }
        ]
        p = Path(tmp) / "StreamingHistory0.json"
        _make_history_json(entries, p)
        df = load_streaming_history(str(p))
    assert len(df) == 1
    assert "radiohead - creep" in df["track_id"].values


def test_load_directory_finds_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            entries = [
                {"endTime": f"2024-01-0{i+1} 10:00", "artistName": f"Artist{i}", "trackName": f"Track{i}", "msPlayed": 100000},
            ]
            _make_history_json(entries, Path(tmp) / f"StreamingHistory{i}.json")
        df = load_streaming_history(tmp)
    assert len(df) == 3


def test_load_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "empty.json"
        _make_history_json([], p)
        df = load_streaming_history(str(p))
    assert df.empty


def test_load_invalid_file_returns_empty():
    df = load_streaming_history("/nonexistent/path/history.json")
    assert df.empty


# ---------------------------------------------------------------------------
# build_sessions
# ---------------------------------------------------------------------------

def test_sessions_single_session():
    """Plays within 30 min form one session."""
    base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [base, base + timedelta(minutes=5), base + timedelta(minutes=10)],
        "track_id": ["a - x", "b - y", "c - z"],
        "ms_played": [200000, 200000, 200000],
    })
    sessions = build_sessions(df, gap_minutes=30)
    assert len(sessions) == 1
    assert len(sessions[0]) == 3


def test_sessions_splits_on_gap():
    """Gap > 30 min creates a new session."""
    base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [base, base + timedelta(minutes=5), base + timedelta(minutes=60)],
        "track_id": ["a", "b", "c"],
        "ms_played": [200000, 200000, 200000],
    })
    sessions = build_sessions(df, gap_minutes=30)
    assert len(sessions) == 2


def test_sessions_skips_short_plays():
    """Plays under 30s are excluded."""
    base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame({
        "timestamp": [base, base + timedelta(minutes=1), base + timedelta(minutes=2)],
        "track_id": ["a", "b", "c"],
        "ms_played": [200000, 5000, 200000],  # middle play is < 30s
    })
    sessions = build_sessions(df, gap_minutes=30, min_ms_played=30000)
    # b is skipped; a and c are within 2 min → same session
    assert len(sessions) == 1
    assert "b" not in sessions[0]


def test_sessions_empty_input():
    assert build_sessions(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# build_cooccurrence_matrix
# ---------------------------------------------------------------------------

def test_cooccurrence_basic():
    sessions = [["a", "b", "c"], ["a", "d"]]
    matrix = build_cooccurrence_matrix(sessions)
    # a co-occurs with b, c (from session 1) and d (from session 2)
    assert matrix["a"]["b"] == 1
    assert matrix["a"]["c"] == 1
    assert matrix["a"]["d"] == 1
    assert matrix["b"]["a"] == 1
    # b and d never in same session
    assert matrix.get("b", {}).get("d", 0) == 0


def test_cooccurrence_counts_accumulate():
    sessions = [["a", "b"], ["a", "b"], ["a", "b"]]
    matrix = build_cooccurrence_matrix(sessions)
    assert matrix["a"]["b"] == 3
    assert matrix["b"]["a"] == 3


def test_cooccurrence_empty():
    assert build_cooccurrence_matrix([]) == {}


# ---------------------------------------------------------------------------
# recency_scores
# ---------------------------------------------------------------------------

def test_recency_recent_play_scores_high():
    now = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    recent = datetime(2024, 5, 31, tzinfo=timezone.utc)  # 1 day ago
    old = datetime(2024, 1, 1, tzinfo=timezone.utc)      # ~150 days ago
    df = pd.DataFrame({
        "timestamp": [recent, old],
        "track_id": ["recent_track", "old_track"],
        "ms_played": [200000, 200000],
    })
    scores = recency_scores(df, half_life_days=90, now=now)
    assert scores["recent_track"] > scores["old_track"]


def test_recency_normalised_to_one():
    now = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 5, 31, tzinfo=timezone.utc)],
        "track_id": ["x"],
        "ms_played": [200000],
    })
    scores = recency_scores(df, half_life_days=90, now=now)
    assert max(scores.values()) <= 1.0 + 1e-9


def test_recency_empty_input():
    assert recency_scores(pd.DataFrame()) == {}


# ---------------------------------------------------------------------------
# build_session_model
# ---------------------------------------------------------------------------

def test_build_session_model_integration():
    with tempfile.TemporaryDirectory() as tmp:
        base = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        entries = [
            {"endTime": _iso(base), "artistName": "Radiohead", "trackName": "Karma Police", "msPlayed": 250000},
            {"endTime": _iso(base + timedelta(minutes=4)), "artistName": "Beck", "trackName": "Loser", "msPlayed": 200000},
            {"endTime": _iso(base + timedelta(hours=2)), "artistName": "Portishead", "trackName": "Sour Times", "msPlayed": 220000},
        ]
        p = Path(tmp) / "StreamingHistory0.json"
        _make_history_json(entries, p)
        model = build_session_model(str(p))

    assert "cooccurrence" in model
    assert "recency" in model
    assert "play_counts" in model
    # Radiohead and Beck are in same session → co-occur
    assert model["cooccurrence"].get("radiohead - karma police", {}).get("beck - loser", 0) > 0


def test_build_session_model_empty():
    model = build_session_model("/nonexistent/path")
    assert model["cooccurrence"] == {}
    assert model["recency"] == {}
    assert model["play_counts"] == {}
