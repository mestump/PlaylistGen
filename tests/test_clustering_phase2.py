"""Tests for Phase 2 clustering improvements in playlistgen/clustering.py"""

import pandas as pd
import pytest

from playlistgen.clustering import (
    cluster_by_audio_features,
    cluster_tracks,
    _cluster_hybrid_impl,
)


def _make_df(n=50, with_energy=True, with_mood=True):
    """Helper: generate a test library DataFrame."""
    import random
    moods = ["Happy", "Sad", "Chill", "Energetic", "Groovy"]
    genres = ["Rock", "Pop", "Jazz", "Classical", "Hip-Hop"]
    rows = []
    for i in range(n):
        row = {
            "Name": f"Track {i}",
            "Artist": f"Artist {i % 10}",
            "Genre": genres[i % len(genres)],
            "Score": float(i),
        }
        if with_mood:
            row["Mood"] = moods[i % len(moods)]
        if with_energy:
            row["Energy"] = 0.01 + (i % 10) * 0.005
            row["BPM"] = 80.0 + (i % 10) * 10
            row["SpectralBrightness"] = 0.2 + (i % 5) * 0.05
            row["ZCR"] = 0.05 + (i % 5) * 0.01
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cluster_by_audio_features
# ---------------------------------------------------------------------------

def test_cluster_by_audio_features_returns_clusters():
    df = _make_df(n=60, with_energy=True)
    clusters = cluster_by_audio_features(df, n_clusters=4)
    assert len(clusters) > 0
    total = sum(len(c) for c in clusters)
    assert total == len(df)


def test_cluster_by_audio_features_low_coverage_returns_empty():
    """Returns [] when <30% of tracks have Energy data."""
    df = _make_df(n=50, with_energy=False)
    # Energy column absent
    clusters = cluster_by_audio_features(df, n_clusters=4)
    assert clusters == []


def test_cluster_by_audio_features_partial_coverage():
    """Returns [] when Energy coverage is below threshold."""
    df = _make_df(n=50, with_energy=True)
    # Set 80% of Energy values to NaN → coverage 20%
    df.loc[:39, "Energy"] = None
    clusters = cluster_by_audio_features(df, n_clusters=4)
    assert clusters == []


def test_cluster_by_audio_features_single_cluster():
    df = _make_df(n=10, with_energy=True)
    clusters = cluster_by_audio_features(df, n_clusters=1)
    assert len(clusters) == 1
    assert len(clusters[0]) == 10


# ---------------------------------------------------------------------------
# _cluster_hybrid_impl
# ---------------------------------------------------------------------------

def test_cluster_hybrid_groups_by_mood_first():
    df = _make_df(n=50, with_energy=True, with_mood=True)
    clusters = _cluster_hybrid_impl(df, n_audio_subclusters=2)
    assert len(clusters) > 0
    total = sum(len(c) for c in clusters)
    assert total == len(df)


def test_cluster_hybrid_no_mood_returns_empty():
    df = _make_df(n=30, with_energy=True, with_mood=False)
    clusters = _cluster_hybrid_impl(df)
    assert clusters == []


def test_cluster_hybrid_unknown_mood_skipped():
    df = _make_df(n=30, with_energy=True, with_mood=True)
    df["Mood"] = "Unknown"  # all unknown
    clusters = _cluster_hybrid_impl(df)
    assert clusters == []


# ---------------------------------------------------------------------------
# cluster_tracks — auto strategy
# ---------------------------------------------------------------------------

def test_auto_strategy_selects_audio_when_energy_available():
    df = _make_df(n=60, with_energy=True, with_mood=True)
    # With >30% energy coverage, auto should choose audio strategy
    clusters = cluster_tracks(df, n_clusters=4, strategy="auto")
    assert len(clusters) > 0


def test_auto_strategy_falls_back_to_mood():
    df = _make_df(n=60, with_energy=False, with_mood=True)
    # No energy → should use mood strategy
    clusters = cluster_tracks(df, n_clusters=6, strategy="auto")
    assert len(clusters) > 0


def test_auto_strategy_tfidf_fallback():
    df = _make_df(n=30, with_energy=False, with_mood=False)
    # No energy, no mood → tfidf
    clusters = cluster_tracks(df, n_clusters=3, strategy="auto")
    assert len(clusters) > 0


def test_explicit_audio_strategy():
    df = _make_df(n=60, with_energy=True)
    clusters = cluster_tracks(df, n_clusters=4, strategy="audio")
    assert len(clusters) > 0


def test_explicit_tfidf_strategy():
    df = _make_df(n=30, with_energy=False)
    clusters = cluster_tracks(df, n_clusters=3, strategy="tfidf")
    assert len(clusters) > 0


def test_hybrid_mode():
    df = _make_df(n=60, with_energy=True, with_mood=True)
    clusters = cluster_tracks(df, n_clusters=6, cluster_hybrid_mode=True)
    assert len(clusters) > 0


def test_existing_mood_strategy_still_works():
    df = _make_df(n=50, with_energy=False, with_mood=True)
    clusters = cluster_tracks(df, n_clusters=5, cluster_by_mood=True)
    assert len(clusters) > 0


def test_cluster_tracks_no_data_falls_back():
    """Empty energy + empty mood → tfidf path doesn't crash."""
    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Artist": ["X", "Y", "Z"],
        "Genre": ["Rock", "Pop", "Jazz"],
        "Score": [1.0, 2.0, 3.0],
    })
    clusters = cluster_tracks(df, n_clusters=2, strategy="auto")
    assert len(clusters) > 0
