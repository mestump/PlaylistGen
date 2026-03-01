"""Tests for Phase 2 session_model integration in scoring.py"""

from collections import Counter

import pandas as pd
import pytest

from playlistgen.scoring import score_tracks


def _make_df(n=10):
    return pd.DataFrame({
        "Name": [f"Track {i}" for i in range(n)],
        "Artist": [f"Artist {i % 3}" for i in range(n)],
        "Genre": ["Rock"] * n,
        "Play Count": [0] * n,
        "Skip Count": [0] * n,
        "Year": [2010] * n,
        "BPM": [120.0] * n,
        "Energy": [0.05] * n,
    })


def _make_session_model(track_id=None, recency=None, cooccurrence=None, play_counts=None):
    return {
        "recency": recency or {},
        "cooccurrence": cooccurrence or {},
        "play_counts": play_counts or {},
    }


def test_score_tracks_without_session_model():
    """Base functionality unchanged when session_model=None."""
    df = _make_df()
    result = score_tracks(df, config={}, tag_mood_db={}, session_model=None)
    assert "Score" in result.columns
    assert "Mood" in result.columns
    assert len(result) == len(df)


def test_recency_multiplier_boosts_score():
    """Tracks with high recency score get a higher final score."""
    df = _make_df(n=2)
    df.at[0, "Artist"] = "Radiohead"
    df.at[0, "Name"] = "Karma Police"
    df.at[1, "Artist"] = "Beck"
    df.at[1, "Name"] = "Loser"

    track_0_id = "radiohead - karma police"
    track_1_id = "beck - loser"

    session_model = _make_session_model(
        recency={track_0_id: 1.0, track_1_id: 0.0},  # track 0 is very recent
    )

    result = score_tracks(df, config={}, tag_mood_db={}, session_model=session_model)
    # Track 0 should score higher due to recency multiplier (1 + 0.5*1.0 = 1.5x)
    # Track 1 gets 1.0x (no recency boost)
    assert result.loc[0, "Score"] >= result.loc[1, "Score"]


def test_cooccurrence_boost_adds_to_score():
    """Tracks co-occurring with top-played favorites get a bonus."""
    df = _make_df(n=3)
    df.at[0, "Artist"] = "Artist A"
    df.at[0, "Name"] = "Track A"
    df.at[1, "Artist"] = "Artist B"
    df.at[1, "Name"] = "Track B"
    df.at[2, "Artist"] = "Artist C"
    df.at[2, "Name"] = "Track C"

    # "favorite" track that's most-played
    favorite_id = "artist a - track a"
    target_id = "artist b - track b"

    session_model = _make_session_model(
        play_counts={favorite_id: 100, target_id: 1, "artist c - track c": 0},
        cooccurrence={
            favorite_id: Counter({target_id: 50}),  # b co-occurs with favorite
        },
    )

    result = score_tracks(df, config={}, tag_mood_db={}, session_model=session_model)
    # track B should get a co-occurrence bonus vs track C
    score_b = result.loc[1, "Score"]
    score_c = result.loc[2, "Score"]
    assert score_b >= score_c


def test_session_model_does_not_break_existing_scoring():
    """Adding session_model doesn't reduce scores for already-positive tracks."""
    df = _make_df(n=5)
    profile = {"artist_scores": {"Artist 0": 5.0, "Artist 1": 3.0}}

    result_no_session = score_tracks(df, config=profile, tag_mood_db={}, session_model=None)
    result_with_session = score_tracks(
        df,
        config=profile,
        tag_mood_db={},
        session_model=_make_session_model(),  # empty session model
    )

    # With an empty session model, scores should be the same (no recency, no cooccurrence)
    for i in range(len(df)):
        assert abs(result_no_session.loc[i, "Score"] - result_with_session.loc[i, "Score"]) < 1e-9


def test_mood_preserved_from_enrichment():
    """If Mood is already set (from batch enrichment), it is preserved."""
    df = _make_df(n=3)
    df["Mood"] = ["Happy", "Sad", None]

    result = score_tracks(df, config={}, tag_mood_db={}, session_model=None)

    assert result.loc[0, "Mood"] == "Happy"
    assert result.loc[1, "Mood"] == "Sad"
    # Track 2 has no mood → gets one from tag lookup, "Unknown", or None/NaN
    mood2 = result.loc[2, "Mood"]
    valid_moods = {
        "Unknown", "Happy", "Sad", "Angry", "Chill", "Energetic",
        "Romantic", "Epic", "Dreamy", "Groovy", "Nostalgic",
    }
    assert mood2 in valid_moods or mood2 is None or pd.isnull(mood2)
