"""Tests for scoring.py — verify all bugs are fixed."""

import pandas as pd
import pytest
from playlistgen.scoring import score_tracks, top_tracks


def _make_df(**kwargs):
    """Build a minimal library DataFrame for testing."""
    base = {
        "Name": ["Song A", "Song B", "Song C"],
        "Artist": ["Artist 1", "Artist 2", "Artist 1"],
        "Genre": ["Rock", "Pop", "Jazz"],
        "Location": ["/music/a.mp3", "/music/b.mp3", "/music/c.mp3"],
        "Play Count": [10, 5, 0],
        "Skip Count": [0, 1, 0],
        "Year": [2005, 1995, 2015],
        "BPM": [None, None, None],
        "Duration": [None, None, None],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _make_profile():
    return {
        "artist_scores": {"Artist 1": 100000, "Artist 2": 50000},
        "genre_scores": {"rock": 30, "pop": 10},  # lowercase keys
        "mood_scores": {"Happy": 20, "Chill": 10},
        "year_scores": {"2005": 15, "1995": 5},
        "track_play_counts": {"artist 1 - song a": 3},
        "track_skip_counts": {},
    }


def _make_tag_db():
    return {
        "artist 1 - song a": ["happy", "upbeat", "rock"],
        "artist 2 - song b": ["chill", "pop"],
        "artist 1 - song c": ["jazz", "smooth"],
    }


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

def test_score_tracks_returns_dataframe():
    df = _make_df()
    result = score_tracks(df, config=_make_profile(), tag_mood_db=_make_tag_db())
    assert "Score" in result.columns
    assert "Mood" in result.columns
    assert len(result) == 3


def test_score_tracks_preserves_row_count():
    df = _make_df()
    result = score_tracks(df, config=_make_profile(), tag_mood_db={})
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Genre scores (FIXED: was always 0)
# ---------------------------------------------------------------------------

def test_genre_score_contributes_to_total():
    """Genre-matching track must score higher than non-matching track (all else equal)."""
    df = pd.DataFrame({
        "Name": ["Rock Song", "Unknown Song"],
        "Artist": ["X", "X"],
        "Genre": ["Rock", "Unknown"],
        "Location": ["/a.mp3", "/b.mp3"],
        "Play Count": [0, 0],
        "Skip Count": [0, 0],
        "Year": [None, None],
    })
    profile = {
        "artist_scores": {},
        "genre_scores": {"rock": 50},  # Rock should boost score
        "mood_scores": {},
        "year_scores": {},
        "track_play_counts": {},
        "track_skip_counts": {},
    }
    result = score_tracks(df, config=profile, tag_mood_db={})
    rock_score = result[result["Genre"] == "Rock"]["Score"].iloc[0]
    unknown_score = result[result["Genre"] == "Unknown"]["Score"].iloc[0]
    assert rock_score > unknown_score, "Genre affinity should raise the Rock track's score"


# ---------------------------------------------------------------------------
# Year scores (FIXED: was always 0 due to path parsing)
# ---------------------------------------------------------------------------

def test_year_score_uses_year_column():
    """Year from the DataFrame column must affect scoring (not file path)."""
    df = pd.DataFrame({
        "Name": ["Old Song", "New Song"],
        "Artist": ["X", "X"],
        "Genre": ["Rock", "Rock"],
        "Location": ["/music/no_year_in_path.mp3", "/music/also_no_year.mp3"],
        "Play Count": [0, 0],
        "Skip Count": [0, 0],
        "Year": [2000, 2010],
    })
    # User listens mostly to 2000s music
    profile = {
        "artist_scores": {},
        "genre_scores": {},
        "mood_scores": {},
        "year_scores": {"2000": 100, "2010": 1},
        "track_play_counts": {},
        "track_skip_counts": {},
    }
    result = score_tracks(df, config=profile, tag_mood_db={})
    old_score = result[result["Year"] == 2000]["Score"].iloc[0]
    new_score = result[result["Year"] == 2010]["Score"].iloc[0]
    assert old_score > new_score, "Year 2000 should score higher than 2010 for this profile"


def test_year_score_zero_for_missing_year():
    """Tracks with no Year should still get a score without crashing."""
    df = pd.DataFrame({
        "Name": ["Song"],
        "Artist": ["X"],
        "Genre": ["Rock"],
        "Location": ["/music/song.mp3"],
        "Play Count": [5],
        "Skip Count": [0],
        "Year": [None],
    })
    result = score_tracks(df, config={}, tag_mood_db={})
    assert "Score" in result.columns
    assert not result["Score"].isna().any()


# ---------------------------------------------------------------------------
# Mood column (FIXED: was always dict-format dependent)
# ---------------------------------------------------------------------------

def test_mood_column_populated_from_list_tags():
    """New tag_db format (List[str]) should produce non-Unknown Mood values."""
    df = _make_df()
    tag_db = {
        "artist 1 - song a": ["happy", "upbeat"],
        "artist 2 - song b": ["chill", "mellow"],
        "artist 1 - song c": [],
    }
    result = score_tracks(df, config={}, tag_mood_db=tag_db)
    moods = result["Mood"].tolist()
    assert moods[0] == "Happy", f"Expected Happy, got {moods[0]}"
    assert moods[1] == "Chill", f"Expected Chill, got {moods[1]}"


def test_mood_column_legacy_dict_format():
    """Old tag_db format (dict with 'tags' key) should still work."""
    df = _make_df()
    tag_db = {
        "artist 1 - song a": {"tags": ["happy", "upbeat"], "mood": "Happy"},
    }
    result = score_tracks(df, config={}, tag_mood_db=tag_db)
    assert result.iloc[0]["Mood"] == "Happy"


def test_mood_column_genre_fallback():
    """When no tags match, mood should fall back to genre-based mapping."""
    df = pd.DataFrame({
        "Name": ["Jazz Track"],
        "Artist": ["Miles Davis"],
        "Genre": ["Jazz"],
        "Location": ["/a.flac"],
        "Play Count": [0],
        "Skip Count": [0],
        "Year": [None],
    })
    result = score_tracks(df, config={}, tag_mood_db={})
    assert result.iloc[0]["Mood"] == "Chill", "Jazz genre should fall back to Chill mood"


# ---------------------------------------------------------------------------
# Skip penalty
# ---------------------------------------------------------------------------

def test_skip_lowers_score():
    df = pd.DataFrame({
        "Name": ["Track A", "Track B"],
        "Artist": ["X", "X"],
        "Genre": ["Rock", "Rock"],
        "Location": ["/a.mp3", "/b.mp3"],
        "Play Count": [5, 5],
        "Skip Count": [0, 3],
        "Year": [None, None],
    })
    result = score_tracks(df, config={}, tag_mood_db={})
    no_skip = result[result["Name"] == "Track A"]["Score"].iloc[0]
    with_skip = result[result["Name"] == "Track B"]["Score"].iloc[0]
    assert no_skip > with_skip


# ---------------------------------------------------------------------------
# top_tracks helper
# ---------------------------------------------------------------------------

def test_top_tracks():
    df = _make_df()
    result = score_tracks(df, config=_make_profile(), tag_mood_db=_make_tag_db())
    top = top_tracks(result, n=2)
    assert len(top) == 2
    scores = top["Score"].tolist()
    assert scores[0] >= scores[1]
