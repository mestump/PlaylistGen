"""Tests for spotify_profile.py — verify genre_scores is now populated."""

import json
import tempfile
from pathlib import Path
import datetime

import pytest
from playlistgen.spotify_profile import build_profile, load_profile


def _write_spotify_file(tmp_path, entries):
    """Write a Spotify streaming-history JSON file."""
    p = tmp_path / "StreamingHistory.json"
    p.write_text(json.dumps(entries))
    return tmp_path


def _make_entries(artist="Artist A", track="Track 1", ms=30000, skipped=False):
    return [{
        "master_metadata_album_artist_name": artist,
        "master_metadata_track_name": track,
        "ms_played": ms,
        "skipped": skipped,
        "ts": "2022-03-15T14:00:00Z",
    }]


# ---------------------------------------------------------------------------
# genre_scores population (FIXED BUG)
# ---------------------------------------------------------------------------

def test_genre_scores_populated(tmp_path):
    """
    genre_scores must be non-empty when the tag_db contains genre-mappable tags.
    Previously this was always an empty dict.
    """
    entries = _make_entries()
    spotify_dir = _write_spotify_file(tmp_path, entries)

    # tag_db with tags that map to known genres
    tag_db = {"artist a - track 1": ["rock", "indie"]}

    profile = build_profile(
        spotify_dir=str(spotify_dir),
        out_path=str(tmp_path / "profile.json"),
        tag_db=tag_db,
    )
    assert profile.get("genre_scores"), (
        "genre_scores should be populated from tag_scores via canonical_genre(); "
        "got: " + repr(profile.get("genre_scores"))
    )
    # "rock" maps to "Rock" → genre_scores should have "rock" (lowercase key)
    assert "rock" in profile["genre_scores"] or "Rock" in profile.get("genre_scores", {}), \
        "Expected 'rock' in genre_scores"


def test_tag_scores_populated(tmp_path):
    entries = _make_entries()
    spotify_dir = _write_spotify_file(tmp_path, entries)
    tag_db = {"artist a - track 1": ["rock", "indie"]}

    profile = build_profile(
        spotify_dir=str(spotify_dir),
        out_path=str(tmp_path / "profile.json"),
        tag_db=tag_db,
    )
    assert "rock" in profile["tag_scores"]
    assert "indie" in profile["tag_scores"]


def test_artist_scores_populated(tmp_path):
    entries = _make_entries(artist="Radiohead", ms=60000)
    spotify_dir = _write_spotify_file(tmp_path, entries)

    profile = build_profile(
        spotify_dir=str(spotify_dir),
        out_path=str(tmp_path / "profile.json"),
        tag_db={},
    )
    assert "Radiohead" in profile["artist_scores"]
    assert profile["artist_scores"]["Radiohead"] == 60000


def test_skip_counts(tmp_path):
    entries = _make_entries(skipped=True)
    spotify_dir = _write_spotify_file(tmp_path, entries)

    profile = build_profile(
        spotify_dir=str(spotify_dir),
        out_path=str(tmp_path / "profile.json"),
        tag_db={},
    )
    assert profile["track_skip_counts"].get("artist a - track 1", 0) == 1


def test_year_scores_from_timestamp(tmp_path):
    entries = [dict(_make_entries()[0], ts="2019-06-01T10:00:00Z")]
    spotify_dir = _write_spotify_file(tmp_path, entries)

    profile = build_profile(
        spotify_dir=str(spotify_dir),
        out_path=str(tmp_path / "profile.json"),
        tag_db={},
    )
    assert "2019" in profile["year_scores"]


# ---------------------------------------------------------------------------
# Missing Spotify data → returns empty dict (not raises)
# ---------------------------------------------------------------------------

def test_build_profile_missing_dir(tmp_path):
    missing = tmp_path / "nonexistent_spotify_dir"
    profile = build_profile(
        spotify_dir=str(missing),
        out_path=str(tmp_path / "profile.json"),
        tag_db={},
    )
    assert profile == {}


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------

def test_load_profile_returns_empty_dict_if_missing(tmp_path):
    profile = load_profile(str(tmp_path / "missing_profile.json"))
    assert profile == {}


def test_load_profile_reads_saved_profile(tmp_path):
    data = {"artist_scores": {"X": 1}, "genre_scores": {"rock": 5}}
    p = tmp_path / "profile.json"
    p.write_text(json.dumps(data))
    loaded = load_profile(str(p))
    assert loaded["artist_scores"] == {"X": 1}
