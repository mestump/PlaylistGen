"""Tests for itunes.py — library loading, Year/BPM/Duration column handling."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from playlistgen.itunes import (
    load_itunes_json,
    build_library_from_dir,
    save_itunes_json,
    convert_itunes_xml,
    _decode_location,
)


# ---------------------------------------------------------------------------
# _decode_location
# ---------------------------------------------------------------------------

def test_decode_file_localhost():
    raw = "file://localhost/Users/alice/Music/track.mp3"
    assert _decode_location(raw) == "/Users/alice/Music/track.mp3"


def test_decode_file_url_encoded():
    raw = "file://localhost/Users/alice/Music/my%20song.mp3"
    assert _decode_location(raw) == "/Users/alice/Music/my song.mp3"


def test_decode_plain_path():
    raw = "/Users/alice/Music/track.mp3"
    assert _decode_location(raw) == raw


# ---------------------------------------------------------------------------
# load_itunes_json
# ---------------------------------------------------------------------------

def _write_json(tmp_path, tracks):
    p = tmp_path / "library.json"
    p.write_text(json.dumps({"tracks": tracks}))
    return str(p)


def test_load_itunes_json_basic(tmp_path):
    tracks = [
        {"Name": "Song A", "Artist": "Artist 1", "Genre": "Rock",
         "Location": "/music/a.mp3", "Play Count": 5, "Skip Count": 1,
         "Year": 2005, "Total Time": 210000},
    ]
    path = _write_json(tmp_path, tracks)
    df = load_itunes_json(path)
    assert len(df) == 1
    assert df.iloc[0]["Name"] == "Song A"
    assert df.iloc[0]["Year"] == 2005
    assert df.iloc[0]["Play Count"] == 5


def test_load_itunes_json_duration_ms_to_seconds(tmp_path):
    tracks = [{"Name": "S", "Artist": "A", "Total Time": 210000}]
    path = _write_json(tmp_path, tracks)
    df = load_itunes_json(path)
    # 210000 ms → 210 seconds
    assert df.iloc[0]["Duration"] == 210.0


def test_load_itunes_json_drops_missing_name_artist(tmp_path):
    tracks = [
        {"Name": "S", "Artist": "A"},
        {"Artist": "A"},          # no Name
        {"Name": "S"},             # no Artist
    ]
    path = _write_json(tmp_path, tracks)
    df = load_itunes_json(path)
    assert len(df) == 1


def test_load_itunes_json_decodes_location(tmp_path):
    tracks = [{"Name": "S", "Artist": "A",
               "Location": "file://localhost/Users/alice/Music/my%20song.mp3"}]
    path = _write_json(tmp_path, tracks)
    df = load_itunes_json(path)
    assert df.iloc[0]["Location"] == "/Users/alice/Music/my song.mp3"


def test_load_itunes_json_year_out_of_range_set_to_nan(tmp_path):
    tracks = [{"Name": "S", "Artist": "A", "Year": 1800}]
    path = _write_json(tmp_path, tracks)
    df = load_itunes_json(path)
    assert pd.isna(df.iloc[0]["Year"])


# ---------------------------------------------------------------------------
# build_library_from_dir
# ---------------------------------------------------------------------------

def test_build_library_from_dir_parses_filename(tmp_path):
    f = tmp_path / "Artist - Title.mp3"
    f.write_bytes(b"")  # empty file — no mutagen tags
    df = build_library_from_dir(str(tmp_path))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Artist"] == "Artist"
    assert row["Name"] == "Title"
    assert row["Location"] == str(f.resolve())


def test_build_library_from_dir_unknown_artist_no_dash(tmp_path):
    f = tmp_path / "JustTitle.mp3"
    f.write_bytes(b"")
    df = build_library_from_dir(str(tmp_path))
    assert df.iloc[0]["Artist"] == "Unknown"
    assert df.iloc[0]["Name"] == "JustTitle"


def test_build_library_from_dir_ignores_non_audio(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "Artist - Song.mp3").write_bytes(b"")
    df = build_library_from_dir(str(tmp_path))
    assert len(df) == 1


def test_build_library_from_dir_missing_dir():
    with pytest.raises(FileNotFoundError):
        build_library_from_dir("/this/does/not/exist")


# ---------------------------------------------------------------------------
# save_itunes_json round-trip
# ---------------------------------------------------------------------------

def test_save_and_reload(tmp_path):
    df = pd.DataFrame([{
        "Name": "Song", "Artist": "Artist", "Genre": "Rock",
        "Location": "/music/song.mp3", "Play Count": 3, "Skip Count": 0,
    }])
    out = tmp_path / "out.json"
    save_itunes_json(df, out)
    df2 = load_itunes_json(str(out))
    assert len(df2) == 1
    assert df2.iloc[0]["Name"] == "Song"
