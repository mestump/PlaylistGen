"""Tests for playlistgen.metadata — audio tag extraction and DataFrame enrichment."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from playlistgen.metadata import _strip_file_url, read_audio_tags, enrich_dataframe


# ---------------------------------------------------------------------------
# _strip_file_url
# ---------------------------------------------------------------------------


class TestStripFileUrl:
    def test_localhost_url(self):
        url = "file://localhost/Users/me/Music/song.mp3"
        assert _strip_file_url(url) == "/Users/me/Music/song.mp3"

    def test_file_url_without_localhost(self):
        url = "file:///Users/me/Music/song.mp3"
        assert _strip_file_url(url) == "/Users/me/Music/song.mp3"

    def test_percent_encoded_spaces(self):
        url = "file://localhost/Users/me/My%20Music/song.mp3"
        assert _strip_file_url(url) == "/Users/me/My Music/song.mp3"

    def test_plain_path_unchanged(self):
        path = "/Users/me/Music/song.mp3"
        assert _strip_file_url(path) == path

    def test_windows_style_path(self):
        path = "C:\\Users\\me\\Music\\song.mp3"
        assert _strip_file_url(path) == path

    def test_percent_encoded_special_chars(self):
        url = "file://localhost/Music/%C3%A9t%C3%A9.mp3"
        result = _strip_file_url(url)
        assert result == "/Music/été.mp3"


# ---------------------------------------------------------------------------
# read_audio_tags
# ---------------------------------------------------------------------------


class TestReadAudioTags:
    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", False)
    def test_returns_empty_when_mutagen_unavailable(self):
        result = read_audio_tags("/fake/path.mp3")
        assert result == {"year": None, "bpm": None, "genre": None, "duration_sec": None, "album": None}

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    def test_returns_empty_for_missing_file(self, mock_path):
        mock_path.return_value.exists.return_value = False
        result = read_audio_tags("/no/such/file.mp3")
        assert all(v is None for v in result.values())

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_reads_all_tags(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        tags_dict = {"date": ["1993"], "bpm": ["92"], "genre": ["Alternative Rock"], "album": ["Pablo Honey"]}
        audio = MagicMock()
        audio.info.length = 245.6
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        audio.tags.__bool__ = lambda self: True
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/creep.mp3")
        assert result["year"] == 1993
        assert result["bpm"] == 92
        assert result["genre"] == "Alternative Rock"
        assert result["album"] == "Pablo Honey"
        assert result["duration_sec"] == 245

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_mutafile_returns_none(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        mock_muta.return_value = None
        result = read_audio_tags("/Music/broken.mp3")
        assert all(v is None for v in result.values())

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_no_tags_returns_duration_only(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 180.0
        audio.tags = None
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/notags.mp3")
        assert result["duration_sec"] == 180
        assert result["year"] is None

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_year_from_date_field(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 200
        tags_dict = {"date": ["2003-06-15"]}
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/song.mp3")
        assert result["year"] == 2003

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_bpm_out_of_range_excluded(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 100
        tags_dict = {"bpm": ["10"]}  # Below 40 threshold
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/slow.mp3")
        assert result["bpm"] is None

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_numeric_genre_code_excluded(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 100
        tags_dict = {"genre": ["(17)"]}
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/id3genre.mp3")
        assert result["genre"] is None

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_exception_returns_empty(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        mock_muta.side_effect = Exception("corrupt file")
        result = read_audio_tags("/Music/corrupt.mp3")
        assert all(v is None for v in result.values())

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_year_invalid_string_skipped(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 100
        tags_dict = {"date": ["abcd"]}
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/badyear.mp3")
        assert result["year"] is None

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.Path")
    @patch("playlistgen.metadata.MutaFile")
    def test_bpm_with_comma_decimal(self, mock_muta, mock_path):
        mock_path.return_value.exists.return_value = True
        audio = MagicMock()
        audio.info.length = 100
        tags_dict = {"bpm": ["128,5"]}
        audio.tags = MagicMock()
        audio.tags.get = lambda key, default=None: tags_dict.get(key, default)
        mock_muta.return_value = audio

        result = read_audio_tags("/Music/euro.mp3")
        assert result["bpm"] == 128


# ---------------------------------------------------------------------------
# enrich_dataframe
# ---------------------------------------------------------------------------


class TestEnrichDataframe:
    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame({"Location": ["/a.mp3"], "Year": [None]})
        result = enrich_dataframe(df, enabled=False)
        assert result is df  # Same object, no copy

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", False)
    def test_no_mutagen_returns_unchanged(self):
        df = pd.DataFrame({"Location": ["/a.mp3"], "Year": [None]})
        result = enrich_dataframe(df, enabled=True)
        assert result is df

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.read_audio_tags")
    def test_fills_missing_values(self, mock_tags):
        mock_tags.return_value = {
            "year": 2005, "bpm": 120, "genre": "Rock",
            "duration_sec": 210, "album": "X&Y",
        }
        df = pd.DataFrame({
            "Location": ["/song.mp3"],
            "Artist": ["Coldplay"],
            "Name": ["Fix You"],
            "Genre": [None],
        })
        result = enrich_dataframe(df)
        assert result["Year"].iloc[0] == 2005
        assert result["BPM"].iloc[0] == 120
        assert result["Duration"].iloc[0] == 210
        assert result["Album"].iloc[0] == "X&Y"

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.read_audio_tags")
    def test_preserves_existing_values(self, mock_tags):
        mock_tags.return_value = {
            "year": 9999, "bpm": 999, "genre": "Wrong",
            "duration_sec": 999, "album": "Wrong",
        }
        df = pd.DataFrame({
            "Location": ["/song.mp3"],
            "Year": [2005],
            "BPM": [120],
            "Genre": ["Rock"],
            "Duration": [210],
            "Album": ["X&Y"],
        })
        result = enrich_dataframe(df)
        assert result["Year"].iloc[0] == 2005
        assert result["BPM"].iloc[0] == 120
        assert result["Album"].iloc[0] == "X&Y"

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.read_audio_tags")
    def test_skips_empty_location(self, mock_tags):
        df = pd.DataFrame({
            "Location": [None, "", "nan"],
            "Year": [None, None, None],
        })
        result = enrich_dataframe(df)
        mock_tags.assert_not_called()

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.read_audio_tags")
    def test_coerces_numeric_columns(self, mock_tags):
        mock_tags.return_value = {
            "year": 2020, "bpm": 140, "genre": None,
            "duration_sec": 300, "album": None,
        }
        df = pd.DataFrame({
            "Location": ["/a.mp3"],
            "Year": ["not_a_number"],
            "BPM": [None],
            "Duration": [None],
        })
        result = enrich_dataframe(df)
        # "not_a_number" stays because it's not _is_missing, but gets coerced to NaN
        assert pd.isna(result["Year"].iloc[0])
        assert result["BPM"].iloc[0] == 140

    @patch("playlistgen.metadata.MUTAGEN_AVAILABLE", True)
    @patch("playlistgen.metadata.read_audio_tags")
    def test_handles_multiple_rows(self, mock_tags):
        mock_tags.side_effect = [
            {"year": 2001, "bpm": 90, "genre": "Pop", "duration_sec": 200, "album": "A"},
            {"year": None, "bpm": None, "genre": None, "duration_sec": None, "album": None},
        ]
        df = pd.DataFrame({
            "Location": ["/a.mp3", "/b.mp3"],
            "Genre": [None, None],
        })
        result = enrich_dataframe(df)
        assert result["Year"].iloc[0] == 2001
        assert pd.isna(result["Year"].iloc[1])
