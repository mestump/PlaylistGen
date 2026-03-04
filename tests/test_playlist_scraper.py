"""Tests for playlistgen.playlist_scraper — Spotify/YouTube/Apple Music playlist fetching."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from playlistgen.playlist_scraper import (
    _get_spotify_client,
    fetch_spotify_playlists,
    fetch_youtube_playlists,
    fetch_apple_music_playlists,
)


class TestGetSpotifyClient:
    @patch("playlistgen.playlist_scraper.spotipy", None)
    def test_returns_none_without_spotipy(self):
        assert _get_spotify_client() is None

    @patch("playlistgen.playlist_scraper.spotipy")
    def test_returns_none_without_credentials(self, mock_spotipy):
        with patch.dict("os.environ", {}, clear=True):
            assert _get_spotify_client(client_id=None, client_secret=None) is None

    @patch("playlistgen.playlist_scraper.SpotifyClientCredentials")
    @patch("playlistgen.playlist_scraper.spotipy")
    def test_returns_client_with_credentials(self, mock_spotipy, mock_auth):
        client = _get_spotify_client(client_id="id", client_secret="secret")
        mock_spotipy.Spotify.assert_called_once()
        assert client is not None


class TestFetchSpotifyPlaylists:
    @patch("playlistgen.playlist_scraper._get_spotify_client")
    def test_returns_empty_without_client(self, mock_client):
        mock_client.return_value = None
        result = fetch_spotify_playlists("rock")
        assert result == []

    @patch("playlistgen.playlist_scraper._get_spotify_client")
    def test_returns_dataframes(self, mock_client):
        sp = MagicMock()
        mock_client.return_value = sp
        sp.search.return_value = {
            "playlists": {
                "items": [{"id": "abc123"}]
            }
        }
        sp.playlist_items.return_value = {
            "items": [
                {
                    "track": {
                        "name": "Bohemian Rhapsody",
                        "artists": [{"name": "Queen"}],
                        "album": {"name": "A Night at the Opera", "release_date": "1975-10-31"},
                    }
                },
                {"track": None},  # Should be skipped
            ]
        }
        result = fetch_spotify_playlists("rock", limit=1, client_id="id", client_secret="secret")
        assert len(result) == 1
        df = result[0]
        assert len(df) == 1
        assert df.iloc[0]["Title"] == "Bohemian Rhapsody"
        assert df.iloc[0]["Artist"] == "Queen"
        assert df.iloc[0]["Year"] == "1975"


class TestPlaceholders:
    def test_youtube_returns_empty(self):
        assert fetch_youtube_playlists() == []

    def test_apple_music_returns_empty(self):
        assert fetch_apple_music_playlists() == []
