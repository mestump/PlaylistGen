"""
Tests for playlistgen/spotify_export.py

All Spotify API calls are intercepted with unittest.mock so no real network
access or OAuth flow is required.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal spotipy stub that satisfies the module-level
# ``try: import spotipy`` block inside spotify_export.py.
# ---------------------------------------------------------------------------

def _make_spotipy_stub():
    """Return (spotipy_mod, SpotifyOAuth_cls, Spotify_cls) fakes."""
    spotipy_mod = ModuleType("spotipy")
    oauth2_mod = ModuleType("spotipy.oauth2")

    SpotifyOAuth = MagicMock(name="SpotifyOAuth")
    Spotify = MagicMock(name="Spotify")

    oauth2_mod.SpotifyOAuth = SpotifyOAuth
    spotipy_mod.oauth2 = oauth2_mod
    spotipy_mod.Spotify = Spotify

    return spotipy_mod, SpotifyOAuth, Spotify


def _inject_spotipy(monkeypatch):
    """
    Inject fake spotipy into sys.modules so that spotify_export.py sees it,
    and reload the module so SPOTIPY_AVAILABLE = True is set.
    Returns (spotipy_mod, SpotifyOAuth_cls, Spotify_cls).
    """
    spotipy_mod, SpotifyOAuth, Spotify = _make_spotipy_stub()
    monkeypatch.setitem(sys.modules, "spotipy", spotipy_mod)
    monkeypatch.setitem(sys.modules, "spotipy.oauth2", spotipy_mod.oauth2)

    # Force-reload the module under test so it picks up the stub.
    import importlib
    import playlistgen.spotify_export as _se
    importlib.reload(_se)
    monkeypatch.setattr(_se, "SPOTIPY_AVAILABLE", True)
    monkeypatch.setattr(_se, "spotipy", spotipy_mod)
    monkeypatch.setattr(_se, "SpotifyOAuth", SpotifyOAuth)

    return _se, spotipy_mod, SpotifyOAuth, Spotify


def _make_sp_client(Spotify, *, user_id="testuser", playlist_id="pl123", playlist_url="https://open.spotify.com/playlist/pl123"):
    """Return a configured mock Spotify client instance."""
    sp = MagicMock(name="sp_instance")
    Spotify.return_value = sp

    sp.current_user.return_value = {"id": user_id}
    sp.user_playlist_create.return_value = {
        "id": playlist_id,
        "external_urls": {"spotify": playlist_url},
    }
    sp.playlist_add_items.return_value = {}
    return sp


def _track_uri(n=1):
    return f"spotify:track:track{n:04d}"


def _search_result(uri):
    """Minimal Spotify search result payload containing one track."""
    return {"tracks": {"items": [{"uri": uri}]}}


def _empty_search_result():
    return {"tracks": {"items": []}}


# ---------------------------------------------------------------------------
# _search_track
# ---------------------------------------------------------------------------

class TestSearchTrack:
    def test_returns_uri_when_track_found(self, monkeypatch):
        se, *_, Spotify = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        expected_uri = _track_uri(1)
        sp.search.return_value = _search_result(expected_uri)

        uri = se._search_track(sp, "Radiohead", "Creep")

        sp.search.assert_called_once_with(
            q="artist:Radiohead track:Creep", type="track", limit=1
        )
        assert uri == expected_uri

    def test_returns_none_when_no_items(self, monkeypatch):
        se, *_ = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        sp.search.return_value = _empty_search_result()

        uri = se._search_track(sp, "Unknown Artist", "Unknown Song")

        assert uri is None

    def test_returns_none_on_api_exception(self, monkeypatch):
        se, *_ = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        sp.search.side_effect = Exception("Connection refused")

        uri = se._search_track(sp, "Artist", "Song")

        assert uri is None

    def test_search_query_format(self, monkeypatch):
        """Query string must follow 'artist:X track:Y' Spotify syntax."""
        se, *_ = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        sp.search.return_value = _empty_search_result()

        se._search_track(sp, "The Beatles", "Hey Jude")

        _, kwargs = sp.search.call_args
        assert kwargs["q"] == "artist:The Beatles track:Hey Jude"
        assert kwargs["type"] == "track"
        assert kwargs["limit"] == 1


# ---------------------------------------------------------------------------
# export_playlist_to_spotify — happy path
# ---------------------------------------------------------------------------

class TestExportPlaylistToSpotify:
    def _sample_df(self, n=3):
        return pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(n)],
                "Name": [f"Song{i}" for i in range(n)],
            }
        )

    def test_creates_playlist_and_adds_tracks(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify, playlist_url="https://open.spotify.com/playlist/abc")

        uri_a = _track_uri(1)
        uri_b = _track_uri(2)
        uri_c = _track_uri(3)
        sp.search.side_effect = [
            _search_result(uri_a),
            _search_result(uri_b),
            _search_result(uri_c),
        ]

        df = self._sample_df(3)
        cfg = {
            "SPOTIFY_CLIENT_ID": "cid",
            "SPOTIFY_CLIENT_SECRET": "secret",
            "SPOTIFY_REDIRECT_URI": "http://localhost:8888/callback",
        }

        result = se.export_playlist_to_spotify(df, "Test Playlist", cfg=cfg)

        assert result == "https://open.spotify.com/playlist/abc"
        sp.user_playlist_create.assert_called_once()
        create_kwargs = sp.user_playlist_create.call_args[1]
        assert create_kwargs["name"] == "Test Playlist"
        assert create_kwargs["public"] is True

        sp.playlist_add_items.assert_called_once_with(
            "pl123", [uri_a, uri_b, uri_c]
        )

    def test_returns_playlist_url(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        expected_url = "https://open.spotify.com/playlist/xyz789"
        sp = _make_sp_client(Spotify, playlist_url=expected_url)
        sp.search.return_value = _search_result(_track_uri())

        df = pd.DataFrame({"Artist": ["Artist"], "Name": ["Song"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        url = se.export_playlist_to_spotify(df, "My Mix", cfg=cfg)

        assert url == expected_url

    def test_private_playlist_creation(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)
        sp.search.return_value = _search_result(_track_uri())

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(df, "Private Mix", cfg=cfg, public=False)

        create_kwargs = sp.user_playlist_create.call_args[1]
        assert create_kwargs["public"] is False

    def test_custom_description_is_forwarded(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)
        sp.search.return_value = _search_result(_track_uri())

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(
            df, "My Mix", cfg=cfg, description="A custom description"
        )

        create_kwargs = sp.user_playlist_create.call_args[1]
        assert create_kwargs["description"] == "A custom description"

    def test_default_description_contains_track_count(self, monkeypatch):
        """When no description is supplied one is auto-generated with track count."""
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)
        sp.search.side_effect = [_search_result(_track_uri(i)) for i in range(2)]

        df = pd.DataFrame({"Artist": ["A1", "A2"], "Name": ["S1", "S2"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(df, "Mix", cfg=cfg, description="")

        create_kwargs = sp.user_playlist_create.call_args[1]
        assert "2" in create_kwargs["description"]


# ---------------------------------------------------------------------------
# export_playlist_to_spotify — no tracks found
# ---------------------------------------------------------------------------

class TestExportNoTracksFound:
    def test_returns_none_when_no_tracks_matched(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)
        sp.search.return_value = _empty_search_result()

        df = pd.DataFrame({"Artist": ["Ghost", "Phantom"], "Name": ["Mist", "Shadow"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        result = se.export_playlist_to_spotify(df, "Empty Mix", cfg=cfg)

        assert result is None
        sp.user_playlist_create.assert_not_called()

    def test_skips_rows_with_empty_artist_or_name(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)
        # Only the third row is valid
        sp.search.return_value = _search_result(_track_uri())

        df = pd.DataFrame(
            {
                "Artist": ["", "ValidArtist", ""],
                "Name": ["Song1", "ValidSong", ""],
            }
        )
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        result = se.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        # Only one valid row -> one search call
        assert sp.search.call_count == 1
        assert result is not None


# ---------------------------------------------------------------------------
# export_playlist_to_spotify — missing credentials
# ---------------------------------------------------------------------------

class TestExportMissingCredentials:
    def test_returns_none_when_no_client_id(self, monkeypatch):
        se, *_ = _inject_spotipy(monkeypatch)
        # No env vars set, no cfg keys
        monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
        monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {}  # no credentials

        result = se.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        assert result is None

    def test_returns_none_when_spotipy_unavailable(self, monkeypatch):
        """When spotipy is not installed the function returns None immediately."""
        import importlib
        import playlistgen.spotify_export as se_mod

        # Mark library as unavailable
        monkeypatch.setattr(se_mod, "SPOTIPY_AVAILABLE", False)

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        result = se_mod.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        assert result is None


# ---------------------------------------------------------------------------
# export_playlist_to_spotify — API error handling
# ---------------------------------------------------------------------------

class TestExportAPIErrors:
    def test_returns_none_when_current_user_fails(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        Spotify.return_value = sp
        sp.current_user.side_effect = Exception("401 Unauthorized")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        result = se.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        assert result is None

    def test_returns_none_when_playlist_create_fails(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = MagicMock()
        Spotify.return_value = sp
        sp.current_user.return_value = {"id": "user1"}
        sp.search.return_value = _search_result(_track_uri())
        sp.user_playlist_create.side_effect = Exception("403 Forbidden")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        result = se.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        assert result is None

    def test_continues_when_add_items_fails(self, monkeypatch):
        """
        If playlist_add_items raises, the function should log and not re-raise;
        the playlist URL is still returned because the playlist was created.
        """
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify, playlist_url="https://open.spotify.com/playlist/fail")
        sp.search.return_value = _search_result(_track_uri())
        sp.playlist_add_items.side_effect = Exception("500 Server Error")

        df = pd.DataFrame({"Artist": ["A"], "Name": ["S"]})
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        # Should not raise
        result = se.export_playlist_to_spotify(df, "Mix", cfg=cfg)

        # The URL comes from the playlist dict, not from add_items, so it is
        # returned even when adding tracks fails.
        assert result == "https://open.spotify.com/playlist/fail"


# ---------------------------------------------------------------------------
# Batching — more than 100 tracks
# ---------------------------------------------------------------------------

class TestBatching:
    def test_tracks_split_into_batches_of_100(self, monkeypatch):
        """150 found tracks must result in exactly two playlist_add_items calls."""
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)

        n = 150
        # Each search call returns a unique URI
        sp.search.side_effect = [_search_result(_track_uri(i)) for i in range(n)]

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(n)],
                "Name": [f"Song{i}" for i in range(n)],
            }
        )
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(df, "Big Mix", cfg=cfg)

        assert sp.playlist_add_items.call_count == 2
        first_batch = sp.playlist_add_items.call_args_list[0][0][1]
        second_batch = sp.playlist_add_items.call_args_list[1][0][1]
        assert len(first_batch) == 100
        assert len(second_batch) == 50

    def test_exactly_100_tracks_one_batch(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)

        n = 100
        sp.search.side_effect = [_search_result(_track_uri(i)) for i in range(n)]

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(n)],
                "Name": [f"Song{i}" for i in range(n)],
            }
        )
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(df, "Hundred Mix", cfg=cfg)

        assert sp.playlist_add_items.call_count == 1

    def test_201_tracks_three_batches(self, monkeypatch):
        se, spotipy_mod, SpotifyOAuth, Spotify = _inject_spotipy(monkeypatch)
        sp = _make_sp_client(Spotify)

        n = 201
        sp.search.side_effect = [_search_result(_track_uri(i)) for i in range(n)]

        df = pd.DataFrame(
            {
                "Artist": [f"Artist{i}" for i in range(n)],
                "Name": [f"Song{i}" for i in range(n)],
            }
        )
        cfg = {"SPOTIFY_CLIENT_ID": "cid", "SPOTIFY_CLIENT_SECRET": "sec"}

        se.export_playlist_to_spotify(df, "Huge Mix", cfg=cfg)

        assert sp.playlist_add_items.call_count == 3
        batch_sizes = [
            len(c[0][1]) for c in sp.playlist_add_items.call_args_list
        ]
        assert batch_sizes == [100, 100, 1]
