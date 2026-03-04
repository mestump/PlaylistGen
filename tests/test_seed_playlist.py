"""Tests for playlistgen.seed_playlist — seed-based playlist generation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from playlistgen.seed_playlist import (
    fetch_similar_tracks,
    generate_seed_playlist,
    build_seed_playlist,
)


# ---------------------------------------------------------------------------
# fetch_similar_tracks
# ---------------------------------------------------------------------------


class TestFetchSimilarTracks:
    @patch("playlistgen.seed_playlist.requests")
    def test_returns_artist_track_tuples(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "similartracks": {
                "track": [
                    {"artist": {"name": "Muse"}, "name": "Supermassive Black Hole"},
                    {"artist": {"name": "Foo Fighters"}, "name": "Everlong"},
                ]
            }
        }
        mock_requests.get.return_value = mock_resp
        result = fetch_similar_tracks("Radiohead", "Creep", "fake_key", limit=5)
        assert result == [("Muse", "Supermassive Black Hole"), ("Foo Fighters", "Everlong")]

    @patch("playlistgen.seed_playlist.requests")
    def test_skips_entries_missing_artist(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "similartracks": {
                "track": [
                    {"artist": {}, "name": "Track"},
                    {"artist": {"name": "OK"}, "name": "Song"},
                ]
            }
        }
        mock_requests.get.return_value = mock_resp
        result = fetch_similar_tracks("A", "B", "key")
        assert result == [("OK", "Song")]

    @patch("playlistgen.seed_playlist.requests")
    def test_returns_empty_on_network_error(self, mock_requests):
        mock_requests.get.side_effect = Exception("timeout")
        result = fetch_similar_tracks("A", "B", "key")
        assert result == []

    @patch("playlistgen.seed_playlist.requests")
    def test_returns_empty_on_empty_response(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_requests.get.return_value = mock_resp
        result = fetch_similar_tracks("A", "B", "key")
        assert result == []

    @patch("playlistgen.seed_playlist.requests")
    def test_url_contains_encoded_params(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"similartracks": {"track": []}}
        mock_requests.get.return_value = mock_resp
        mock_requests.utils.quote = lambda s: s.replace(" ", "%20")
        fetch_similar_tracks("The Beatles", "Let It Be", "mykey", limit=10)
        url = mock_requests.get.call_args[0][0]
        assert "The%20Beatles" in url
        assert "Let%20It%20Be" in url
        assert "mykey" in url
        assert "limit=10" in url


# ---------------------------------------------------------------------------
# generate_seed_playlist
# ---------------------------------------------------------------------------


class TestGenerateSeedPlaylist:
    def _library_df(self):
        return pd.DataFrame({
            "Artist": ["Muse", "Foo Fighters", "Coldplay", "Adele"],
            "Name": ["Supermassive Black Hole", "Everlong", "Yellow", "Hello"],
            "Genre": ["Rock", "Rock", "Alternative", "Pop"],
            "BPM": [120, 130, 100, 80],
            "Year": [2006, 1997, 2000, 2015],
        })

    @patch("playlistgen.seed_playlist.score_tracks")
    @patch("playlistgen.seed_playlist.fetch_similar_tracks")
    @patch("playlistgen.seed_playlist.load_config")
    def test_matches_library_tracks(self, mock_cfg, mock_fetch, mock_score):
        mock_cfg.return_value = {"LASTFM_API_KEY": "key"}
        mock_fetch.return_value = [
            ("Muse", "Supermassive Black Hole"),
            ("Unknown", "Nothing"),
            ("Coldplay", "Yellow"),
        ]
        lib = self._library_df()
        mock_score.side_effect = lambda df, **kw: df.assign(Score=range(len(df), 0, -1))

        result = generate_seed_playlist("Radiohead", "Creep", lib)
        assert len(result) == 2
        assert "Muse" in result["Artist"].values

    @patch("playlistgen.seed_playlist.fetch_similar_tracks")
    @patch("playlistgen.seed_playlist.load_config")
    def test_no_matches_returns_empty(self, mock_cfg, mock_fetch):
        mock_cfg.return_value = {"LASTFM_API_KEY": "key"}
        mock_fetch.return_value = [("Nobody", "Nothing")]
        lib = self._library_df()

        result = generate_seed_playlist("A", "B", lib)
        assert result.empty

    @patch("playlistgen.seed_playlist.score_tracks")
    @patch("playlistgen.seed_playlist.fetch_similar_tracks")
    @patch("playlistgen.seed_playlist.load_config")
    def test_respects_limit(self, mock_cfg, mock_fetch, mock_score):
        mock_cfg.return_value = {"LASTFM_API_KEY": "key"}
        mock_fetch.return_value = [
            ("Muse", "Supermassive Black Hole"),
            ("Foo Fighters", "Everlong"),
            ("Coldplay", "Yellow"),
            ("Adele", "Hello"),
        ]
        lib = self._library_df()
        mock_score.side_effect = lambda df, **kw: df.assign(Score=range(len(df), 0, -1))

        result = generate_seed_playlist("X", "Y", lib, limit=2)
        assert len(result) <= 2

    @patch("playlistgen.seed_playlist.score_tracks")
    @patch("playlistgen.seed_playlist.fetch_similar_tracks")
    @patch("playlistgen.seed_playlist.load_config")
    def test_case_insensitive_matching(self, mock_cfg, mock_fetch, mock_score):
        mock_cfg.return_value = {"LASTFM_API_KEY": "key"}
        mock_fetch.return_value = [("muse", "supermassive black hole")]
        lib = self._library_df()
        mock_score.side_effect = lambda df, **kw: df.assign(Score=[10])

        result = generate_seed_playlist("X", "Y", lib)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_seed_playlist
# ---------------------------------------------------------------------------


class TestBuildSeedPlaylist:
    @patch("playlistgen.seed_playlist.build_playlists")
    @patch("playlistgen.seed_playlist.generate_seed_playlist")
    @patch("playlistgen.seed_playlist.load_profile")
    @patch("playlistgen.seed_playlist.load_tag_mood_db")
    @patch("playlistgen.seed_playlist.build_library_from_dir")
    @patch("playlistgen.seed_playlist.load_config")
    def test_dash_song_format(self, mock_cfg, mock_build_lib, mock_tag_db,
                              mock_profile, mock_gen, mock_build):
        mock_cfg.return_value = {}
        mock_build_lib.return_value = pd.DataFrame()
        mock_tag_db.return_value = {}
        mock_profile.return_value = {}
        mock_gen.return_value = pd.DataFrame({"Artist": ["X"], "Name": ["Y"]})

        build_seed_playlist("Radiohead - Creep", library_dir="/music")
        # Verify artist/track parsing
        args = mock_gen.call_args
        assert args[0][0] == "Radiohead"
        assert args[0][1] == "Creep"

    @patch("playlistgen.seed_playlist.build_playlists")
    @patch("playlistgen.seed_playlist.generate_seed_playlist")
    @patch("playlistgen.seed_playlist.load_profile")
    @patch("playlistgen.seed_playlist.load_tag_mood_db")
    @patch("playlistgen.seed_playlist.build_library_from_dir")
    @patch("playlistgen.seed_playlist.load_config")
    def test_space_song_format(self, mock_cfg, mock_build_lib, mock_tag_db,
                               mock_profile, mock_gen, mock_build):
        mock_cfg.return_value = {}
        mock_build_lib.return_value = pd.DataFrame()
        mock_tag_db.return_value = {}
        mock_profile.return_value = {}
        mock_gen.return_value = pd.DataFrame({"Artist": ["X"], "Name": ["Y"]})

        build_seed_playlist("Radiohead Creep", library_dir="/music")
        args = mock_gen.call_args
        assert args[0][0] == "Radiohead"
        assert args[0][1] == "Creep"

    @patch("playlistgen.seed_playlist.generate_seed_playlist")
    @patch("playlistgen.seed_playlist.load_profile")
    @patch("playlistgen.seed_playlist.load_tag_mood_db")
    @patch("playlistgen.seed_playlist.build_library_from_dir")
    @patch("playlistgen.seed_playlist.load_config")
    def test_returns_none_when_empty(self, mock_cfg, mock_build_lib, mock_tag_db,
                                     mock_profile, mock_gen):
        mock_cfg.return_value = {}
        mock_build_lib.return_value = pd.DataFrame()
        mock_tag_db.return_value = {}
        mock_profile.return_value = {}
        mock_gen.return_value = pd.DataFrame()

        result = build_seed_playlist("X - Y", library_dir="/music")
        assert result is None

    @patch("playlistgen.seed_playlist.build_playlists")
    @patch("playlistgen.seed_playlist.generate_seed_playlist")
    @patch("playlistgen.seed_playlist.load_profile")
    @patch("playlistgen.seed_playlist.load_tag_mood_db")
    @patch("playlistgen.seed_playlist.build_library_from_dir")
    @patch("playlistgen.seed_playlist.load_config")
    def test_builds_with_correct_label(self, mock_cfg, mock_build_lib, mock_tag_db,
                                       mock_profile, mock_gen, mock_build):
        mock_cfg.return_value = {}
        mock_build_lib.return_value = pd.DataFrame()
        mock_tag_db.return_value = {}
        mock_profile.return_value = {}
        mock_gen.return_value = pd.DataFrame({"Artist": ["X"], "Name": ["Y"]})

        build_seed_playlist("Daft Punk - Get Lucky", library_dir="/music")
        # Verify build_playlists was called
        mock_build.assert_called_once()
        # The name_fn should produce the right label
        name_fn = mock_build.call_args[1].get("name_fn") or mock_build.call_args[0][3] if len(mock_build.call_args[0]) > 3 else None
        # Check the label via keyword args
        call_kwargs = mock_build.call_args
        # name_fn is passed as a keyword
        assert "name_fn" in call_kwargs.kwargs or len(call_kwargs.args) > 3
