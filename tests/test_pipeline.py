"""Tests for playlistgen.pipeline — orchestration logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from playlistgen.pipeline import ensure_itunes_json, ensure_tag_cache, run_pipeline


# ---------------------------------------------------------------------------
# ensure_itunes_json
# ---------------------------------------------------------------------------


class TestEnsureItunesJson:
    def test_returns_json_path_when_exists_and_current(self, tmp_path):
        json_file = tmp_path / "library.json"
        xml_file = tmp_path / "library.xml"
        xml_file.write_text("<xml/>")
        json_file.write_text("{}")
        json_file.touch()

        cfg = {"ITUNES_JSON": str(json_file), "ITUNES_XML": str(xml_file)}
        result = ensure_itunes_json(cfg)
        assert result == json_file

    def test_converts_when_json_missing(self, tmp_path):
        json_file = tmp_path / "library.json"
        xml_file = tmp_path / "library.xml"
        xml_file.write_text("<xml/>")

        cfg = {"ITUNES_JSON": str(json_file), "ITUNES_XML": str(xml_file)}
        with patch("playlistgen.pipeline.convert_itunes_xml") as mock_convert:
            ensure_itunes_json(cfg)
        mock_convert.assert_called_once_with(str(xml_file), str(json_file))

    def test_converts_when_xml_newer(self, tmp_path):
        import time

        json_file = tmp_path / "library.json"
        xml_file = tmp_path / "library.xml"
        json_file.write_text("{}")
        time.sleep(0.05)
        xml_file.write_text("<xml/>")

        cfg = {"ITUNES_JSON": str(json_file), "ITUNES_XML": str(xml_file)}
        with patch("playlistgen.pipeline.convert_itunes_xml") as mock_convert:
            ensure_itunes_json(cfg)
        mock_convert.assert_called_once()


# ---------------------------------------------------------------------------
# ensure_tag_cache
# ---------------------------------------------------------------------------


class TestEnsureTagCache:
    def test_skips_when_no_api_key(self, tmp_path):
        cfg = {}
        with patch("playlistgen.pipeline.generate_tag_mood_cache") as mock_gen:
            ensure_tag_cache(cfg, tmp_path / "lib.json")
        mock_gen.assert_not_called()

    def test_calls_generate_when_api_key_set(self, tmp_path):
        cfg = {"LASTFM_API_KEY": "test-key-123"}
        json_path = tmp_path / "lib.json"
        with patch("playlistgen.pipeline.generate_tag_mood_cache") as mock_gen:
            ensure_tag_cache(cfg, json_path)
        mock_gen.assert_called_once()
        args = mock_gen.call_args
        assert args.kwargs.get("itunes_json_path") == str(json_path)


# ---------------------------------------------------------------------------
# run_pipeline — library loading
# ---------------------------------------------------------------------------


class TestRunPipelineLibraryLoading:
    def test_returns_empty_when_library_dir_empty(self, tmp_path):
        empty_dir = tmp_path / "music"
        empty_dir.mkdir()
        cfg = {
            "ITUNES_JSON": str(tmp_path / "lib.json"),
            "MUTAGEN_ENABLED": False,
        }
        result = run_pipeline(cfg, library_dir=str(empty_dir))
        assert result == []

    def test_returns_empty_when_itunes_json_empty(self, tmp_path):
        json_file = tmp_path / "lib.json"
        json_file.write_text("[]")

        cfg = {
            "ITUNES_JSON": str(json_file),
            "ITUNES_XML": str(tmp_path / "nonexistent.xml"),
        }
        with patch("playlistgen.pipeline.convert_itunes_xml"):
            with patch(
                "playlistgen.pipeline.load_itunes_json",
                return_value=pd.DataFrame(),
            ):
                result = run_pipeline(cfg)
        assert result == []


# ---------------------------------------------------------------------------
# run_pipeline — genre / mood filtering
# ---------------------------------------------------------------------------

def _make_scored_df():
    return pd.DataFrame(
        {
            "Name": ["Track A", "Track B", "Track C", "Track D"],
            "Artist": ["Art1", "Art2", "Art3", "Art4"],
            "Genre": ["Rock", "Jazz", "Rock", "Pop"],
            "Mood": ["Happy", "Sad", "Energetic", "Happy"],
            "Location": ["/a.mp3", "/b.mp3", "/c.mp3", "/d.mp3"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )


class TestRunPipelineFiltering:
    """Test the genre/mood single-playlist path (Stage 6b)."""

    @patch("playlistgen.pipeline.build_playlists")
    @patch("playlistgen.pipeline.score_tracks")
    @patch("playlistgen.pipeline.build_tag_counts", return_value={})
    @patch("playlistgen.pipeline.load_tag_mood_db", return_value={})
    @patch("playlistgen.pipeline.ensure_tag_cache")
    @patch("playlistgen.pipeline.load_itunes_json")
    @patch("playlistgen.pipeline.ensure_itunes_json")
    def test_genre_filter_passes_filtered_df(
        self,
        mock_ensure_json,
        mock_load_json,
        mock_ensure_tags,
        mock_load_tag_db,
        mock_build_tags,
        mock_score,
        mock_build_playlists,
        tmp_path,
    ):
        scored = _make_scored_df()
        mock_ensure_json.return_value = tmp_path / "lib.json"
        mock_load_json.return_value = scored
        mock_score.return_value = scored
        mock_build_playlists.return_value = [("Rock Vibes", scored)]

        cfg = {
            "ITUNES_JSON": str(tmp_path / "lib.json"),
            "LIBROSA_ENABLED": False,
            "AI_BATCH_ENRICH": False,
        }
        run_pipeline(cfg, genre="rock")

        call_args = mock_build_playlists.call_args
        filtered_clusters = call_args[0][0]
        assert len(filtered_clusters) == 1
        cluster_df = filtered_clusters[0]
        assert all(cluster_df["Genre"].str.lower() == "rock")

    @patch("playlistgen.pipeline.build_playlists")
    @patch("playlistgen.pipeline.score_tracks")
    @patch("playlistgen.pipeline.build_tag_counts", return_value={})
    @patch("playlistgen.pipeline.load_tag_mood_db", return_value={})
    @patch("playlistgen.pipeline.ensure_tag_cache")
    @patch("playlistgen.pipeline.load_itunes_json")
    @patch("playlistgen.pipeline.ensure_itunes_json")
    def test_mood_filter_passes_filtered_df(
        self,
        mock_ensure_json,
        mock_load_json,
        mock_ensure_tags,
        mock_load_tag_db,
        mock_build_tags,
        mock_score,
        mock_build_playlists,
        tmp_path,
    ):
        scored = _make_scored_df()
        mock_ensure_json.return_value = tmp_path / "lib.json"
        mock_load_json.return_value = scored
        mock_score.return_value = scored
        mock_build_playlists.return_value = [("Happy Vibes", scored)]

        cfg = {
            "ITUNES_JSON": str(tmp_path / "lib.json"),
            "LIBROSA_ENABLED": False,
            "AI_BATCH_ENRICH": False,
        }
        run_pipeline(cfg, mood="happy")

        call_args = mock_build_playlists.call_args
        filtered_clusters = call_args[0][0]
        assert len(filtered_clusters) == 1
        cluster_df = filtered_clusters[0]
        assert all(cluster_df["Mood"].str.lower() == "happy")

    @patch("playlistgen.pipeline.score_tracks")
    @patch("playlistgen.pipeline.build_tag_counts", return_value={})
    @patch("playlistgen.pipeline.load_tag_mood_db", return_value={})
    @patch("playlistgen.pipeline.ensure_tag_cache")
    @patch("playlistgen.pipeline.load_itunes_json")
    @patch("playlistgen.pipeline.ensure_itunes_json")
    def test_no_match_returns_empty(
        self,
        mock_ensure_json,
        mock_load_json,
        mock_ensure_tags,
        mock_load_tag_db,
        mock_build_tags,
        mock_score,
        tmp_path,
    ):
        scored = _make_scored_df()
        mock_ensure_json.return_value = tmp_path / "lib.json"
        mock_load_json.return_value = scored
        mock_score.return_value = scored

        cfg = {
            "ITUNES_JSON": str(tmp_path / "lib.json"),
            "LIBROSA_ENABLED": False,
            "AI_BATCH_ENRICH": False,
        }
        result = run_pipeline(cfg, genre="metal")
        assert result == []
