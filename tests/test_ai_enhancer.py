"""Tests for playlistgen.ai_enhancer — AI playlist naming and enrichment."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# _summarise_cluster (pure function)
# ---------------------------------------------------------------------------


class TestSummariseCluster:
    def test_basic_summary(self):
        from playlistgen.ai_enhancer import _summarise_cluster

        df = pd.DataFrame({
            "Name": ["A", "B", "C"],
            "Artist": ["Art1", "Art1", "Art2"],
            "Genre": ["Rock", "Rock", "Jazz"],
            "Mood": ["Happy", "Happy", "Sad"],
            "Year": [2020, 2021, 2022],
        })
        summary = _summarise_cluster(df)
        assert "Tracks: 3" in summary
        assert "Mood:" in summary
        assert "Genre:" in summary
        assert "Era:" in summary
        assert "Top artists:" in summary

    def test_era_range(self):
        from playlistgen.ai_enhancer import _summarise_cluster

        df = pd.DataFrame({
            "Name": ["A", "B"],
            "Artist": ["X", "Y"],
            "Year": [1990, 2010],
        })
        summary = _summarise_cluster(df)
        assert "1990" in summary
        assert "2010" in summary

    def test_missing_columns_shows_unknown(self):
        from playlistgen.ai_enhancer import _summarise_cluster

        df = pd.DataFrame({"Name": ["A"], "Artist": ["X"]})
        summary = _summarise_cluster(df)
        assert "unknown" in summary

    def test_all_nan_year_shows_unknown(self):
        from playlistgen.ai_enhancer import _summarise_cluster

        df = pd.DataFrame({
            "Name": ["A"],
            "Artist": ["X"],
            "Year": [None],
        })
        summary = _summarise_cluster(df)
        assert "Era: unknown" in summary


# ---------------------------------------------------------------------------
# _call_claude
# ---------------------------------------------------------------------------


class TestCallClaude:
    def test_successful_response(self):
        from playlistgen.ai_enhancer import _call_claude

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"name": "Midnight Jazz", "cohesion": 8}')]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        name, cohesion = _call_claude("summary", "sk-test", "claude-3", mock_client)
        assert name == "Midnight Jazz"
        assert cohesion == 8

    def test_markdown_fenced_response(self):
        from playlistgen.ai_enhancer import _call_claude

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='```json\n{"name": "Chill Vibes", "cohesion": 6}\n```')]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        name, cohesion = _call_claude("summary", "sk-test", "claude-3", mock_client)
        assert name == "Chill Vibes"
        assert cohesion == 6

    def test_api_error_returns_empty(self):
        from playlistgen.ai_enhancer import _call_claude

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("rate limited")

        name, cohesion = _call_claude("summary", "sk-test", "claude-3", mock_client)
        assert name == ""
        assert cohesion == 0

    def test_invalid_json_returns_empty(self):
        from playlistgen.ai_enhancer import _call_claude

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="not json")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        name, cohesion = _call_claude("summary", "sk-test", "claude-3", mock_client)
        assert name == ""
        assert cohesion == 0


# ---------------------------------------------------------------------------
# enhance_playlists
# ---------------------------------------------------------------------------


class TestEnhancePlaylists:
    def _make_cluster(self, label="Cluster 1"):
        df = pd.DataFrame({
            "Name": ["A", "B"],
            "Artist": ["X", "Y"],
            "Genre": ["Rock", "Rock"],
            "Mood": ["Happy", "Happy"],
        })
        return [(label, df)]

    @patch("playlistgen.ai_enhancer._call_llm")
    def test_replaces_label_on_success(self, mock_llm):
        from playlistgen.ai_enhancer import enhance_playlists

        mock_llm.return_value = ("Summer Anthems", 9)
        labelled = self._make_cluster()
        result = enhance_playlists(labelled, api_key="sk-test", cfg={})
        assert result[0][0] == "Summer Anthems"

    @patch("playlistgen.ai_enhancer._call_llm")
    def test_keeps_original_on_empty_name(self, mock_llm):
        from playlistgen.ai_enhancer import enhance_playlists

        mock_llm.return_value = ("", 0)
        labelled = self._make_cluster("Original Name")
        result = enhance_playlists(labelled, api_key="sk-test", cfg={})
        assert result[0][0] == "Original Name"

    @patch("playlistgen.ai_enhancer._call_llm")
    def test_keeps_original_on_exception(self, mock_llm):
        from playlistgen.ai_enhancer import enhance_playlists

        mock_llm.side_effect = Exception("API down")
        labelled = self._make_cluster("Fallback")
        result = enhance_playlists(labelled, api_key="sk-test", cfg={})
        assert result[0][0] == "Fallback"

    @patch("playlistgen.ai_enhancer._call_llm")
    def test_preserves_dataframe(self, mock_llm):
        from playlistgen.ai_enhancer import enhance_playlists

        mock_llm.return_value = ("New Name", 7)
        labelled = self._make_cluster()
        result = enhance_playlists(labelled, api_key="sk-test", cfg={})
        assert len(result[0][1]) == 2  # DataFrame preserved

    @patch("playlistgen.ai_enhancer._call_llm")
    def test_multiple_clusters(self, mock_llm):
        from playlistgen.ai_enhancer import enhance_playlists

        mock_llm.side_effect = [("Mix A", 8), ("Mix B", 6)]
        df = pd.DataFrame({"Name": ["A"], "Artist": ["X"]})
        labelled = [("C1", df), ("C2", df)]
        result = enhance_playlists(labelled, api_key="sk-test", cfg={})
        assert result[0][0] == "Mix A"
        assert result[1][0] == "Mix B"
