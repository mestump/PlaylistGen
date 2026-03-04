"""Tests for playlistgen.pattern_analyzer — playlist clustering & vectorization."""

import numpy as np
import pandas as pd
import pytest

from playlistgen.pattern_analyzer import (
    _playlist_text,
    vectorize_playlists,
    vectorize_playlist,
    analyze_playlists,
)


def _make_playlist(genres, moods=None):
    n = len(genres)
    return pd.DataFrame({
        "Genre": genres,
        "Mood": moods or [""] * n,
    })


class TestPlaylistText:
    def test_combines_genre_and_mood(self):
        df = pd.DataFrame({"Genre": ["Rock"], "Mood": ["Happy"]})
        result = _playlist_text(df)
        assert result == ["Rock Happy"]

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"Name": ["Song"]})
        result = _playlist_text(df)
        assert len(result) == 1

    def test_handles_nan(self):
        df = pd.DataFrame({"Genre": [None], "Mood": ["Sad"]})
        result = _playlist_text(df)
        assert "Sad" in result[0]


class TestVectorizePlaylists:
    def test_returns_matrix_and_vectorizer(self):
        playlists = [_make_playlist(["Rock"]), _make_playlist(["Jazz"])]
        X, vec = vectorize_playlists(playlists)
        assert X.shape[0] == 2
        assert hasattr(vec, "vocabulary_")

    def test_with_existing_vectorizer(self):
        playlists = [_make_playlist(["Rock", "Pop"]), _make_playlist(["Jazz"])]
        _, vec = vectorize_playlists(playlists)
        X2, vec2 = vectorize_playlists([_make_playlist(["Rock"])], vectorizer=vec)
        assert X2.shape[1] == len(vec.vocabulary_)


class TestVectorizePlaylist:
    def test_empty_df_returns_zeros(self):
        result = vectorize_playlist(pd.DataFrame())
        assert np.allclose(result, 0)

    def test_single_playlist(self):
        df = _make_playlist(["Electronic", "Electronic"])
        result = vectorize_playlist(df)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1


class TestAnalyzePlaylists:
    def test_empty_list_returns_empty_dict(self):
        assert analyze_playlists([]) == {}

    def test_returns_model_and_labels(self):
        playlists = [
            _make_playlist(["Rock", "Rock"]),
            _make_playlist(["Jazz", "Jazz"]),
            _make_playlist(["Pop", "Pop"]),
        ]
        result = analyze_playlists(playlists, n_clusters=2)
        assert "model" in result
        assert "labels" in result
        assert len(result["labels"]) == 3

    def test_clusters_capped_at_playlist_count(self):
        playlists = [_make_playlist(["Rock"]), _make_playlist(["Jazz"])]
        result = analyze_playlists(playlists, n_clusters=10)
        assert result["model"].n_clusters == 2
