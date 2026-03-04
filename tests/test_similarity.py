"""Tests for playlistgen.similarity — TF-IDF playlist vectorization."""

import numpy as np
import pandas as pd
import pytest

from playlistgen.similarity import (
    build_vectorizer,
    playlist_vector,
    score_playlists,
)


def _make_playlist(genres, moods=None):
    n = len(genres)
    return pd.DataFrame({
        "Genre": genres,
        "Mood": moods or [""] * n,
    })


class TestBuildVectorizer:
    def test_returns_fitted_vectorizer(self):
        playlists = [
            _make_playlist(["Rock", "Rock", "Pop"]),
            _make_playlist(["Jazz", "Jazz"]),
        ]
        vec = build_vectorizer(playlists)
        assert hasattr(vec, "vocabulary_")
        assert "rock" in vec.vocabulary_

    def test_single_playlist(self):
        vec = build_vectorizer([_make_playlist(["Electronic"])])
        assert "electronic" in vec.vocabulary_


class TestPlaylistVector:
    def test_returns_numpy_array(self):
        df = _make_playlist(["Rock", "Pop"])
        result = playlist_vector(df)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_with_fitted_vectorizer(self):
        playlists = [_make_playlist(["Rock", "Pop"]), _make_playlist(["Jazz"])]
        vec = build_vectorizer(playlists)
        result = playlist_vector(playlists[0], vec)
        assert result.shape[0] == len(vec.vocabulary_)

    def test_empty_text_returns_zeros(self):
        df = pd.DataFrame({"Genre": [""], "Mood": [""]})
        result = playlist_vector(df)
        assert np.allclose(result, 0)


class TestScorePlaylists:
    def test_similar_playlists_score_high(self):
        rock1 = _make_playlist(["Rock", "Rock", "Alternative"])
        rock2 = _make_playlist(["Rock", "Alternative", "Rock"])
        jazz = _make_playlist(["Jazz", "Jazz", "Swing"])

        vec = build_vectorizer([rock1, rock2, jazz])
        benchmark = [playlist_vector(rock1, vec)]
        scores = score_playlists([rock2, jazz], benchmark, vec)
        assert scores[0] > scores[1]  # rock2 more similar to rock1 than jazz

    def test_empty_candidates(self):
        scores = score_playlists([], [np.array([1, 0])], None)
        assert scores == []

    def test_empty_benchmarks(self):
        df = _make_playlist(["Pop"])
        scores = score_playlists([df], [], None)
        assert scores == [0.0]
