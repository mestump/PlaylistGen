from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _get_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Safely get a text column, returning empty strings if column is missing."""
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df))


def build_vectorizer(playlists: List[pd.DataFrame]) -> TfidfVectorizer:
    texts = [
        " ".join((_get_text_series(p, "Genre") + " " + _get_text_series(p, "Mood")).tolist())
        for p in playlists
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer


def playlist_vector(df: pd.DataFrame, vectorizer: Optional[TfidfVectorizer] = None) -> np.ndarray:
    text = " ".join(
        (_get_text_series(df, "Genre") + " " + _get_text_series(df, "Mood")).tolist()
    )
    if not text.strip():
        return np.zeros(1)
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text])
    else:
        X = vectorizer.transform([text])
    return np.asarray(X.todense()).ravel()


def score_playlists(
    candidates: List[pd.DataFrame],
    benchmark_vecs: List[np.ndarray],
    vectorizer: Optional[TfidfVectorizer] = None,
) -> List[float]:
    if vectorizer is None and benchmark_vecs:
        # assume benchmark vectors were generated with their own vectorizer
        vectorizer = TfidfVectorizer()
    candidate_vecs = [playlist_vector(c, vectorizer) for c in candidates]
    if not candidate_vecs or not benchmark_vecs:
        return [0.0 for _ in candidate_vecs]
    X = np.vstack(candidate_vecs)
    Y = np.vstack(benchmark_vecs)
    sims = cosine_similarity(X, Y).max(axis=1)
    return sims.tolist()
