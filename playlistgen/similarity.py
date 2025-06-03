from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_vectorizer(playlists: List[pd.DataFrame]) -> TfidfVectorizer:
    texts = [
        " ".join((p.get("Genre", "").fillna("") + " " + p.get("Mood", "").fillna(" ")).tolist())
        for p in playlists
    ]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer


def playlist_vector(df: pd.DataFrame, vectorizer: Optional[TfidfVectorizer] = None) -> np.ndarray:
    text = " ".join(
        (df.get("Genre", "").fillna("") + " " + df.get("Mood", "").fillna("")).tolist()
    )
    if not text:
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
