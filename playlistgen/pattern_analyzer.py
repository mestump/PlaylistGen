import logging
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _playlist_text(df: pd.DataFrame) -> List[str]:
    return (
        df.get("Genre", "").fillna("").astype(str)
        + " "
        + df.get("Mood", "").fillna("").astype(str)
    ).tolist()


def vectorize_playlists(
    playlists: List[pd.DataFrame],
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Vectorize multiple playlists with a shared TF-IDF vectorizer."""
    texts = [" ".join(_playlist_text(p)) for p in playlists]
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return np.asarray(X.todense()), vectorizer


def vectorize_playlist(df: pd.DataFrame, vectorizer: Optional[TfidfVectorizer] = None) -> np.ndarray:
    """Vectorize a single playlist using the provided TF-IDF vectorizer."""
    if df.empty:
        return np.zeros(1)
    text = " ".join(_playlist_text(df))
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text])
    else:
        X = vectorizer.transform([text])
    return np.asarray(X.todense()).ravel()


def analyze_playlists(playlists: List[pd.DataFrame], n_clusters: int = 5) -> Dict:
    """Cluster playlists to find common patterns.

    Returns a dictionary with the trained model, cluster labels, and vectors.
    """
    if not playlists:
        logging.warning("No playlists provided for analysis")
        return {}

    X, vectorizer = vectorize_playlists(playlists)
    k = min(n_clusters, len(playlists))
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return {"model": model, "labels": labels, "vectors": X, "vectorizer": vectorizer}
