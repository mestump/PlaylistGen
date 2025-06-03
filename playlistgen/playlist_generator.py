import logging
from typing import Dict, List, Optional

import pandas as pd

from .scoring import score_tracks
from .clustering import cluster_tracks


def generate_candidates(
    itunes_df: pd.DataFrame,
    profile: Optional[Dict] = None,
    preferences: Optional[Dict] = None,
) -> List[pd.DataFrame]:
    """Generate playlist candidates from the user's library."""
    logging.info("Generating candidate playlists")
    scored = score_tracks(itunes_df, config=profile or {})

    prefs = preferences or {}
    if "genres" in prefs:
        genres = [g.lower() for g in prefs["genres"]]
        scored = scored[scored["Genre"].str.lower().isin(genres)]
    if "moods" in prefs:
        scored = scored[scored["Mood"].isin(prefs["moods"])]

    n_clusters = prefs.get("clusters", 5)
    clusters = cluster_tracks(scored, n_clusters=n_clusters)
    return clusters
