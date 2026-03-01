"""
Track clustering for PlaylistGen.

Groups library tracks into themed playlists using one of three strategies
(tried in order):

  1. Mood-based   — One cluster per canonical mood (CLUSTER_BY_MOOD=true).
  2. Year-based   — Clusters by year range (YEAR_MIX_ENABLED=true).
                    Uses the 'Year' column directly — no more path-based extraction.
  3. TF-IDF KMeans fallback — Clusters by text features (Genre + Artist + Mood).
"""

import logging

import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# Human-readable adjective for each canonical mood
MOOD_ADJECTIVES = {
    "Happy": "Joyful",
    "Sad": "Melancholic",
    "Angry": "Fiery",
    "Chill": "Chill",
    "Energetic": "Energetic",
    "Romantic": "Romantic",
    "Epic": "Epic",
    "Dreamy": "Dreamy",
    "Groovy": "Groovy",
    "Nostalgic": "Nostalgic",
    "Unknown": "",
}


def humanize_label(mood: str = None, genre: str = None) -> str:
    """Return a natural playlist label from mood and/or genre strings."""
    mood_adj = MOOD_ADJECTIVES.get(mood or "", mood) if mood else None
    if mood_adj and genre and genre.lower() not in ("unknown", ""):
        return f"{mood_adj} {genre}"
    if genre and genre.lower() not in ("unknown", ""):
        return genre
    if mood_adj:
        return mood_adj
    return "Mix"


def name_cluster(df: pd.DataFrame = None, i: int = None) -> str:
    """Generate a descriptive label for a cluster from its dominant mood and genre."""
    mood, genre = None, None
    if df is not None:
        if "Mood" in df.columns and df["Mood"].notnull().any():
            candidates = df["Mood"][df["Mood"] != "Unknown"]
            if not candidates.empty:
                mood = candidates.mode().iloc[0]
        if "Genre" in df.columns and df["Genre"].notnull().any():
            candidates = df["Genre"][df["Genre"].str.lower() != ""]
            if not candidates.empty:
                genre = candidates.mode().iloc[0]
    if mood or genre:
        return humanize_label(mood, genre)
    return f"Cluster {(i or 0) + 1}"


def cluster_tracks(
    df: pd.DataFrame,
    n_clusters: int = 6,
    use_hdbscan: bool = False,
    cluster_by_year: bool = False,
    year_range: int = 0,
    cluster_by_mood: bool = False,
    min_tracks_per_year: int = 25,
) -> list:
    """
    Cluster tracks into themed playlists.

    Strategy priority:
      1. Mood-based (if cluster_by_mood=True and Mood column is populated)
      2. Year-based (if cluster_by_year=True and Year column is populated)
      3. TF-IDF KMeans / HDBSCAN fallback

    Returns a list of DataFrames, one per cluster.
    """

    # ------------------------------------------------------------------
    # Strategy 1: Mood-based
    # ------------------------------------------------------------------
    if cluster_by_mood:
        if "Mood" not in df.columns or df["Mood"].isnull().all():
            logging.warning(
                "CLUSTER_BY_MOOD enabled but no Mood data — falling back."
            )
        else:
            mood_groups = []
            for mood, group in df.groupby("Mood"):
                if mood and mood != "Unknown" and not group.empty:
                    mood_groups.append(group.copy())
            if mood_groups:
                logging.info(
                    "Mood-based clusters: %d groups — sizes: %s",
                    len(mood_groups),
                    [len(g) for g in mood_groups],
                )
                return mood_groups
            logging.warning("No non-Unknown mood clusters found — falling back.")

    # ------------------------------------------------------------------
    # Strategy 2: Year-based
    # Uses df['Year'] directly (populated by iTunes XML or mutagen enrichment)
    # ------------------------------------------------------------------
    if cluster_by_year:
        year_col = df.get("Year") if "Year" in df.columns else None
        if year_col is not None and year_col.notna().any():
            df = df.copy()
            # Coerce to numeric; out-of-range already set to NaN by itunes.py
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            year_groups = []

            if year_range and year_range > 0:
                valid = df.dropna(subset=["Year"])
                if not valid.empty:
                    min_year = int(valid["Year"].min())
                    max_year = int(valid["Year"].max())
                    start = min_year
                    while start <= max_year:
                        end = start + year_range
                        group = df[
                            df["Year"].notna()
                            & (df["Year"] >= start)
                            & (df["Year"] < end)
                        ]
                        if len(group) >= min_tracks_per_year:
                            year_groups.append(group.copy())
                        start += year_range
            else:
                for year, group in df.dropna(subset=["Year"]).groupby("Year"):
                    if len(group) >= min_tracks_per_year:
                        year_groups.append(group.copy())

            if year_groups:
                logging.info(
                    "Year-based clusters: %d groups — sizes: %s",
                    len(year_groups),
                    [len(g) for g in year_groups],
                )
                return year_groups
            logging.warning(
                "No year-based clusters met the minimum track threshold (%d) — falling back.",
                min_tracks_per_year,
            )
        else:
            logging.warning(
                "YEAR_MIX_ENABLED but no valid Year data in library — falling back."
            )

    # ------------------------------------------------------------------
    # Strategy 3: TF-IDF KMeans / HDBSCAN
    # ------------------------------------------------------------------
    if not SKLEARN_AVAILABLE:
        logging.warning(
            "sklearn not available — splitting library into %d equal parts.", n_clusters
        )
        df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)
        parts = [
            df_sorted.iloc[i::n_clusters].reset_index(drop=True)
            for i in range(n_clusters)
        ]
        return [p for p in parts if not p.empty]

    df = df.copy()
    mood_str = df["Mood"].fillna("") if "Mood" in df.columns else ""
    genre_str = df["Genre"].fillna("") if "Genre" in df.columns else ""
    df["_text"] = (
        df[["Genre", "Artist"]].fillna("").agg(" ".join, axis=1)
        + " " + df.get("Mood", pd.Series("", index=df.index)).fillna("")
    )

    vectorizer = TfidfVectorizer(max_features=1000, min_df=1)
    X = vectorizer.fit_transform(df["_text"])

    if use_hdbscan and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
    else:
        n = min(n_clusters, len(df))
        clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X)

    df["_cluster"] = labels
    clusters = [
        grp.drop(columns=["_text", "_cluster"]).copy()
        for _, grp in df.groupby("_cluster")
        if len(grp) > 0
    ]
    logging.info(
        "KMeans clusters: %d groups — sizes: %s",
        len(clusters),
        [len(c) for c in clusters],
    )
    return clusters
