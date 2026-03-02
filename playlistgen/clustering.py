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
    from sklearn.preprocessing import MinMaxScaler

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
            candidates = df["Mood"][df["Mood"].notna() & (df["Mood"] != "Unknown")]
            m = candidates.mode()
            if not m.empty:
                mood = m.iloc[0]
        if "Genre" in df.columns and df["Genre"].notnull().any():
            candidates = df["Genre"][df["Genre"].notna() & (df["Genre"].str.lower() != "")]
            m = candidates.mode()
            if not m.empty:
                genre = m.iloc[0]
    if mood or genre:
        return humanize_label(mood, genre)
    return f"Cluster {(i or 0) + 1}"


def cluster_by_audio_features(
    df: pd.DataFrame,
    n_clusters: int = 6,
) -> list:
    """
    Cluster tracks using numeric audio features: BPM, Energy, SpectralBrightness, ZCR.

    Features are normalised to [0, 1] before KMeans clustering. Produces
    playlists differentiated by sonic texture rather than genre text labels.

    Falls back to [] if sklearn is unavailable or energy coverage is < 30%.
    """
    if not SKLEARN_AVAILABLE:
        return []

    feature_cols = [
        c for c in ["BPM", "Energy", "SpectralBrightness", "ZCR"]
        if c in df.columns
    ]
    if not feature_cols:
        return []

    df = df.copy()
    feat_df = df[feature_cols].copy()
    for col in feature_cols:
        feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce")
        median_val = feat_df[col].median()
        feat_df[col] = feat_df[col].fillna(
            median_val if pd.notnull(median_val) else 0
        )

    energy_col = "Energy" if "Energy" in feat_df.columns else feature_cols[0]
    coverage = df[energy_col].notna().mean() if energy_col in df.columns else 0
    if coverage < 0.3:
        logging.warning(
            "Audio feature coverage %.0f%% < 30%% — skipping audio clustering.",
            coverage * 100,
        )
        return []

    scaler = MinMaxScaler()
    X = scaler.fit_transform(feat_df.values)

    n = min(n_clusters, len(df))
    clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = clusterer.fit_predict(X)

    df["_cluster"] = labels
    clusters = [
        grp.drop(columns=["_cluster"]).copy()
        for _, grp in df.groupby("_cluster")
        if len(grp) > 0
    ]
    logging.info(
        "Audio feature clusters: %d groups — sizes: %s",
        len(clusters),
        [len(c) for c in clusters],
    )
    return clusters


def _cluster_hybrid_impl(
    df: pd.DataFrame,
    n_audio_subclusters: int = 2,
) -> list:
    """
    Internal: group by mood first, then sub-cluster each mood by audio features.

    Produces focused playlists like "Chill – Acoustic", "Chill – Electronic",
    "Sad – Quiet", "Sad – Driving" etc.
    """
    if "Mood" not in df.columns or df["Mood"].isnull().all():
        return []

    result = []
    for mood, mood_group in df.groupby("Mood"):
        if not mood or mood == "Unknown" or mood_group.empty:
            continue
        if len(mood_group) < 10:
            result.append(mood_group.copy())
            continue
        sub_clusters = cluster_by_audio_features(
            mood_group, n_clusters=n_audio_subclusters
        )
        if sub_clusters:
            result.extend(sub_clusters)
        else:
            result.append(mood_group.copy())

    return result


def cluster_tracks(
    df: pd.DataFrame,
    n_clusters: int = 6,
    use_hdbscan: bool = False,
    cluster_by_year: bool = False,
    year_range: int = 0,
    cluster_by_mood: bool = False,
    cluster_hybrid_mode: bool = False,
    min_tracks_per_year: int = 25,
    strategy: str = "auto",
) -> list:
    """
    Cluster tracks into themed playlists.

    When strategy="auto" (default), the best available strategy is chosen:
      1. Audio feature KMeans  (if Energy column has >30% coverage)
      2. Mood-based grouping   (if Mood column has >50% non-Unknown coverage)
      3. TF-IDF KMeans         (text-based fallback)

    Explicit strategies: "audio", "mood", "year", "tfidf".
    cluster_hybrid_mode=True: group by mood then sub-cluster by audio features.

    Returns a list of DataFrames, one per cluster.
    """

    # ------------------------------------------------------------------
    # Hybrid mode: mood groups → audio sub-clusters
    # ------------------------------------------------------------------
    if cluster_hybrid_mode:
        result = _cluster_hybrid_impl(
            df, n_audio_subclusters=max(1, n_clusters // 5)
        )
        if result:
            logging.info(
                "Hybrid clusters: %d groups — sizes: %s",
                len(result),
                [len(g) for g in result],
            )
            return result
        logging.warning("Hybrid clustering produced no groups — falling back.")

    # ------------------------------------------------------------------
    # Auto strategy selection
    # ------------------------------------------------------------------
    if strategy == "auto":
        energy_coverage = 0.0
        mood_coverage = 0.0
        if "Energy" in df.columns:
            energy_coverage = df["Energy"].notna().mean()
        if "Mood" in df.columns:
            mood_coverage = (
                df["Mood"].notna() & (df["Mood"] != "Unknown")
            ).mean()

        if energy_coverage > 0.3:
            strategy = "audio"
        elif mood_coverage > 0.5 or cluster_by_mood:
            strategy = "mood"
        elif cluster_by_year:
            strategy = "year"
        else:
            strategy = "tfidf"
        logging.info(
            "Auto strategy selected: '%s' "
            "(energy_cov=%.0f%%, mood_cov=%.0f%%)",
            strategy,
            energy_coverage * 100,
            mood_coverage * 100,
        )

    # ------------------------------------------------------------------
    # Strategy: audio features
    # ------------------------------------------------------------------
    if strategy == "audio":
        result = cluster_by_audio_features(df, n_clusters=n_clusters)
        if result:
            return result
        logging.warning(
            "Audio clustering failed — falling back to mood/tfidf."
        )
        # Determine next best fallback
        if "Mood" in df.columns and (df["Mood"] != "Unknown").mean() > 0.5:
            strategy = "mood"
        else:
            strategy = "tfidf"

    # ------------------------------------------------------------------
    # Strategy: mood-based
    # ------------------------------------------------------------------
    if strategy == "mood" or cluster_by_mood:
        if "Mood" not in df.columns or df["Mood"].isnull().all():
            logging.warning(
                "Mood strategy selected but no Mood data — falling back."
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
    # Strategy: year-based
    # ------------------------------------------------------------------
    if strategy == "year" or cluster_by_year:
        year_col = df.get("Year") if "Year" in df.columns else None
        if year_col is not None and year_col.notna().any():
            df = df.copy()
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
                "No year-based clusters met the minimum track threshold (%d)"
                " — falling back.",
                min_tracks_per_year,
            )
        else:
            logging.warning(
                "Year strategy but no valid Year data — falling back."
            )

    # ------------------------------------------------------------------
    # Strategy: TF-IDF KMeans / HDBSCAN (fallback)
    # ------------------------------------------------------------------
    if not SKLEARN_AVAILABLE:
        logging.warning(
            "sklearn not available — splitting library into %d equal parts.",
            n_clusters,
        )
        df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)
        parts = [
            df_sorted.iloc[i::n_clusters].reset_index(drop=True)
            for i in range(n_clusters)
        ]
        return [p for p in parts if not p.empty]

    df = df.copy()
    df["_text"] = (
        df[["Genre", "Artist"]].fillna("").agg(" ".join, axis=1)
        + " "
        + df.get("Mood", pd.Series("", index=df.index)).fillna("")
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
        "TF-IDF KMeans clusters: %d groups — sizes: %s",
        len(clusters),
        [len(c) for c in clusters],
    )
    return clusters
