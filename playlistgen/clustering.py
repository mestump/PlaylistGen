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
}


def humanize_label(mood: str | None, genre: str | None) -> str:
    """Return a more natural playlist label from mood/genre."""
    mood_adj = MOOD_ADJECTIVES.get(mood, mood) if mood else None
    if mood_adj and genre:
        return f"{mood_adj} {genre}"
    if genre:
        return genre
    if mood_adj:
        return mood_adj
    return "Mix"


def name_cluster(df=None, i=None):
    """Generate a descriptive name for a cluster based on its top mood and genre."""
    mood = None
    genre = None
    if df is not None:
        if "Mood" in df.columns and df["Mood"].notnull().any():
            mood = df["Mood"].mode().iloc[0]
        if "Genre" in df.columns and df["Genre"].notnull().any():
            genre = df["Genre"].mode().iloc[0]
    if mood or genre:
        return humanize_label(mood, genre)
    if i is not None:
        return f"Cluster {i + 1}"
    return "Cluster"


def cluster_tracks(
    df: pd.DataFrame,
    n_clusters: int = 6,
    use_hdbscan: bool = False,
    cluster_by_year: bool = False,
    year_range: int = 0,
    cluster_by_mood: bool = False,
    min_tracks_per_year: int = 25,
):
    """
    Cluster tracks into themes using one of the following strategies (in order):
      - Mood-based grouping (if enabled)
      - Year-range grouping (if enabled)
      - Text/mood feature clustering (fallback)

    Returns a list of DataFrames, one per cluster.
    """

    def extract_year(row):
        loc = row.get("Location", "")
        for part in str(loc).split("/"):
            if part.isdigit() and 1900 < int(part) < 2100:
                return int(part)
        return row.get("Year")

    if cluster_by_mood:
        if "Mood" not in df.columns or df["Mood"].isnull().all():
            logging.warning(
                "CLUSTER_BY_MOOD enabled but no Mood column — falling back to other clustering"
            )
        else:
            mood_groups = []
            for mood, group in df.groupby("Mood"):
                if mood and not group.empty:
                    mood_groups.append(group)
            if mood_groups:
                logging.info(
                    f"Generated {len(mood_groups)} mood-based clusters: {[len(g) for g in mood_groups]}"
                )
                return mood_groups
            logging.warning(
                "No mood-based clusters found — falling back to other clustering"
            )

    if cluster_by_year:
        # Only attempt year-based clustering if at least one track yields a valid year
        year_vals = (
            df["Year"]
            if "Year" in df.columns and df["Year"].notnull().any()
            else df.apply(extract_year, axis=1)
        )
        if year_vals.notnull().any():
            df["Year"] = year_vals
            year_groups = []
            if year_range and year_range > 0:
                min_year = int(df["Year"].min())
                max_year = int(df["Year"].max())
                start = min_year
                while start <= max_year:
                    end = start + year_range
                    group = df[(df["Year"] >= start) & (df["Year"] < end)]
                    if len(group) >= min_tracks_per_year:
                        year_groups.append(group)
                    start += year_range
            else:
                for year, group in df.groupby("Year"):
                    if len(group) >= min_tracks_per_year:
                        year_groups.append(group)
            if year_groups:
                logging.info(
                    f"Clustered into {len(year_groups)} year-based clusters: {[len(g) for g in year_groups]}"
                )
                return year_groups
            logging.warning(
                "No year-based clusters found — falling back to text/mood clustering"
            )
        else:
            logging.warning(
                "YEAR_MIX_ENABLED but no valid Year data found — falling back to text/mood clustering"
            )

    if not SKLEARN_AVAILABLE:
        logging.warning(
            f"sklearn not available; using simple split clustering into {n_clusters} groups"
        )
        df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)
        clusters = [
            df_sorted.iloc[i::n_clusters].reset_index(drop=True)
            for i in range(n_clusters)
        ]
        return [c for c in clusters if not c.empty]

    df = df.copy()
    if "Mood" in df.columns:
        df["mood_str"] = df["Mood"].fillna("")
    else:
        df["mood_str"] = ""
    df["text_features"] = (
        df[["Genre", "Name", "Artist"]].fillna("").agg(" ".join, axis=1)
        + " "
        + df["mood_str"]
    )

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["text_features"])

    if use_hdbscan and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(X)

    df["Cluster"] = labels
    clusters = [group for _, group in df.groupby("Cluster") if len(group) > 0]
    logging.info(f"Generated {len(clusters)} fallback clusters")
    logging.info(f"Created {len(clusters)} clusters: {[len(c) for c in clusters]}")
    return clusters
