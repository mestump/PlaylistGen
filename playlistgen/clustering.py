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

def name_cluster(df=None, i=None):
    return f"Cluster {i + 1}" if i is not None else "Cluster"

def cluster_tracks(
    df: pd.DataFrame,
    n_clusters: int = 6,
    use_hdbscan: bool = False,
    cluster_by_year: bool = False,
    min_tracks_per_year: int = 25
):
    """
    Clusters tracks using either:
      - Year-based grouping
      - Mood/text feature clustering (fallback)

    Returns: List of DataFrames (one per cluster/year group)
    """

    def extract_year(row):
        loc = row.get('Location', '')
        for part in str(loc).split('/'):
            if part.isdigit() and 1900 < int(part) < 2100:
                return int(part)
        return row.get('Year')

    if cluster_by_year:
        if 'Year' not in df.columns or df['Year'].isnull().all():
            df['Year'] = df.apply(extract_year, axis=1)

        year_groups = []
        for year, group in df.groupby('Year'):
            if len(group) >= min_tracks_per_year:
                year_groups.append(group)

        if year_groups:
            logging.info(f"Clustered into {len(year_groups)} year groups")
            logging.info(f'Created {len(year_groups)} year-based clusters: {[len(g) for g in year_groups]}')
            return year_groups
        else:
            logging.warning("No year-based clusters found — falling back to mood/text clustering")

    if not SKLEARN_AVAILABLE:
        logging.warning(f"sklearn not available; using simple split clustering into {n_clusters} groups")
        df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)
        clusters = [df_sorted.iloc[i::n_clusters].reset_index(drop=True) for i in range(n_clusters)]
        return [c for c in clusters if not c.empty]

    df = df.copy()
    if 'Mood' in df.columns:
        df['mood_str'] = df['Mood'].fillna('')
    else:
        df['mood_str'] = ''
    df['text_features'] = df[['Genre', 'Name', 'Artist']].fillna('').agg(' '.join, axis=1) + ' ' + df['mood_str']

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text_features'])

    if use_hdbscan and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
    else:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(X)

    df['Cluster'] = labels
    clusters = [group for _, group in df.groupby('Cluster') if len(group) > 0]
    logging.info(f"Generated {len(clusters)} fallback clusters")
    logging.info(f'Created {len(clusters)} clusters: {[len(c) for c in clusters]}')
    return clusters
