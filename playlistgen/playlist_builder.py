import logging
import pandas as pd
from pathlib import Path
from .config import load_config
from .utils import sanitize_label


def cap_artist(df: pd.DataFrame, max_per_artist: int) -> pd.DataFrame:
    return df.groupby("Artist", group_keys=False).head(max_per_artist)


def fill_short_pool(
    df: pd.DataFrame, global_df: pd.DataFrame, target_len: int, max_per_artist: int
) -> pd.DataFrame:
    """
    If df is shorter than target_len, fill remaining slots with random tracks from global_df,
    without exceeding max_per_artist for any artist and avoiding duplicates.
    """
    need = target_len - len(df)
    if need <= 0:
        return df
    counts = df["Artist"].value_counts().to_dict()
    leftovers = global_df.drop(df.index).copy()
    leftovers = leftovers.drop_duplicates(subset=["Artist", "Name"])
    leftovers = leftovers[
        leftovers["Artist"].map(lambda a: counts.get(a, 0) < max_per_artist)
    ]
    if leftovers.empty:
        return df
    fill_n = min(need, len(leftovers))
    filler = leftovers.sample(n=fill_n)
    return pd.concat([df, filler], ignore_index=True)


def reorder_playlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fairly interleave tracks from different artists in round-robin fashion to maximize variety.
    """
    from itertools import zip_longest

    records = df.to_dict("records")
    artist_to_tracks = {}
    for rec in records:
        artist_to_tracks.setdefault(rec["Artist"], []).append(rec)
    artists = sorted(
        artist_to_tracks.keys(), key=lambda a: len(artist_to_tracks[a]), reverse=True
    )
    track_lists = [artist_to_tracks[a] for a in artists]
    interleaved = []
    for group in zip_longest(*track_lists):
        for rec in group:
            if rec:
                interleaved.append(rec)
    return pd.DataFrame(interleaved)


def save_m3u(df: pd.DataFrame, label: str, out_dir: str = None):
    logging.info(f"Writing playlist: {label} with {len(df)} tracks")
    top_moods = df["Mood"].value_counts().head(3) if "Mood" in df.columns else []
    top_genres = df["Genre"].value_counts().head(3) if "Genre" in df.columns else []
    top_artists = df["Artist"].value_counts().head(3) if "Artist" in df.columns else []

    import pandas as pd

    def safe_index(obj):
        return list(obj.index) if isinstance(obj, pd.Series) else []

    logging.info(
        f"Top moods: {safe_index(top_moods)} | "
        f"Top genres: {safe_index(top_genres)} | "
        f"Top artists: {safe_index(top_artists)}"
    )

    cfg = load_config()
    if out_dir is None:
        out_dir = cfg.get("OUTPUT_DIR", "./mixes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = sanitize_label(label)
    if "Date" in df.columns:
        date = df["Date"].iloc[0]
        fname = f"{safe} [{date}].m3u"
    else:
        fname = f"{safe}.m3u"
    path = out_dir / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        # Skip tracks with missing or 'nan' file paths
        valid_df = df[
            df["Location"].notna() & (df["Location"].astype(str).str.lower() != "nan")
        ]
        for _, r in valid_df.iterrows():
            f.write(f"#EXTINF:-1,{r['Artist']} - {r['Name']}\n")
            f.write(f"{r['Location']}\n")
    logging.info(f"Saved {path}")


def build_playlists(
    clusters: list,
    global_df: pd.DataFrame,
    tracks_per_mix: int = None,
    max_per_artist: int = None,
    save: bool = True,
    name_fn=None,
    num_playlists: int = None,
):
    """
    Build playlists for each cluster, capping number of playlists to config or argument.
    """
    cfg = load_config()
    if tracks_per_mix is None:
        tracks_per_mix = int(cfg.get("TRACKS_PER_MIX", 50))
    if max_per_artist is None:
        max_per_artist = int(cfg.get("MAX_PER_ARTIST", 5))
    if num_playlists is None:
        num_playlists = int(cfg.get("NUM_PLAYLISTS", len(clusters)))

    # Optionally shuffle clusters for variety (handled upstream in CLI)
    playlists = []
    for i, cluster in enumerate(clusters[:num_playlists]):
        label = name_fn(cluster, i) if name_fn else f"Cluster {i+1}"
        playlist = cap_artist(
            cluster.sort_values("Score", ascending=False), max_per_artist
        )
        if len(playlist) < tracks_per_mix:
            playlist = fill_short_pool(
                playlist, global_df, tracks_per_mix, max_per_artist
            )
        else:
            playlist = playlist.head(tracks_per_mix)
        playlist = playlist.drop_duplicates(subset=["Artist", "Name"]).reset_index(
            drop=True
        )
        playlist = reorder_playlist(playlist)
        if save:
            save_m3u(playlist, label)
        playlists.append((label, playlist))
    return playlists
