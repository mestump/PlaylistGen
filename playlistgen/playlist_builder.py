"""
Playlist assembly and M3U export for PlaylistGen.

Provides the full playlist-building pipeline:
  cap_artist()      — Limit tracks per artist for diversity.
  fill_short_pool() — Backfill short playlists from the global library.
  reorder_playlist()— Energy-arc + artist round-robin ordering.
  save_m3u()        — Write an extended M3U file (iTunes/Music.app compatible).
  build_playlists() — Orchestrate all of the above for a list of clusters.

M3U output:
  - Absolute file paths (iTunes/Music.app sync for iPod).
  - Real EXTINF duration (seconds) when the Duration column is available.
  - file:// URL decoding so iTunes-exported paths work on any system.
"""

import logging
from itertools import zip_longest
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from .config import load_config
from .utils import sanitize_label


# ---------------------------------------------------------------------------
# Artist diversity helpers
# ---------------------------------------------------------------------------


def cap_artist(df: pd.DataFrame, max_per_artist: int) -> pd.DataFrame:
    """Keep at most max_per_artist tracks per artist."""
    return df.groupby("Artist", group_keys=False).head(max_per_artist)


def fill_short_pool(
    df: pd.DataFrame,
    global_df: pd.DataFrame,
    target_len: int,
    max_per_artist: int,
) -> pd.DataFrame:
    """
    If df has fewer than target_len tracks, fill remaining slots with random
    tracks from global_df, respecting max_per_artist and avoiding duplicates.
    """
    need = target_len - len(df)
    if need <= 0:
        return df

    counts = df["Artist"].value_counts().to_dict()
    pool = (
        global_df.drop(index=df.index, errors="ignore")
        .drop_duplicates(subset=["Artist", "Name"])
        .copy()
    )
    pool = pool[pool["Artist"].map(lambda a: counts.get(a, 0) < max_per_artist)]

    if pool.empty:
        return df

    filler = pool.sample(n=min(need, len(pool)), random_state=42)
    return pd.concat([df, filler], ignore_index=True)


# ---------------------------------------------------------------------------
# Playlist ordering
# ---------------------------------------------------------------------------


def _round_robin_by_artist(df: pd.DataFrame) -> pd.DataFrame:
    """Interleave tracks from different artists so no two consecutive same-artist tracks."""
    records = df.to_dict("records")
    by_artist: dict = {}
    for rec in records:
        by_artist.setdefault(rec["Artist"], []).append(rec)

    # Artists with more tracks go first so they stay spread out
    artists = sorted(by_artist, key=lambda a: len(by_artist[a]), reverse=True)
    track_lists = [by_artist[a] for a in artists]

    interleaved = []
    for group in zip_longest(*track_lists):
        for rec in group:
            if rec is not None:
                interleaved.append(rec)
    return pd.DataFrame(interleaved)


def _energy_arc_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order tracks by energy arc: build-up → peak → cool-down, using BPM as
    the energy proxy.  Falls back to artist round-robin when fewer than 30%
    of tracks have BPM data.
    """
    if "BPM" not in df.columns:
        return _round_robin_by_artist(df)

    bpm_valid = df["BPM"].notna() & (df["BPM"] > 0)
    if bpm_valid.sum() < len(df) * 0.3:
        return _round_robin_by_artist(df)

    df = df.copy()
    df["_bpm_filled"] = df["BPM"].fillna(df["BPM"].median())

    # Divide into thirds: low / mid / high energy
    q33 = df["_bpm_filled"].quantile(0.33)
    q67 = df["_bpm_filled"].quantile(0.67)

    low = df[df["_bpm_filled"] <= q33].sort_values("Score", ascending=False)
    mid = df[(df["_bpm_filled"] > q33) & (df["_bpm_filled"] <= q67)].sort_values(
        "Score", ascending=False
    )
    high = df[df["_bpm_filled"] > q67].sort_values("Score", ascending=False)

    # Arc: mid → high → low  (start moderate, peak, wind down)
    n = len(df)
    third = max(n // 3, 1)
    ordered = pd.concat([
        mid.head(third),
        high.head(third),
        low.head(n - 2 * third),
    ]).drop_duplicates(subset=["Artist", "Name"])

    # Pad with any tracks missed due to rounding
    if len(ordered) < len(df):
        remaining = df.drop(index=ordered.index, errors="ignore")
        ordered = pd.concat([ordered, remaining]).drop_duplicates(subset=["Artist", "Name"])

    return _round_robin_by_artist(ordered.drop(columns=["_bpm_filled"]))


def reorder_playlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply energy-arc ordering if BPM data is available; otherwise fall back to
    artist round-robin interleaving.
    """
    return _energy_arc_order(df)


# ---------------------------------------------------------------------------
# M3U export
# ---------------------------------------------------------------------------


def _resolve_path(raw: str) -> str:
    """Decode file://localhost URLs to plain filesystem paths."""
    if raw.startswith("file://localhost"):
        return unquote(raw.replace("file://localhost", ""))
    if raw.startswith("file://"):
        return unquote(raw.replace("file://", ""))
    return raw


def save_m3u(df: pd.DataFrame, label: str, out_dir: str = None) -> Path:
    """
    Write an extended M3U playlist file.

    - Uses absolute file paths (iTunes/Music.app iPod sync compatible).
    - Writes real EXTINF duration when Duration column is available.
    - Decodes file:// URLs from iTunes exports.
    - Skips tracks with missing or invalid Location values.

    Returns the Path of the written file.
    """
    cfg = load_config()
    if out_dir is None:
        out_dir = cfg.get("OUTPUT_DIR", "./mixes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe = sanitize_label(label)
    out_path = out_dir / f"{safe}.m3u"

    valid = df[
        df["Location"].notna()
        & (df["Location"].astype(str).str.strip() != "")
        & (df["Location"].astype(str).str.lower() != "nan")
    ].copy()

    # Log playlist composition
    def _top(col, n=3):
        if col in df.columns and df[col].notna().any():
            return list(df[col].value_counts().head(n).index)
        return []

    logging.info(
        "Writing '%s': %d tracks | moods: %s | genres: %s | artists: %s",
        label,
        len(valid),
        _top("Mood"),
        _top("Genre"),
        _top("Artist"),
    )

    has_duration = "Duration" in valid.columns

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for _, row in valid.iterrows():
            loc = _resolve_path(str(row["Location"]))
            artist = str(row.get("Artist", ""))
            name = str(row.get("Name", ""))

            duration = -1
            if has_duration and pd.notna(row.get("Duration")):
                try:
                    duration = int(float(row["Duration"]))
                except (TypeError, ValueError):
                    duration = -1

            f.write(f"#EXTINF:{duration},{artist} - {name}\n")
            f.write(f"{loc}\n")

    logging.info("Saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_playlists(
    clusters: list,
    global_df: pd.DataFrame,
    tracks_per_mix: int = None,
    max_per_artist: int = None,
    save: bool = True,
    name_fn=None,
    num_playlists: int = None,
    out_dir: str = None,
) -> list:
    """
    Build and (optionally) save M3U playlists for each cluster.

    Args:
        clusters:       List of track DataFrames (one per cluster).
        global_df:      The full scored library (used to backfill short playlists).
        tracks_per_mix: Target playlist length (default from config).
        max_per_artist: Max tracks per artist (default from config).
        save:           Whether to write M3U files.
        name_fn:        Callable(df, index) → str for playlist label.
        num_playlists:  Cap on number of playlists built.
        out_dir:        Override output directory.

    Returns:
        List of (label, DataFrame) tuples.
    """
    cfg = load_config()
    if tracks_per_mix is None:
        tracks_per_mix = int(cfg.get("TRACKS_PER_MIX", 50))
    if max_per_artist is None:
        max_per_artist = int(cfg.get("MAX_PER_ARTIST", 4))
    if num_playlists is None:
        num_playlists = int(cfg.get("NUM_PLAYLISTS", len(clusters)))

    playlists = []
    for i, cluster in enumerate(clusters[:num_playlists]):
        label = name_fn(cluster, i) if name_fn else f"Cluster {i + 1}"

        # Sort by score, cap per-artist, fill to target length
        playlist = cap_artist(
            cluster.sort_values("Score", ascending=False), max_per_artist
        )
        if len(playlist) < tracks_per_mix:
            playlist = fill_short_pool(
                playlist, global_df, tracks_per_mix, max_per_artist
            )
        else:
            playlist = playlist.head(tracks_per_mix)

        playlist = (
            playlist.drop_duplicates(subset=["Artist", "Name"])
            .reset_index(drop=True)
        )
        playlist = reorder_playlist(playlist)

        if save:
            save_m3u(playlist, label, out_dir=out_dir)

        playlists.append((label, playlist))
        logging.info("Built playlist '%s' with %d tracks.", label, len(playlist))

    return playlists
