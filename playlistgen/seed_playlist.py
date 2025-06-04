# Utilities for generating playlists from a seed song using Last.fm similarity
import logging
from typing import List, Tuple, Optional

import pandas as pd

from .config import load_config
from .itunes import load_itunes_json, build_library_from_dir
from .scoring import score_tracks
from .tag_mood_service import load_tag_mood_db
from .playlist_builder import build_playlists
from .spotify_profile import load_profile

try:
    import requests
except ImportError:  # pragma: no cover - requests is required in runtime
    requests = None


def fetch_similar_tracks(
    artist: str, track: str, api_key: str, limit: int = 20
) -> List[Tuple[str, str]]:
    """Return a list of (artist, track) tuples similar to the given song."""
    if not requests:
        raise RuntimeError("The requests package is required")
    url = (
        "https://ws.audioscrobbler.com/2.0/?method=track.getsimilar"
        f"&artist={requests.utils.quote(artist)}"
        f"&track={requests.utils.quote(track)}"
        f"&api_key={api_key}&format=json&limit={limit}"
    )
    logging.info("Fetching similar tracks from Last.fm")
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        tracks = []
        for t in data.get("similartracks", {}).get("track", []):
            a = t.get("artist", {}).get("name")
            n = t.get("name")
            if a and n:
                tracks.append((a, n))
        return tracks
    except Exception:
        logging.exception("Failed to fetch similar tracks")
        return []


def generate_seed_playlist(
    artist: str,
    track: str,
    library_df: pd.DataFrame,
    profile: Optional[dict] = None,
    tag_mood_db: Optional[dict] = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Generate a playlist seeded from the given song limited to the library."""
    cfg = load_config()
    api_key = cfg.get("LASTFM_API_KEY")
    similar = fetch_similar_tracks(artist, track, api_key, limit=limit * 2)

    matches = []
    for a, n in similar:
        mask = (
            library_df["Artist"].str.lower() == a.lower()
        ) & (library_df["Name"].str.lower() == n.lower())
        rows = library_df[mask]
        if not rows.empty:
            matches.append(rows.iloc[0])
        if len(matches) >= limit:
            break

    if not matches:
        logging.warning("No similar tracks found in library")
        return pd.DataFrame()

    df = pd.DataFrame(matches)
    scored = score_tracks(df, config=profile or {}, tag_mood_db=tag_mood_db)
    scored = scored.sort_values("Score", ascending=False).head(limit)
    scored = scored.drop_duplicates(subset=["Artist", "Name"]).reset_index(drop=True)
    return scored


def build_seed_playlist(
    song: str,
    cfg: Optional[dict] = None,
    library_dir: str = None,
    limit: int = 20,
):
    """High level helper used by the CLI."""
    if cfg is None:
        cfg = load_config()
    if " - " in song:
        artist, track = [s.strip() for s in song.split(" - ", 1)]
    else:
        parts = song.split()
        artist, track = parts[0], " ".join(parts[1:])

    if library_dir:
        library_df = build_library_from_dir(library_dir)
    else:
        from .pipeline import ensure_itunes_json

        itunes_json = ensure_itunes_json(cfg)
        library_df = load_itunes_json(str(itunes_json))

    tag_db = load_tag_mood_db(cfg.get("TAG_MOOD_CACHE"))
    profile = load_profile(cfg.get("PROFILE_PATH"))

    playlist_df = generate_seed_playlist(
        artist,
        track,
        library_df,
        profile=profile,
        tag_mood_db=tag_db,
        limit=limit,
    )
    if playlist_df.empty:
        return None

    label = f"Seed Mix - {artist} - {track}"
    build_playlists([playlist_df], library_df, tracks_per_mix=limit, name_fn=lambda *_: label)
    return playlist_df
