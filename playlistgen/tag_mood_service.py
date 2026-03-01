"""
tag_mood_service — backward-compatibility shim.

All functionality has been moved to:
  lastfm_client.py  — API fetch + SQLite cache + rate limiting
  mood_map.py       — Mood keyword mapping and canonical_mood()

Existing call sites continue to work unchanged.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .config import load_config
from .lastfm_client import (
    generate_tag_cache,
    load_tag_db_from_sqlite,
    init_cache_db,
    fetch_track_tags,
    migrate_json_to_sqlite,
)
from .mood_map import MOODS, PRIORITY, canonical_mood, build_tag_counts

# Re-export so any code doing `from .tag_mood_service import canonical_mood` still works
__all__ = [
    "MOODS",
    "PRIORITY",
    "canonical_mood",
    "fetch_lastfm_tags",
    "batch_tag_and_mood",
    "load_tag_mood_db",
    "generate_tag_mood_cache",
]

_cfg = load_config()
_API_KEY = _cfg.get("LASTFM_API_KEY")
_CACHE_DB = _cfg.get("LASTFM_CACHE_DB") or _cfg.get("CACHE_DB")
_TAG_MOOD_CACHE = _cfg.get("TAG_MOOD_CACHE")


def fetch_lastfm_tags(
    artist: str,
    track: str,
    api_key: Optional[str] = None,
    cache_db=None,  # ignored — kept for signature compat
) -> list:
    """
    Fetch Last.fm tags for (artist, track).
    Shim over lastfm_client.fetch_track_tags().
    """
    key = api_key or _API_KEY
    if not key:
        return []
    db_path = _CACHE_DB or str(Path.home() / ".playlistgen" / "lastfm.sqlite")
    conn = init_cache_db(db_path)
    tags = fetch_track_tags(artist, track, key, conn)
    conn.close()
    return tags


def batch_tag_and_mood(
    track_list,
    api_key=None,
    out_json_path=None,
    shelve_path=None,  # ignored — kept for signature compat
):
    """
    Fetch tags + compute moods for all (artist, track) pairs.
    Shim that delegates to lastfm_client.generate_tag_cache().
    Returns (processed_count, 0).
    """
    key = api_key or _API_KEY
    db_path = _CACHE_DB or str(Path.home() / ".playlistgen" / "lastfm.sqlite")
    legacy = out_json_path or _TAG_MOOD_CACHE

    tag_db = generate_tag_cache(
        list(track_list),
        api_key=key or "",
        db_path=db_path,
        json_legacy_path=legacy,
    )
    return len(tag_db), 0


def load_tag_mood_db(path=None) -> dict:
    """
    Load the tag/mood database.

    Tries the SQLite cache first; falls back to the legacy JSON file.
    Returns a dict mapping "artist - track" → List[str] of tags.
    """
    db_path = _CACHE_DB or str(Path.home() / ".playlistgen" / "lastfm.sqlite")

    # Prefer SQLite
    if Path(db_path).exists():
        return load_tag_db_from_sqlite(db_path)

    # Fall back to old JSON cache and migrate it
    json_path = path or _TAG_MOOD_CACHE
    if json_path and Path(json_path).exists():
        conn = init_cache_db(db_path)
        migrate_json_to_sqlite(str(json_path), conn)
        conn.close()
        return load_tag_db_from_sqlite(db_path)

    return {}


def generate_tag_mood_cache(
    itunes_json_path, spotify_dir, tag_mood_path=None
):
    """
    Scan iTunes JSON + Spotify history, fetch Last.fm tags for all unique tracks.
    Shim over lastfm_client.generate_tag_cache().
    """
    import os
    from glob import glob

    tracks = []

    if itunes_json_path and Path(itunes_json_path).exists():
        with open(itunes_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get("tracks", []):
            a = t.get("Artist")
            n = t.get("Name")
            if a and n:
                tracks.append((a, n))

    if spotify_dir:
        for file in glob(os.path.join(str(spotify_dir), "*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    for entry in json.load(f):
                        a = entry.get("master_metadata_album_artist_name")
                        n = entry.get("master_metadata_track_name")
                        if a and n:
                            tracks.append((a, n))
            except Exception as exc:
                logging.warning("Could not read Spotify file %s: %s", file, exc)

    key = _API_KEY
    if not key:
        logging.warning(
            "LASTFM_API_KEY not set — skipping tag cache generation. "
            "Set it in config.yml or the environment."
        )
        return

    db_path = _CACHE_DB or str(Path.home() / ".playlistgen" / "lastfm.sqlite")
    legacy = tag_mood_path or _TAG_MOOD_CACHE

    logging.info("Fetching Last.fm tags for %d tracks…", len(set(tracks)))
    generate_tag_cache(
        tracks,
        api_key=key,
        db_path=db_path,
        json_legacy_path=legacy,
    )
