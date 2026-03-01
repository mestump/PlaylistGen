"""
Unified Last.fm API client for PlaylistGen.

Supersedes tag_mood_service.py (Last.fm calls) and genre_service.py.

Design:
- SQLite cache (WAL mode) — no file-locking issues like shelve.
- Module-level rate limiter: enforces LASTFM_RATE_LIMIT_MS between calls.
- Retry once on HTTP 429 / 5xx with a 2 s sleep before the second attempt.
- Stores only raw tag lists; mood classification happens at score time via
  mood_map.canonical_mood(), so keyword changes never require a cache rebuild.
- One-time migration from the old JSON cache format on first run.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logging.warning("requests not installed — Last.fm tag fetching disabled.")

from .utils import progress_bar

_LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
_last_call_time: float = 0.0  # module-level timestamp of last API call


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def init_cache_db(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite tag cache at db_path.
    Returns an open connection with WAL journaling enabled.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tag_cache (
            key        TEXT PRIMARY KEY,
            tags_json  TEXT NOT NULL,
            fetched_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _get_cached(conn: sqlite3.Connection, key: str) -> Optional[List[str]]:
    """Return cached tag list for key, or None if not yet cached."""
    row = conn.execute(
        "SELECT tags_json FROM tag_cache WHERE key = ?", (key,)
    ).fetchone()
    if row is not None:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return []
    return None


def _set_cached(conn: sqlite3.Connection, key: str, tags: List[str]) -> None:
    """Upsert a tag list into the cache."""
    conn.execute(
        "INSERT OR REPLACE INTO tag_cache (key, tags_json, fetched_at) VALUES (?, ?, ?)",
        (key, json.dumps(tags), int(time.time())),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


def fetch_track_tags(
    artist: str,
    track: str,
    api_key: str,
    conn: sqlite3.Connection,
    rate_limit_ms: int = 200,
) -> List[str]:
    """
    Fetch Last.fm top tags for (artist, track).

    - Checks SQLite cache first; returns immediately on hit.
    - Enforces rate_limit_ms delay between successive API calls.
    - Retries once on HTTP 429 / 5xx (2 s sleep before retry).
    - Caches an empty list on second failure so we don't hammer the API.

    Returns a (possibly empty) list of tag name strings.
    """
    global _last_call_time

    if not api_key or not _REQUESTS_AVAILABLE:
        return []

    key = f"{artist.lower().strip()} - {track.lower().strip()}"
    cached = _get_cached(conn, key)
    if cached is not None:
        return cached

    # Rate limiting
    elapsed = time.time() - _last_call_time
    wait = (rate_limit_ms / 1000.0) - elapsed
    if wait > 0:
        time.sleep(wait)

    def _do_request() -> List[str]:
        params = {
            "method": "track.gettoptags",
            "artist": artist,
            "track": track,
            "api_key": api_key,
            "format": "json",
            "autocorrect": "1",
        }
        resp = _requests.get(_LASTFM_BASE, params=params, timeout=8)
        _last_call_time = time.time()
        if resp.status_code == 200:
            data = resp.json()
            return [
                t["name"]
                for t in data.get("toptags", {}).get("tag", [])
                if t.get("name")
            ]
        if resp.status_code in (429, 500, 502, 503, 504):
            raise RuntimeError(f"HTTP {resp.status_code}")
        # 4xx errors (track not found, etc.) — cache empty to avoid retrying
        return []

    try:
        tags = _do_request()
    except Exception as exc:
        logging.debug("Last.fm first attempt failed for %s - %s: %s", artist, track, exc)
        time.sleep(2.0)
        try:
            tags = _do_request()
        except Exception as exc2:
            logging.warning("Last.fm failed for %s - %s: %s", artist, track, exc2)
            tags = []

    _set_cached(conn, key, tags)
    return tags


# ---------------------------------------------------------------------------
# Migration from old JSON cache
# ---------------------------------------------------------------------------


def migrate_json_to_sqlite(json_path: str, conn: sqlite3.Connection) -> int:
    """
    One-time migration: read the old JSON tag cache and import entries into
    SQLite. Skips keys already present in the database.

    Old format:  {"artist - track": {"tags": [...], "mood": "..."}, ...}
    Also handles: {"artist - track": [...], ...}  (raw list format)

    Returns the number of new entries migrated.
    """
    p = Path(json_path)
    if not p.exists():
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            old_db = json.load(f)
    except Exception as exc:
        logging.warning("Could not read old tag cache %s: %s", json_path, exc)
        return 0

    migrated = 0
    for key, val in old_db.items():
        if isinstance(val, list):
            tags = val
        elif isinstance(val, dict):
            tags = val.get("tags", [])
        else:
            continue
        if _get_cached(conn, key) is None:
            _set_cached(conn, key, tags)
            migrated += 1

    if migrated:
        logging.info(
            "Migrated %d entries from %s to SQLite cache.", migrated, json_path
        )
    return migrated


# ---------------------------------------------------------------------------
# High-level batch API
# ---------------------------------------------------------------------------


def generate_tag_cache(
    track_list: List[tuple],
    api_key: str,
    db_path: str,
    json_legacy_path: Optional[str] = None,
    rate_limit_ms: int = 200,
) -> Dict[str, List[str]]:
    """
    Fetch Last.fm tags for all (artist, track) pairs in track_list.

    - Opens (or creates) the SQLite cache at db_path.
    - Optionally migrates the old JSON cache first.
    - Skips pairs already cached.
    - Returns the full tag_db as a dict: {"artist - track" -> [tag, ...]}

    Args:
        track_list:       List of (artist, track) tuples.
        api_key:          Last.fm API key.
        db_path:          Path to the SQLite cache file.
        json_legacy_path: Optional path to an old JSON cache to migrate.
        rate_limit_ms:    Minimum milliseconds between API calls.
    """
    conn = init_cache_db(db_path)

    if json_legacy_path:
        migrate_json_to_sqlite(json_legacy_path, conn)

    unique_tracks = list(
        {(a.strip(), t.strip()) for a, t in track_list if a and t}
    )
    logging.info(
        "Fetching Last.fm tags for %d unique tracks (cached ones skipped)...",
        len(unique_tracks),
    )

    for artist, track in progress_bar(unique_tracks, desc="Last.fm tags"):
        fetch_track_tags(artist, track, api_key, conn, rate_limit_ms)

    conn.close()
    return load_tag_db_from_sqlite(db_path)


def load_tag_db_from_sqlite(db_path: str) -> Dict[str, List[str]]:
    """
    Load all cached tag entries from SQLite into a plain dict.

    Returns: {"artist - track" -> [tag, ...]}
    """
    p = Path(db_path)
    if not p.exists():
        return {}
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT key, tags_json FROM tag_cache").fetchall()
    conn.close()
    result: Dict[str, List[str]] = {}
    for key, tags_json in rows:
        try:
            result[key] = json.loads(tags_json)
        except json.JSONDecodeError:
            result[key] = []
    return result
