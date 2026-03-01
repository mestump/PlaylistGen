"""
Spotify user taste profile builder for PlaylistGen.

Reads Spotify streaming-history JSON exports (from the Spotify Data Download
request) and builds a preference profile with:

  artist_scores      — Total ms listened per artist
  genre_scores       — Normalised genre affinity derived from Last.fm tag history
  tag_scores         — Raw Last.fm tag counts from listened tracks
  mood_scores        — Canonical mood counts from listened tracks
  year_scores        — Listening year distribution (year → play count)
  track_play_counts  — Per-track play count
  track_skip_counts  — Per-track skip count
  generated_at       — ISO timestamp
"""

import datetime
import json
import logging
from collections import Counter
from pathlib import Path

from .config import load_config
from .mood_map import canonical_genre
from .utils import progress_bar

logging.basicConfig(level=logging.INFO)


def _get_track_id(artist: str, track: str) -> str:
    return f"{artist} - {track}".strip().lower()


def build_profile(
    spotify_dir=None,
    tag_mood_path=None,  # kept for backward-compat signature; not used
    out_path=None,
    tag_db: dict = None,
) -> dict:
    """
    Build a user taste profile from Spotify streaming-history JSON files.

    Args:
        spotify_dir:  Directory containing Spotify JSON export files.
                      Falls back to config['SPOTIFY_DIR'].
        tag_db:       Pre-loaded tag database dict ({"artist - track" → [tags]}).
                      If None, the function loads it from the SQLite cache.
        out_path:     Where to write the resulting profile JSON.
                      Falls back to config['PROFILE_PATH'].

    Returns:
        Profile dict (also written to out_path).
        Returns an empty dict (without saving) if no Spotify files are found.
    """
    cfg = load_config()
    spotify_dir = Path(spotify_dir or cfg["SPOTIFY_DIR"])
    out_path = Path(out_path or cfg["PROFILE_PATH"])

    # Load tag DB if not provided (needed for mood/genre enrichment)
    if tag_db is None:
        from .tag_mood_service import load_tag_mood_db
        tag_db = load_tag_mood_db()

    files = list(spotify_dir.rglob("*.json"))
    if not files:
        logging.warning(
            "No Spotify JSON files found in %s — "
            "personalization will be disabled (scoring by play count and mood only).",
            spotify_dir,
        )
        return {}

    artist_scores: Counter = Counter()
    mood_scores: Counter = Counter()
    tag_scores: Counter = Counter()
    year_scores: Counter = Counter()
    track_play_counts: Counter = Counter()
    track_skip_counts: Counter = Counter()

    for fpath in files:
        logging.info("Processing Spotify log: %s", fpath.name)
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Failed to load %s: %s", fpath.name, exc)
            continue

        for entry in progress_bar(data, desc=f"Parsing {fpath.name}"):
            artist = entry.get("master_metadata_album_artist_name")
            track = entry.get("master_metadata_track_name")
            if not artist or not track:
                continue

            ms_played: int = entry.get("ms_played", 0)
            skipped: bool = bool(entry.get("skipped", False))
            ts: str = entry.get("ts", "")

            track_id = _get_track_id(artist, track)
            tags = tag_db.get(track_id, [])
            # Handle legacy dict format {"tags": [...], "mood": "..."}
            if isinstance(tags, dict):
                tags = tags.get("tags", [])

            # Derive mood from cached tags + mood_map
            from .mood_map import canonical_mood
            mood = canonical_mood(tags) if tags else None

            artist_scores[artist] += ms_played
            track_play_counts[track_id] += 1
            if skipped:
                track_skip_counts[track_id] += 1

            if mood:
                mood_scores[mood] += 1
            for t in tags:
                tag_scores[t.lower()] += 1

            if ts:
                try:
                    year = datetime.datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    ).year
                    year_scores[year] += 1
                except Exception:
                    pass

    # --- Derive genre_scores from tag_scores via canonical_genre() ---
    # tag_scores contains raw Last.fm tag counts (e.g. {"rock": 120, "indie": 80}).
    # We map each tag to a normalised iTunes-style genre and aggregate.
    genre_scores: Counter = Counter()
    for tag, count in tag_scores.items():
        g = canonical_genre(tag)
        if g:
            genre_scores[g.lower()] += count  # lowercase to match scoring lookups

    profile = {
        "artist_scores": dict(artist_scores.most_common()),
        "genre_scores": dict(genre_scores.most_common()),   # FIXED: was always {}
        "tag_scores": dict(tag_scores.most_common()),
        "mood_scores": dict(mood_scores.most_common()),
        "year_scores": {str(y): c for y, c in sorted(year_scores.items())},
        "track_play_counts": dict(track_play_counts),
        "track_skip_counts": dict(track_skip_counts),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logging.info(
        "Saved taste profile to %s  (artists: %d, genres: %d, moods: %d)",
        out_path,
        len(artist_scores),
        len(genre_scores),
        len(mood_scores),
    )
    return profile


def load_profile(path=None) -> dict:
    """
    Load a saved taste profile from disk.
    Returns an empty dict if the file does not exist (Spotify data is optional).
    """
    cfg = load_config()
    p = Path(path or cfg["PROFILE_PATH"])
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    logging.info(
        "No taste profile found at %s — running without personalization.", p
    )
    return {}
