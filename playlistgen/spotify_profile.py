import json
import datetime
import logging
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from .tag_mood_service import load_tag_mood_db
from .config import load_config

logging.basicConfig(level=logging.INFO)

def build_profile(spotify_dir=None, tag_mood_path=None, out_path=None) -> dict:
    """
    Build a user taste profile from Spotify JSON logs, using mood/tag data.
    """
    cfg = load_config()
    spotify_dir = Path(spotify_dir or cfg['SPOTIFY_DIR'])
    tag_mood_path = Path(tag_mood_path or cfg.get('TAG_MOOD_CACHE', Path.home() / '.playlistgen' / 'lastfm_tags_cache.json'))
    out_path = Path(out_path or cfg['PROFILE_PATH'])

    tag_mood_db = load_tag_mood_db(tag_mood_path)
    files = list(spotify_dir.rglob('*.json'))
    if not files:
        logging.error(f"No Spotify JSON files found in {spotify_dir}")
        return {}

    artist_scores = Counter()
    mood_scores = Counter()
    tag_scores = Counter()
    year_scores = Counter()
    track_play_counts = Counter()
    track_skip_counts = Counter()

    def get_track_id(artist, track):
        return f"{artist} - {track}".strip().lower()

    for f in files:
        logging.info(f"Processing Spotify log: {f.name}")
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            logging.warning(f"Failed to load {f.name}: {e}")
            continue

        for entry in tqdm(data, desc=f"Entries in {f.name}"):
            artist = entry.get('master_metadata_album_artist_name')
            track = entry.get('master_metadata_track_name')
            if not artist or not track:
                continue

            ms_played = entry.get('ms_played', 0)
            skipped = entry.get('skipped', False)
            ts = entry.get('ts')

            track_id = get_track_id(artist, track)
            tag_mood = tag_mood_db.get(track_id, {})
            mood = tag_mood.get("mood")
            tags = tag_mood.get("tags", [])

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
                    year = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).year
                    year_scores[year] += 1
                except Exception:
                    pass

    profile = {
        "artist_scores": dict(artist_scores.most_common()),
        "tag_scores": dict(tag_scores.most_common()),   # <-- these are the Last.fm tag counts
        "mood_scores": dict(mood_scores.most_common()),
        "year_scores": dict(sorted(year_scores.items())),
        "track_play_counts": dict(track_play_counts),
        "track_skip_counts": dict(track_skip_counts),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logging.info(f"Saved taste profile to {out_path}")

    return profile

def load_profile(path=None):
    cfg = load_config()
    path = Path(path or cfg['PROFILE_PATH'])
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Taste profile not found: {path}")
