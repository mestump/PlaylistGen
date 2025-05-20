import os
import json
import time
from pathlib import Path
import shelve
import logging
import re
import math
import collections
# Load config for API keys
from .config import load_config

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ---- Mood Heuristic Mapping ----
MOODS = {
    "Happy":      ["happy", "feel good", "cheerful", "uplifting", "good mood"],
    "Sad":        ["sad", "melancholy", "melancholic", "heartbreak", "somber"],
    "Angry":      ["angry", "aggressive", "fierce", "rage"],
    "Chill":      ["chill", "chillout", "mellow", "laid back", "relax", "soothing", "calm"],
    "Energetic":  ["energetic", "high energy", "party", "dance", "upbeat", "fast"],
    "Romantic":   ["romantic", "love song", "ballad"],
    "Epic":       ["epic", "anthemic", "dramatic", "orchestral"],
    "Dreamy":     ["dreamy", "ethereal", "ambient", "spacey"],
    "Groovy":     ["groovy", "funky", "swing"],
    "Nostalgic":  ["nostalgia", "retro", "oldies", "classic"],
}
PRIORITY = ["Happy","Sad","Chill","Energetic","Romantic","Epic","Dreamy","Groovy","Nostalgic"]

# ---- Config / Cache ----
_cfg = load_config()
CACHE_PATH = Path(_cfg.get('TAG_MOOD_CACHE'))
SHELVE_PATH = Path(_cfg.get('CACHE_DB'))
# Last.fm API key: environment variable overrides config
API_KEY = os.getenv("LASTFM_API_KEY") or load_config().get("LASTFM_API_KEY")

def canonical_mood(tags, tag_counts=None):
    scores = collections.defaultdict(float)
    for raw in tags:
        clean = re.sub(r"[^\w\s]", " ", raw.lower()).strip()
        for mood, keys in MOODS.items():
            for k in keys:
                if k in clean:
                    # Weight by inverse tag frequency if available (optional)
                    weight = 1.0
                    if tag_counts and clean in tag_counts:
                        count = tag_counts.get(clean, 1)
                        if count <= 1:
                            weight = 1.0
                        else:
                            weight = 1.0 / math.log10(count)
                    scores[mood] += weight
    if scores:
        best = max(scores, key=lambda m: (scores[m], -PRIORITY.index(m) if m in PRIORITY else 99))
        return best
    return None

def fetch_lastfm_tags(artist, track, api_key, cache_db=None):
    """Returns a list of tags from Last.fm for (artist, track)."""
    if not api_key:
        raise RuntimeError(
            "LASTFM_API_KEY is not set. Please set the environment variable or add it to config.yml"
        )
    key = f"{artist.lower()} - {track.lower()}"
    if cache_db is not None:
        if key in cache_db:
            return cache_db[key]
    url = (
        f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags"
        f"&artist={requests.utils.quote(artist)}"
        f"&track={requests.utils.quote(track)}"
        f"&api_key={api_key}&format=json"
    )
    try:
        resp = requests.get(url, timeout=5)
        tags = []
        data = resp.json()
        for tag in data.get("toptags", {}).get("tag", []):
            tname = tag.get("name", "")
            if tname:
                tags.append(tname)
        if cache_db is not None:
            cache_db[key] = tags
        # Optional throttling:
        # time.sleep(0.1)
        return tags
    except Exception:
        logging.exception(f"Failed to fetch Last.fm tags for {artist} - {track}")
        if cache_db is not None:
            cache_db[key] = []
        return []

def batch_tag_and_mood(track_list, api_key=API_KEY, out_json_path=CACHE_PATH, shelve_path=None):
    """
    For each (artist, track) in track_list:
      - Look up cached tags, else pull from Last.fm
      - Map to mood (using canonical_mood)
      - Save results to disk as JSON and Shelve
    """
    processed = 0
    skipped = 0
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    if shelve_path is None:
        shelve_path = SHELVE_PATH
    Path(shelve_path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress if resuming
    if Path(out_json_path).exists():
        with open(out_json_path, "r", encoding="utf-8") as f:
            mood_db = json.load(f)
    else:
        mood_db = {}

    tag_counts = collections.Counter()
    with shelve.open(str(shelve_path)) as cache_db:
        for artist, track in tqdm(track_list, desc="Fetching tags/moods"):
            track_id = f"{artist} - {track}".strip().lower()
            if track_id in mood_db and mood_db[track_id].get('mood') is not None:
                logging.info(f"Skipping {artist} - {track}: mood already cached")
                skipped += 1
                continue

            logging.info(f"Processing {artist} - {track}")
            processed += 1
            tags = fetch_lastfm_tags(artist, track, api_key, cache_db)
            for t in tags:
                tag_counts[t.lower()] += 1

            mood = canonical_mood(tags, tag_counts)
            mood_db[track_id] = {"tags": tags, "mood": mood}

            # Save every 100 items for crash-resume
            if len(mood_db) % 100 == 0:
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(mood_db, f, indent=2)

        # Save once more at the end
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(mood_db, f, indent=2)
    logging.info(f"Done: {len(mood_db)} tracks written to {out_json_path}")
    logging.info(f"Mood-tagged {processed} tracks; skipped {skipped} tracks")
    return processed, skipped

def load_tag_mood_db(path=CACHE_PATH):
    """Load the tag/mood cache if available, else return empty dict."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_tag_mood_cache(itunes_json_path, spotify_dir, tag_mood_path=CACHE_PATH):
    """
    Scan all tracks in the provided iTunes JSON and Spotify directory,
    fetch Last.fm tags for each, and save them to the mood/tag cache as JSON.
    """
    Path(tag_mood_path).parent.mkdir(parents=True, exist_ok=True)
    tracks = []
    # Load all tracks from iTunes JSON
    if itunes_json_path and Path(itunes_json_path).exists():
        with open(itunes_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for t in data.get('tracks', []):
            artist = t.get('Artist')
            name = t.get('Name')
            if artist and name:
                tracks.append((artist, name))

    # Add all tracks from Spotify play history
    if spotify_dir:
        from glob import glob
        import os
        for file in glob(os.path.join(spotify_dir, "*.json")):
            with open(file, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    artist = entry.get('master_metadata_album_artist_name')
                    title = entry.get('master_metadata_track_name')
                    if artist and title:
                        tracks.append((artist, title))

    tracks = list(set(tracks))  # deduplicate

    logging.info(f"Fetching tags for {len(tracks)} tracks. This can take a while...")
    processed, skipped = batch_tag_and_mood(
        tracks,
        api_key=API_KEY,
        out_json_path=tag_mood_path,
        shelve_path=_cfg.get('CACHE_DB', None)
    )
    logging.info(f"Mood-tagged {processed} tracks; skipped {skipped} tracks")

    # Sanity check
    if Path(tag_mood_path).exists():
        with open(tag_mood_path, "r", encoding="utf-8") as f:
            tag_db = json.load(f)
        logging.info(f"Tag DB now contains {len(tag_db)} tracks. Example entry: {next(iter(tag_db.items())) if tag_db else 'None'}")
    else:
        logging.error(f"Tag mood DB file {tag_mood_path} was NOT written!")


